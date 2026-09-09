"""CPU tests of the actual streaming method without importing CUDA-only models."""

import ast
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


def _streaming_method():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "causal_inference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "CausalInferencePipeline")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "generate_next_block")
    namespace = {"torch": torch, "time": time}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["generate_next_block"]


GENERATE = _streaming_method()


def _pipeline(steps=4, solver="flow_euler", first_frame=None):
    # Nonuniform shifted timesteps ensure the selected intervals, rather than
    # a fixed step size or scheduler-table neighbors, determine the trajectory.
    raw_sigma = torch.linspace(1.0, 1.0 / steps, steps)
    timesteps = 1000.0 * (5.0 * raw_sigma / (1.0 + 4.0 * raw_sigma))
    args = SimpleNamespace(context_noise=0)
    if solver is not None:
        args.streaming_solver = solver
    return SimpleNamespace(
        args=args, denoising_step_list=timesteps, pyramid_sample_ratio=None,
        conditional_dict={"first_frame_latents": first_frame},
        current_start_frame=0 if first_frame is not None else 1,
        num_input_frames=0, frame_seq_length=390,
        kv_cache1=[{}], crossattn_cache=[{}],
        _stream_block_diffusion_times=[],
        scheduler=SimpleNamespace(add_noise=Mock(side_effect=AssertionError("unexpected renoise"))),
    )


class CausalFlowSolverTests(unittest.TestCase):
    def test_euler_selected_40_and_4_step_trajectories_and_clean_cache(self):
        for steps in (40, 4):
            with self.subTest(steps=steps):
                pipeline = _pipeline(steps)
                noise = torch.full((1, 3, 1, 2, 2), 7.0)
                calls = []

                def generator(**kwargs):
                    value = kwargs["noisy_image_or_video"]
                    calls.append((value.clone(), kwargs["timestep"].clone(),
                                  kwargs["current_start"], torch.is_grad_enabled()))
                    self.assertIs(kwargs["kv_cache"], pipeline.kv_cache1)
                    return torch.full_like(value, 3.0), torch.full_like(value, -999.0)

                pipeline.generator = generator
                with patch.object(torch.cuda, "synchronize"), patch.object(
                    torch, "randn_like", side_effect=AssertionError("Euler must not draw noise")
                ):
                    result = GENERATE(pipeline, noise)

                self.assertEqual(len(calls), steps + 1)
                for index, (value, timestep, start, grad_enabled) in enumerate(calls[:-1]):
                    sigma = pipeline.denoising_step_list[index] / 1000.0
                    expected = 7.0 + (sigma - 1.0) * 3.0
                    torch.testing.assert_close(value, torch.full_like(value, expected))
                    torch.testing.assert_close(
                        timestep, torch.full_like(timestep, pipeline.denoising_step_list[index])
                    )
                    self.assertEqual(start, 390)
                    self.assertFalse(grad_enabled)
                torch.testing.assert_close(result, torch.full_like(result, 4.0))
                torch.testing.assert_close(calls[-1][0], result)
                self.assertTrue(torch.all(calls[-1][1] == 0))
                self.assertEqual(calls[-1][2], 390)
                self.assertEqual(pipeline.current_start_frame, 4)
                self.assertEqual(len(pipeline._stream_block_diffusion_times), 1)
                pipeline.scheduler.add_noise.assert_not_called()

    def test_euler_preserves_clean_first_frame_at_every_call(self):
        first = torch.full((1, 1, 1, 2, 2), 123.0)
        pipeline = _pipeline(first_frame=first)
        calls = []

        def generator(**kwargs):
            value = kwargs["noisy_image_or_video"]
            calls.append((value.clone(), kwargs["timestep"].clone()))
            self.assertTrue(kwargs["replace_first_timestep_and_noise_latents"])
            return torch.full_like(value, 3.0), torch.full_like(value, -999.0)

        pipeline.generator = generator
        with patch.object(torch.cuda, "synchronize"):
            result = GENERATE(pipeline, torch.full((1, 3, 1, 2, 2), 7.0))
        for value, timestep in calls:
            torch.testing.assert_close(value[:, :1], first)
            self.assertEqual(timestep[0, 0].item(), 0)
        torch.testing.assert_close(result[:, :1], first)
        torch.testing.assert_close(result[:, 1:], torch.full_like(result[:, 1:], 4.0))

    def test_default_and_explicit_renoise_keep_legacy_path(self):
        for solver in (None, "renoise"):
            with self.subTest(solver=solver):
                pipeline = _pipeline(solver=solver)
                calls = []

                def generator(**kwargs):
                    value = kwargs["noisy_image_or_video"]
                    calls.append((value.clone(), kwargs["timestep"].clone()))
                    return torch.full_like(value, 99.0), torch.full_like(value, 17.0)

                pipeline.generator = generator
                pipeline.scheduler.add_noise = Mock(side_effect=lambda clean, noise, t: clean + 100)
                with patch.object(torch.cuda, "synchronize"), patch.object(
                    torch, "randn_like", side_effect=lambda value: torch.zeros_like(value)
                ) as random_noise:
                    result = GENERATE(pipeline, torch.full((1, 3, 1, 2, 2), 7.0))
                self.assertEqual(random_noise.call_count, 3)
                self.assertEqual(pipeline.scheduler.add_noise.call_count, 3)
                for index, call in enumerate(pipeline.scheduler.add_noise.call_args_list):
                    clean, noise, timestep = call.args
                    torch.testing.assert_close(clean, torch.full_like(clean, 17.0))
                    self.assertTrue(torch.all(noise == 0))
                    torch.testing.assert_close(
                        timestep, torch.full_like(timestep, pipeline.denoising_step_list[index + 1])
                    )
                torch.testing.assert_close(result, torch.full_like(result, 17.0))
                torch.testing.assert_close(calls[-1][0], result)
                self.assertTrue(torch.all(calls[-1][1] == 0))

    def test_invalid_solver_and_euler_pyramid_rejected_before_generation(self):
        for solver, pyramid, message in (
            ("wrong", None, "unsupported streaming_solver"),
            ("flow_euler", [1.0] * 4, "does not support pyramid"),
        ):
            with self.subTest(solver=solver, pyramid=pyramid):
                pipeline = _pipeline(solver=solver)
                pipeline.pyramid_sample_ratio = pyramid
                pipeline.generator = Mock()
                with self.assertRaisesRegex(ValueError, message):
                    GENERATE(pipeline, torch.zeros(1, 3, 1, 2, 2))
                pipeline.generator.assert_not_called()
                self.assertEqual(pipeline.current_start_frame, 1)


if __name__ == "__main__":
    unittest.main()
