import unittest
from types import SimpleNamespace

import torch

from training.eval.wan_causal_adapter import WanCausalRolloutAdapter


class PrimingTests(unittest.TestCase):
    def make_adapter(self, advance):
        first = torch.zeros(1, 1, 48, 2, 2)
        condition = {'first_frame_latents': first}
        pipeline = SimpleNamespace(
            frame_seq_length=390, conditional_dict=condition,
            kv_cache1=[{'global_end_index': torch.tensor(0)} for _ in range(2)],
            crossattn_cache=[], current_start_frame=0, num_input_frames=1)
        pipeline.reset_stream = lambda **kwargs: None
        pipeline.get_condition_split = lambda value, start, count: value
        calls = []

        def generator(**kwargs):
            calls.append((torch.is_grad_enabled(), kwargs))
            if not torch.is_grad_enabled():
                for cache in pipeline.kv_cache1:
                    cache['global_end_index'].fill_(advance)

        pipeline.generator = generator
        adapter = WanCausalRolloutAdapter(
            pipeline=pipeline, torch_module=torch, device='cpu',
            checkpoint_path='unused', checkpoint_sha256='unused',
            checkpoint_stage='causal_teacher_forcing_v1')
        return adapter, pipeline, first, calls

    def test_first_frame_uses_kv_inference_and_restores_grad(self):
        adapter, pipeline, first, calls = self.make_adapter(390)
        with torch.enable_grad():
            adapter._prime_transformer(first)
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(len(calls), 1)
        grad, arguments = calls[0]
        self.assertFalse(grad)
        self.assertIs(arguments['noisy_image_or_video'], first)
        self.assertEqual(arguments['current_start'], 0)
        self.assertEqual(arguments['timestep'].item(), 0)
        self.assertEqual(adapter._cache_position(), 390)
        self.assertEqual(pipeline.current_start_frame, 1)
        self.assertEqual(pipeline.num_input_frames, 0)
        self.assertIsNone(pipeline.conditional_dict['first_frame_latents'])

    def test_bad_cache_does_not_discard_first_frame(self):
        for advance in (0, 780):
            with self.subTest(advance=advance):
                adapter, pipeline, first, calls = self.make_adapter(advance)
                with torch.enable_grad():
                    with self.assertRaisesRegex(RuntimeError, 'prime exactly one frame'):
                        adapter._prime_transformer(first)
                    self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(pipeline.current_start_frame, 0)
                self.assertEqual(pipeline.num_input_frames, 1)
                self.assertIs(pipeline.conditional_dict['first_frame_latents'], first)


if __name__ == '__main__':
    unittest.main()
