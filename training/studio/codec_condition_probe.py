"""B2: one independent E(D(z0)) condition, not training or a new default.

Only the initial clean latent changes. Reuse the pinned six-second R005 A
records; no CFG cache, endpoint latent slicing, extra noise, or lossy feedback.
The Creator launcher owns authorization, shared GPU locks, and the watchdog.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, write_json
from training.demo.latency_engine import ReusableActionEngine
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, annotate_report, load_negative
from training.studio.timestep_precision_probe import EXPECTED_INPUTS

REVISION = 'R005-one-codec-condition-roundtrip-original6s-v1'
CASE = dict(name='codec_once', mode='segment6', **SETTINGS)
QUALITY_ROOT = DATA_ROOT / 'creator-runtime/quality-20260928'
SCENES = {
    'mountain': dict(source_job=str(R005_SOURCE_JOB),
        baseline_case=str(QUALITY_ROOT / 'quality-cfg5/text_cfg5'),
        original_first_latent_sha256_float32=EXPECTED_INPUTS['first_latent_sha256_float32']),
    'grass': dict(source_job=str(QUALITY_ROOT / 'grassland-source'),
        baseline_case=str(QUALITY_ROOT / 'quality-grassland-cfg5/grass_text_cfg5'),
        original_first_latent_sha256_float32='93ce16854c998b146f385d445908e116050e67a1e095798aaa25f4771fc8b915'),
}
FIXED = dict(mode='codec_condition_probe', max_seconds=600, checkpoint_sha256=R005_SHA,
    negative_cache=DEFAULT_NEGATIVE_CACHE, residency='co_resident', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=96, codec_cycles=1,
    noise_sha256_float32=EXPECTED_INPUTS['noise_sha256_float32'], **SETTINGS)
POLICY = dict(transform='z0_rt = E(D(z0)); z0 is the independently encoded BF16 initial condition',
    cycles=1, single_rgb_frame=True, use_vae_temporal_cache=False, reset_before_and_after_each_codec_call=True,
    decode_dtype='torch.bfloat16', encoder_input_dtype='torch.bfloat16',
    intermediate_rgb='float32 [-1,1] before the existing BF16 encoder cast',
    uint8_or_video_feedback=False, first_video_frame='quantized preview of actual D(z0) condition; never fed back',
    frozen_original_weights=True, future_noise_changed=False, actions_changed=False, prompt_changed=False,
    euler_and_cfg_changed=False, generated_history_training=False)


def validate_request(request):
    require(type(request) is dict and request.get('scene') in SCENES, 'expected pinned mountain or grass scene')
    fixed = dict(FIXED, **SCENES[request['scene']])
    require(set(request) == set(fixed) | {'scene', 'job_id', 'source_revision', 'experiment_revision'},
            'unknown codec probe fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in fixed.items()),
            'codec probe fixes original R005 / one codec cycle / WJ96 / 40EulerCFG5 / 600seconds')
    require(type(request['job_id']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and type(request['source_revision']) is str and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid codec probe identity')


def validate_baseline(request, *, scene, lineage, negative_source):
    """Small historical records only; live z0/noise binding is checked before denoising."""
    target = contained(request['baseline_case'], DATA_ROOT)
    result = json.loads((target / 'result.json').read_text(encoding='utf-8'))
    inputs = json.loads((target.parent / 'inputs.json').read_text(encoding='utf-8'))
    generation = result.get('generation', {})
    require(result.get('status') == generation.get('outcome') == 'completed'
            and result.get('mode') == 'segment6' and result.get('seed') == 42
            and generation.get('frames_written') == 97 and generation.get('future_frames') == 96
            and generation.get('fps') == 16 and generation.get('width') == WIDTH and generation.get('height') == HEIGHT
            and generation.get('residency') == 'co_resident' and generation.get('fastpath') is False
            and generation.get('actual_model_forward_calls') == 80, 'historical baseline geometry/runtime differs')
    for key, value in SETTINGS.items():
        actual = result.get(key, 'flow_euler' if key == 'solver' else None)
        require(actual == value, f'historical baseline {key} differs')
    require(generation.get('sampler') == dict(solver='flow_euler', steps=40, shift=5.0,
            cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2), 'historical baseline sampler differs')
    require(inputs.get('source_job') == request['source_job']
            and inputs.get('checkpoint_sha256') == lineage['sha256'] == R005_SHA
            and inputs.get('prompt') == scene['prompt'] and inputs.get('scene_id') == scene['scene_id']
            and inputs.get('seed') == scene['seed'] == 42
            and inputs.get('source_actions') == scene['action_segments']
            and generation.get('negative_prompt') == negative_source, 'historical baseline source/text/actions differ')
    require(generation.get('first_latent_sha256_float32') == request['original_first_latent_sha256_float32']
            and generation.get('noise_sha256_float32') == request['noise_sha256_float32'],
            'historical baseline first latent/noise differs')
    require((target / 'raw.mp4').is_file(), 'historical baseline video missing; do not silently substitute')
    return dict(case=str(target), raw_video=str(target / 'raw.mp4'), result=str(target / 'result.json'),
        original_first_latent_sha256_float32=generation['first_latent_sha256_float32'],
        noise_sha256_float32=generation['noise_sha256_float32'],
        elapsed_seconds=result['elapsed_seconds'], historical_A_reused=True,
        timing_comparison='historical not interleaved; this test diagnoses quality, not speed',
        live_original_latent_and_noise_verification='required before sampling')


def codec_once(vae, first, *, device, synchronize):
    """Decode an independent one-frame latent and encode float RGB with fresh state.

    This deliberately rejects arbitrary tail temporal slots or multi-frame input.
    The caller supplies z0 produced by encoding one RGB frame, not a video slice.
    CPU device is only for unit fixtures, never an actual-model CPU fallback.
    """
    import torch
    require(torch.is_tensor(first) and tuple(first.shape) == (1, 1, 48, HEIGHT // 16, WIDTH // 16)
            and first.device.type == 'cpu' and first.dtype == torch.bfloat16
            and bool(torch.isfinite(first).all()), 'one independent finite CPU BF16 initial latent required')
    timings = []

    def call(name, operation):
        vae.model.clear_cache()
        synchronize()
        tick, status = time.monotonic(), 'failed'
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext():
                value = operation()
            synchronize()
            status = 'completed'
            return value
        finally:
            vae.model.clear_cache()
            timings.append(dict(stage=name, seconds=time.monotonic() - tick, status=status))

    decoded = call('codec_decode_single_initial', lambda: vae.decode_to_pixel(
        first.to(device=device, dtype=torch.bfloat16), use_cache=False, return_in_cpu=True))
    require(tuple(decoded.shape) == (1, 1, 3, HEIGHT, WIDTH) and decoded.device.type == 'cpu'
            and decoded.is_floating_point() and bool(torch.isfinite(decoded).all())
            and decoded.min() >= -1 and decoded.max() <= 1, 'single-frame float RGB decoder geometry/range differs')
    pixels = decoded.permute(0, 2, 1, 3, 4).detach().float().contiguous().clone()
    encoded = call('codec_encode_reconstructed_initial', lambda: vae.encode_to_latent(
        pixels.to(device=device, dtype=torch.bfloat16)))
    require(tuple(encoded.shape) == tuple(first.shape) and encoded.is_floating_point()
            and bool(torch.isfinite(encoded).all()), 'reencoded single condition latent differs')
    return encoded.detach().to(device='cpu', dtype=torch.bfloat16).clone(), pixels, timings


def preview_rgb(pixels):
    return pixels[0, :, 0].permute(1, 2, 0).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()


class CodecConditionRuntime(QualityRuntime):
    def configure_probe(self, request, target):
        self.configure(CASE)
        self.probe_request, self.probe_target = request, Path(target)
        self.condition_record, self.condition_pixels, self.original_first = None, None, None
        self.condition_first, self._encode_started, self._generate_started = None, False, False

    def encode(self, pixels):
        import numpy as np
        from PIL import Image
        from training.demo.action_backend import tensor_sha
        torch = self.torch
        require(not self._encode_started and tuple(pixels.shape) == (1, 3, 1, HEIGHT, WIDTH),
                'one independent initial RGB encoding only; no retry or video-tail slicing')
        self._encode_started = True
        self.original_first = super().encode(pixels).to(dtype=torch.bfloat16).clone()
        original_digest = tensor_sha(self.original_first)
        require(original_digest == self.probe_request['original_first_latent_sha256_float32'],
                'live original z0 differs from historical baseline; no candidate generation')
        self.condition_first, self.condition_pixels, timings = codec_once(
            self.vae, self.original_first, device='cuda', synchronize=self.synchronize)
        np.save(self.probe_target / 'original-condition-float.npy', pixels.detach().cpu().float().numpy(), allow_pickle=False)
        np.save(self.probe_target / 'codec-condition-float.npy', self.condition_pixels.numpy(), allow_pickle=False)
        torch.save(dict(original_first=self.original_first, codec_first=self.condition_first),
                   self.probe_target / 'condition-latents.pt')
        Image.fromarray(preview_rgb(pixels.detach().cpu().float())).save(self.probe_target / 'original-condition.png')
        Image.fromarray(preview_rgb(self.condition_pixels)).save(self.probe_target / 'codec-condition.png')
        self.condition_record = dict(policy=copy.deepcopy(POLICY), codec_timings=timings,
            original_first_latent_sha256_float32=original_digest,
            actual_first_latent_sha256_float32=tensor_sha(self.condition_first),
            latent_shape=list(self.condition_first.shape), latent_dtype=str(self.condition_first.dtype),
            actual_encoder_condition_float=str(self.probe_target / 'codec-condition-float.npy'),
            actual_encoder_condition_preview=str(self.probe_target / 'codec-condition.png'),
            original_encoder_condition_float=str(self.probe_target / 'original-condition-float.npy'),
            condition_latents=str(self.probe_target / 'condition-latents.pt'),
            pixel_quantization_only_for_saved_png_and_video_display=True,
            actual_condition_equal_to_original=bool(torch.equal(self.original_first, self.condition_first)),
            original_encode_calls=1, added_encode_calls=1, added_decode_calls=1)
        write_json(self.probe_target / 'condition.json', self.condition_record)
        return self.condition_first

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        from training.demo.action_backend import tensor_sha
        torch = self.torch
        require(not self._generate_started and self.condition_record is not None
                and steps == 40 and fastpath is False and torch.equal(first, self.condition_first),
                'single exact CFG candidate with transformed first latent required; no retry')
        self._generate_started = True
        expected_keys = torch.zeros_like(keys)
        expected_keys[:, :, [0, 5]] = 1
        require(tuple(keys.shape) == (1, 96, 8) and torch.equal(keys, expected_keys)
                and tensor_sha(noise) == self.probe_request['noise_sha256_float32'],
                'fixed WJ96 and original future noise required')
        bindings = dict(prompt_sha256_float32=tensor_sha(prompt), actions_sha256_float32=tensor_sha(keys),
                        noise_sha256_float32=tensor_sha(noise), first_latent_sha256_float32=tensor_sha(first))
        self.condition_record['actual_generation_inputs'] = bindings
        write_json(self.probe_target / 'condition.json', self.condition_record)
        callback = getattr(self, 'phase_callback', None)
        if callback:
            callback('denoising_exact_CFG5')
        latent = super().generate(first, prompt, keys, noise, steps, fastpath=False)
        require(bindings == dict(prompt_sha256_float32=tensor_sha(prompt), actions_sha256_float32=tensor_sha(keys),
            noise_sha256_float32=tensor_sha(noise), first_latent_sha256_float32=tensor_sha(first)),
            'sampling mutated caller-owned input tensors')
        require(self.forward_calls == 80, 'codec probe must keep all 80 exact CFG model forwards')
        return latent


class CodecConditionEngine(ReusableActionEngine):
    def _segment_rollout(self, **options):
        require(self._runtime.condition_pixels is not None, 'actual codec condition missing')
        options['initial_rgb'] = preview_rgb(self._runtime.condition_pixels)
        report = super()._segment_rollout(**options)
        report['displayed_initial_frame'] = 'D(z0) quantized for display only; inference used unquantized float RGB'
        report['initial_condition'] = 'one_independent_codec_roundtrip_E_D_E_original_RGB'
        return report


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', CASE['name'])),
            'new codec probe output required')
    began, runtime, engine, cold = time.monotonic(), None, None, 0.
    def check():
        require(time.monotonic() - began < 570, 'codec probe finite deadline; no retry')
        validate_active_authority(lease)
    check()
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'original R005 source required')
    selected = actions[:96].copy()
    expected = np.zeros((96, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(selected.dtype == np.float32 and np.array_equal(selected, expected), 'original WJ96 required')
    negative, negative_source = load_negative(request['negative_cache'], config.base_model_path)
    baseline = validate_baseline(request, scene=scene, lineage=lineage, negative_source=negative_source)
    inputs = dict(mode=FIXED['mode'], scene=request['scene'], scene_id=scene['scene_id'], prompt=scene['prompt'],
        prompt_condition=lineage['prompt_condition'], source_job=request['source_job'], source_actions=scene['action_segments'],
        actual_actions=[dict(frames=96, keys=['W', 'J'])], checkpoint=lineage['path'], checkpoint_sha256=R005_SHA,
        checkpoint_step=1040, seed=42, width=WIDTH, height=HEIGHT, residency='co_resident', fastpath=False,
        cases=[CASE], negative_prompt=negative_source, source_revision=request['source_revision'],
        experiment_revision=REVISION, codec_policy=POLICY, historical_baseline=baseline,
        training_launched=False, default_changed=False, quality_improvement_verified=False)
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    target = directory / CASE['name']
    target.mkdir(exist_ok=False)
    def progress(phase):
        write_json(directory / 'progress.json', dict(status='running', phase=phase, scene=request['scene']))
    progress('loading')
    try:
        check()
        tick = time.monotonic()
        runtime = CodecConditionRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        runtime.configure_probe(request, target)
        runtime.phase_callback = progress
        engine = CodecConditionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        check()
        progress('single_codec_condition_then_exact_generation')
        started = time.monotonic()
        report, endpoint = engine.run_segment(initial=initial, prompt=prompt, actions=selected, seed=42,
            output_path=target / 'raw.mp4', steps=40)
        del endpoint
        annotate_report(report, CASE, runtime, negative_source, cold)
        report['codec_condition'] = copy.deepcopy(runtime.condition_record)
        require(report['frames_written'] == 97 and runtime.forward_calls == 80
                and report['noise_sha256_float32'] == request['noise_sha256_float32']
                and report['first_latent_sha256_float32'] == runtime.condition_record['actual_first_latent_sha256_float32'],
                'actual output frame/input/forward contract differs')
        entry = dict(**CASE, status='completed', seed=42, generation=report, raw_video=str(target / 'raw.mp4'),
            elapsed_seconds=time.monotonic() - started, elapsed_seconds_scope='case_run_includes_codec_preparation_excludes_cold_load',
            historical_baseline=baseline, quality_evaluation='pending_visual_comparison', quality_improvement_verified=False)
        write_json(target / 'result.json', entry)
    except BaseException as error:
        report = copy.deepcopy(engine.last_run_report or {}) if engine else {}
        if runtime is not None:
            annotate_report(report, CASE, runtime, negative_source, cold)
            report['codec_condition'] = copy.deepcopy(getattr(runtime, 'condition_record', None))
        entry = dict(**CASE, status='failed', generation=report, error=f'{type(error).__name__}: {error}', automatic_retry=False)
        write_json(target / 'result.json', entry)
        write_json(directory / 'progress.json', dict(status='failed', cases=[entry]))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode=FIXED['mode'], scene=request['scene'], cases=[entry], load_seconds=cold,
        elapsed_seconds=time.monotonic() - began, default_changed=False, original_demo_changed=False,
        training_launched=False, quality_improvement_verified=False, automatic_retry=False, realtime=False)
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision, scene):
    require(scene in SCENES, 'choose mountain or grass')
    directory = contained(output, DATA_ROOT)
    request = dict(copy.deepcopy(FIXED), **SCENES[scene], scene=scene, job_id=directory.name,
                   source_revision=source_revision, experiment_revision=REVISION)
    validate_request(request)
    require(not directory.exists(), 'new output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_request_prepared', request=str(directory / 'request.json'), gpu_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--scene', choices=tuple(SCENES), required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
