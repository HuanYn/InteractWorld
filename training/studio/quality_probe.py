"""Bounded quality ablations; old Demo defaults and model weights stay unchanged."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import time

from training.demo.contracts import require, write_json
from training.demo.latency_engine import _WanRuntime, ReusableActionEngine

DEFAULT_NEGATIVE_CACHE = '/DATA_DISK/InteractWorld/agent-runtime/continuation-20260925/assets/laboratory-prompts-v2.pt'


def validate_cases(request):
    cases = request.get('cases')
    require(isinstance(cases, list) and 1 <= len(cases) <= 4, 'quality probe requires 1–4 cases')
    result, names = [], set()
    for value in cases:
        require(isinstance(value, dict) and set(value) <=
                {'name', 'steps', 'cfg_scale', 'lora_strength', 'action_scale', 'mode', 'solver'}, 'unknown quality case fields')
        case = dict(steps=40, cfg_scale=1, lora_strength=1, action_scale=.03,
                    mode=request.get('probe_mode', 'segment6'), solver='flow_euler')
        case.update(value)
        name = case.get('name')
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name)
                and name not in names, 'unsafe or duplicate quality case name')
        require(type(case['steps']) is int and case['steps'] == 40, 'quality probe fixes steps at 40')
        for field, options in [('cfg_scale', (1, 3, 5)), ('lora_strength', (0, .5, 1)), ('action_scale', (0, .03))]:
            require(type(case[field]) in (int, float) and case[field] in options, f'invalid {field}')
        require(case['mode'] in ('segment6', 'full15'), 'quality mode must be segment6 or full15')
        require(case['solver'] in ('flow_euler', 'flow_unipc'), 'unsupported quality solver')
        names.add(name)
        result.append(case)
    require(request.get('fastpath', False) is False, 'text CFG must not reuse a fixed-text fastpath cache')
    return result


def load_negative(path, base_model):
    import torch
    path = Path(path)
    require(path.is_absolute() and path.suffix == '.pt', 'negative_cache must be an absolute .pt path')
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = metadata['source_model']
    text = metadata['prompts']['negative']
    require(payload['source_model'] == model and payload['prompts']['negative'] == text,
            'negative cache/receipt source or text differs')
    require(Path(model).name == Path(base_model).name, 'negative embedding uses a different model asset')
    rows = [row for row in metadata['rows'] if row.get('key') == 'negative']
    require(len(rows) == 1 and type(rows[0]['tokens']) is int and 1 <= rows[0]['tokens'] <= 512,
            'negative token count missing or ambiguous')
    embedding, tokens = payload['embeddings']['negative'], rows[0]['tokens']
    require(torch.is_tensor(embedding) and embedding.device.type == 'cpu' and embedding.dtype == torch.bfloat16
            and tuple(embedding.shape) == (512, 4096) and bool(torch.isfinite(embedding).all())
            and not bool(torch.count_nonzero(embedding[tokens:])), 'invalid negative embedding or padding')
    return embedding, dict(path=str(path), receipt=str(path.with_suffix('.json')), source_model=model,
                           text=text, tokens=tokens, cached=True, reencoded=False)


class TextCFGFlow:
    """Only text changes between branches; cfg=1 is the exact single-call path."""
    def __init__(self, model, scale, negative):
        self.model, self.scale, self.negative = model, scale, negative
        self.forward_calls = 0

    def __call__(self, *args, conditional_dict, **kwargs):
        self.forward_calls += 1
        positive = self.model(*args, conditional_dict=conditional_dict, **kwargs)
        if self.scale == 1:
            return positive
        require(self.negative is not None, 'CFG requires a verified negative embedding')
        negative_conditions = dict(conditional_dict, prompt_embeds=[self.negative])
        self.forward_calls += 1
        negative = self.model(*args, conditional_dict=negative_conditions, **kwargs)
        # Euler consumes velocity only; a conditional x0 would not be the guided x0.
        return negative[0] + self.scale * (positive[0] - negative[0]), None


def capture_lora_scales(model):
    from training.models.lora import LoRALinear
    modules = [(module, module.scaling) for module in model.modules() if isinstance(module, LoRALinear)]
    require(bool(modules), 'no LoRALinear modules found')
    return modules


def set_lora_strength(modules, strength):
    for module, original_scaling in modules:
        module.scaling = original_scaling * strength


class QualityRuntime(_WanRuntime):
    def __init__(self, state, config, residency, negative):
        super().__init__(state, config, residency)
        self.negative = negative
        try:
            self.lora_scales = capture_lora_scales(self.model)
        except BaseException:
            self.close()
            raise
        self.forward_calls = 0

    def configure(self, case):
        set_lora_strength(self.lora_scales, case['lora_strength'])
        self.config.action_scale = case['action_scale']
        self.cfg_scale, self.forward_calls = case['cfg_scale'], 0
        self.solver, self.solver_traces = case.get('solver', 'flow_euler'), []

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(fastpath is False, 'fixed-text fastpath is disabled for quality probes')
        negative = self.negative.to(device='cuda', dtype=self.torch.bfloat16) if self.cfg_scale != 1 else None
        original_model = self.model
        flow = TextCFGFlow(original_model, self.cfg_scale, negative)
        self.model = flow
        try:
            if self.solver == 'flow_unipc':
                from training.studio.unipc_probe import generate
                trace = {}
                self.solver_traces.append(trace)
                return generate(self, flow, first, prompt, keys, noise, steps, trace)
            # Same conditions, BF16 noise/sigmas, first-slot t=0 and Euler as before.
            return super().generate(first, prompt, keys, noise, steps, fastpath=False)
        finally:
            self.model = original_model
            self.forward_calls += flow.forward_calls

    def close(self):
        self.lora_scales, self.negative = [], None  # Do not keep GPU layers alive through the scaling registry.
        super().close()


def annotate_report(report, case, runtime, negative_source, load_seconds):
    report.update(action_scale=case['action_scale'], lora_strength=case['lora_strength'],
                  load_seconds=load_seconds, actual_model_forward_calls=runtime.forward_calls,
                  lora_linear_count=len(runtime.lora_scales), negative_prompt=negative_source if case['cfg_scale'] != 1 else None,
                  mathematical_changes=dict(text_cfg='v_negative + scale * (v_positive - v_negative)' if case['cfg_scale'] != 1 else 'single positive forward unchanged',
                                            lora='original_layer_scaling * lora_strength; action scale independent',
                                            same_action_and_image_condition_in_both_text_branches=True))
    report['sampler'] = dict(solver=case.get('solver', 'flow_euler'), steps=case['steps'], shift=5.0,
                             cfg='text_classifier_free' if case['cfg_scale'] != 1 else 'none',
                             cfg_scale=case['cfg_scale'], model_forwards_per_step=1 if case['cfg_scale'] == 1 else 2)
    if case.get('solver', 'flow_euler') == 'flow_unipc':
        report['sampler']['window_schedules'] = copy.deepcopy(getattr(runtime, 'solver_traces', []))
        report['mathematical_changes']['solver'] = (
            'UniPC order2 bh2, diffusers flow schedule and FP32 future-state integration; '
            'NOT identical Euler sigma nodes or precision; model input remains BF16')
    return report


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    began = time.monotonic()
    cases = validate_cases(request)
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= 3600, 'finite max_seconds required')
    directory = Path(directory).resolve()
    require(directory.is_dir() and not (directory / 'inputs.json').exists(), 'new quality job directory required')
    validate_active_authority(lease)
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    composition = {}
    if 'action_adapter_donor_source' in request:
        from training.studio.adapter_hybrid import apply_action_adapter_donor, composition_fields
        state, metadata = apply_action_adapter_donor(request, state, config, lineage)
        composition = composition_fields(metadata)
    require(scene['seed'] == 42, 'quality source must use the fixed seed 42')
    negative, negative_source = load_negative(request.get('negative_cache', DEFAULT_NEGATIVE_CACHE), config.base_model_path)
    write_json(directory / 'inputs.json', dict(scene_id=scene['scene_id'], prompt=scene['prompt'], seed=42,
        source_job=request['source_job'], source_actions=scene['action_segments'], negative_prompt=negative_source,
        checkpoint=lineage['path'], checkpoint_sha256=lineage['sha256'], cases=cases,
        source_revision=request.get('source_revision'), experiment_revision=request.get('experiment_revision'),
        input_loading_seconds=time.monotonic() - began, fastpath=False, width=832, height=480, **composition))
    Image.fromarray(initial).save(directory / 'initial.png')
    residency = request.get('residency', 'co_resident')
    require(residency in ('phase', 'co_resident'), 'invalid quality residency')
    loaded = time.monotonic()
    runtime = QualityRuntime(state, config, residency, negative)
    engine, results = None, []
    try:
        runtime.synchronize()
        cold_seconds = time.monotonic() - loaded
        engine = ReusableActionEngine(state, config, residency=residency, fastpath=False, _runtime=runtime)
        engine.load_seconds = cold_seconds  # Injecting a runtime otherwise reports near-zero construction.
        for case in cases:
            require(time.monotonic() - began < maximum - 30, 'quality job deadline reached; no next case started')
            validate_active_authority(lease)
            target = directory / case['name']
            require(target.resolve().parent == directory, 'quality output escaped job directory')
            target.mkdir(exist_ok=False)
            runtime.configure(case)
            engine.action_scale = case['action_scale']
            started = time.monotonic()
            try:
                options = dict(initial=initial, prompt=prompt, seed=42, output_path=target / 'raw.mp4', steps=case['steps'])
                if case['mode'] == 'segment6':
                    report, endpoint = engine.run_segment(actions=actions[:96].copy(), **options)
                    np.save(target / 'endpoint.npy', endpoint.numpy(), allow_pickle=False)
                else:
                    report = engine.run(actions=actions.copy(), window6=True, **options)
                annotate_report(report, case, runtime, negative_source, cold_seconds)
                report.update(composition)
                expected = case['steps'] * (1 if case['cfg_scale'] == 1 else 2) * (1 if case['mode'] == 'segment6' else 3)
                require(runtime.forward_calls == expected, 'actual forward count differs from configured CFG')
                entry = dict(**case, status='completed', seed=42, generation=report,
                             raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
                             elapsed_seconds_scope='case_run_excludes_shared_cold_load', quality_evaluation='pending_visual_comparison', realtime=False,
                             **composition)
                results.append(entry)
                write_json(target / 'result.json', entry)
                write_json(directory / 'progress.json', dict(status='running', cases=results))
            except Exception as error:
                report = copy.deepcopy(engine.last_run_report or {})
                annotate_report(report, case, runtime, negative_source, cold_seconds)
                report.update(composition)
                write_json(target / 'result.json', dict(**case, status='failed', generation=report,
                    error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started, **composition))
                raise  # No retries or silent fall-through to other cases.
    finally:
        engine.close() if engine is not None else runtime.close()
    return dict(status='completed', mode='quality_probe', cases=results, load_seconds=cold_seconds,
                load_seconds_scope='one_time_real_runtime_construction_and_synchronization',
                elapsed_seconds=time.monotonic() - began, elapsed_seconds_scope='source_loading_through_engine_close',
                realtime=False, quality_evaluation='pending_visual_comparison', **composition)
