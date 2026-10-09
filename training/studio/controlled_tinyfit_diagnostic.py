"""One fixed before/after, 32-forward signal-localization probe; never rollout.

Only the existing worker may execute this module. Preparation reads small JSON,
config and mmap actions, not weights/latent values. Execution uses real cached
future latents only as noised training-flow targets, never as a video rollout.
No backward, optimizer, VAE, saved features, parameter sweeps or automatic retry.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import controlled_action_diagnostic as previous
from training.studio import controlled_lora_pair_eval as paired
from training.studio import controlled_matched_window_eval as matched
from training.studio import controlled_tinyfit_eval as tiny

MODE = 'controlled_tinyfit_diagnostic'
REVISION = 'controlled-tinyfit-before-after-fixed2-sigma2-action4-signal-v1'
ROOT = matched.BASE / 'teacher-tinyfit-diagnostic-v1'
JOB_ID = 'ctfd-paired32-v1'
RESCHEDULE_JOB_ID = 'ctfd-paired32-v2-cpu-preflight'
ENDPOINTS = ('parent', 'tinyfit20')
TARGETS = ('WJ', 'WL')
SIGMAS = (.25, .75)
CONDITIONS = ('correct', 'jl_swap', 'zero', 'gate_off')
SIGNALS = ('raw_action_feature', 'normalized_action_feature',
           'fp32_controlled_delta', 'actual_bf16_delta')
OUTPUT_BYTES_LIMIT = 10_000_000
action_condition = previous.action_condition


def protocol():
    return dict(endpoints=list(ENDPOINTS), targets=list(TARGETS), sigmas=list(SIGMAS),
        conditions=list(CONDITIONS), model_forward_calls=32, pairs_per_endpoint=4,
        seed=42, noise_namespace=REVISION, shared_noise_across_endpoints_sigmas_conditions=True,
        selection='existing_top1_WJ_and_WL_matched_training_windows;no_reselection',
        cfg=False, backward=False, decoding=False, first_latent_loss=False,
        target='noise-clean;BF16 target and BF16 noisy input as existing teacher loss',
        ratio_cap=.1, zero_actions_are_not_gate_off=True,
        feature_statistics='all_elements;streaming_one_temporal_slot;no_saved_feature_tensors',
        actual_bf16_delta='BF16(future+BF16(fp32_delta))-future;exact_helper_formula',
        model_loading='sequential_one_full_model;release_before_next_endpoint',
        visual_quality_verified=False, action_correctness_verified=False,
        interpretation='local_condition_preference;no_counterfactual_swap_ground_truth;not_generalization')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def _tiny_job(target):
    return tiny.ROOT / tiny.CASES[(target, 'original')]


def _reschedule_binding(binding):
    """One explicit new identity after the settled timeout, not a retry loop."""
    preflight_path = ROOT/'cpu-preflight-v2.json'
    old_directory = ROOT/JOB_ID
    old_request_path = old_directory/'request.json'
    settlement_path = old_directory/'settlement.json'
    settlement_request_path = old_directory/'settlement-request.json'
    preflight, old, settlement, context = map(_read, (
        preflight_path, old_request_path, settlement_path, settlement_request_path))
    validate_request(old, old_directory)
    require(old['binding'] == binding, 'rescheduling must preserve the original diagnostic inputs')
    require(preflight.get('status') == 'completed' and preflight.get('returncode') == 0
            and preflight.get('cuda_visible_devices') == ''
            and preflight.get('gpu_reservation_attempted') is False
            and preflight.get('model_construction_requested') is False
            and preflight.get('weights_read_requested') is False,
            'successful CPU-only v2 import preflight required')
    child = preflight.get('child_result', {})
    require(child.get('status') == 'import_completed'
            and child.get('cuda_hardware_queries') is False
            and child.get('models_constructed') == 0
            and child.get('model_weights_loaded') is False,
            'CPU preflight must not load models/weights or query CUDA hardware')
    require(settlement.get('status') == 'settled' and settlement.get('outcome') == 'failed'
            and context.get('outcome') == 'failed' and context.get('worker_returncode') == 124,
            'original v1 must be a settled worker timeout, not an active or successful job')
    require(context.get('job_id') == JOB_ID and context.get('job_directory') == str(old_directory)
            and context.get('request_sha256') == sha256(old_request_path)
            and context.get('lease', {}).get('job_id') == JOB_ID
            and settlement.get('reservation_id') == context.get('lease', {}).get('reservation_id')
            and bool(settlement.get('reservation_id')),
            'v1 timeout settlement must bind the original request and reservation')
    paths = (preflight_path, old_request_path, settlement_path, settlement_request_path)
    return dict(previous_job_id=JOB_ID, new_job_id=RESCHEDULE_JOB_ID,
        reason='explicit_reschedule_after_settled_timeout_and_successful_cpu_import_preflight',
        previous_outcome='failed', previous_worker_returncode=124, automatic_retry=False,
        mathematics_unchanged=True, noise_namespace=REVISION,
        files={str(path):sha256(path) for path in paths})


def _metadata_binding():
    """Read existing small records and stat weights, never read checkpoint bytes."""
    parent = matched._source_evaluation()
    files = [matched.SELECTION, matched.SOURCE_EVAL, Path(parent['training_config'])]
    require(sha256(parent['training_config']) == parent['training_config_sha256'],
            'parent config changed')
    windows, tiny_sources = {}, []
    for target in TARGETS:
        baseline_dir = tiny.baseline_job(target, 'original')
        after_dir = _tiny_job(target)
        paths = [baseline_dir/'request.json', baseline_dir/'result.json',
                 after_dir/'request.json', after_dir/'result.json']
        before_request, before, after_request, after = [_read(p) for p in paths]
        matched.validate_request(before_request)
        tiny.validate_request(after_request)
        require(before_request['target'] == after_request['target'] == target
                and before_request['condition'] == after_request['condition'] == 'original',
                'only completed original-action matched sources accepted')
        require(before_request['source_evaluation_sha256'] == sha256(matched.SOURCE_EVAL)
                and before_request['selection_sha256'] == sha256(matched.SELECTION),
                'original matched selection/source changed')
        require(after_request['baseline_request_sha256'] == sha256(paths[0])
                and after_request['baseline_result_sha256'] == sha256(paths[1]),
                'tinyfit baseline records changed')
        for request, result in ((before_request, before), (after_request, after)):
            require(result['status'] == 'completed' and result['request'] == request,
                    'completed source with exact saved request required')
            require(result['frames_written'] == 97, 'source evaluation must have97 frames')
        row = matched._selected(target)
        shared = ('selection', 'first_latent_sha256_float32', 'prompt_sha256_float32',
                  'original_action_sha256_float32', 'actual_action_sha256_float32', 'actual_actions')
        require(all(before['lineage'][key] == after['lineage'][key] for key in shared),
                'before/after static input lineage differs')
        require(before['lineage']['selection'] == row
                and before['lineage']['checkpoint'] == parent['checkpoint']
                and before['lineage']['checkpoint_sha256'] == parent['checkpoint_sha256']
                and after['lineage']['checkpoint'] == after_request['checkpoint'],
                'saved source selection/checkpoint changed')
        parent_lineage = after['lineage']['checkpoint_lineage']['parent_lineage']
        require(parent_lineage['parent_checkpoint'] == parent['checkpoint']
                and parent_lineage['parent_arm'] == 'frozen' and parent_lineage['parent_step'] == 20,
                'tinyfit is not derived from the matched parent endpoint')
        require(_stat(row['shard_path']) == row['shard_stat'], 'selected cache stat changed')
        windows[target] = dict(selection=row, **{key:before['lineage'][key] for key in shared if key != 'selection'})
        files.extend(paths)
        tiny_sources.append(after_request)
    for key in ('checkpoint', 'checkpoint_sha256', 'training_config', 'training_config_sha256'):
        require(tiny_sources[0][key] == tiny_sources[1][key], 'tinyfit endpoints differ across windows')
    after_source = tiny_sources[0]
    require(sha256(after_source['training_config']) == after_source['training_config_sha256'],
            'tinyfit config changed')
    files.append(Path(after_source['training_config']))
    endpoints = {}
    for name, source, request_path in (
            ('parent', parent, matched.SOURCE_EVAL),
            ('tinyfit20', after_source, _tiny_job('WJ')/'request.json')):
        contained(source['checkpoint'], matched.DATA_ROOT)
        endpoints[name] = dict(evaluation_request=str(request_path), checkpoint=source['checkpoint'],
            checkpoint_sha256=source['checkpoint_sha256'], checkpoint_stat=_stat(source['checkpoint']),
            training_config=source['training_config'], training_config_sha256=source['training_config_sha256'])
    return dict(files={str(path):sha256(path) for path in files}, windows=windows, endpoints=endpoints)


def validate_request(request, directory):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('protocol') == protocol(), 'explicit fixed tinyfit diagnostic required')
    job_id = request.get('job_id')
    require(job_id in (JOB_ID, RESCHEDULE_JOB_ID) and job_id == Path(directory).name,
            'unique fixed diagnostic job required')
    if job_id == JOB_ID:
        require('reschedule' not in request, 'original v1 request must remain unchanged')
    else:
        reschedule = request.get('reschedule', {})
        expected = dict(previous_job_id=JOB_ID, new_job_id=RESCHEDULE_JOB_ID,
            reason='explicit_reschedule_after_settled_timeout_and_successful_cpu_import_preflight',
            previous_outcome='failed', previous_worker_returncode=124, automatic_retry=False,
            mathematics_unchanged=True, noise_namespace=REVISION)
        require(set(reschedule) == set(expected)|{'files'}
                and all(reschedule.get(key) == value for key,value in expected.items()),
                'v2 requires explicit unchanged-protocol rescheduling evidence')
        paths = (ROOT/'cpu-preflight-v2.json', ROOT/JOB_ID/'request.json',
                 ROOT/JOB_ID/'settlement.json', ROOT/JOB_ID/'settlement-request.json')
        require(set(reschedule['files']) == {str(path) for path in paths},
                'v2 requires the exact CPU preflight and original timeout records')
        for digest in reschedule['files'].values():
            require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),
                    'v2 small-record identity required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 900,
            'finite diagnostic watchdog must be <=900 seconds')
    require(isinstance(request.get('source_revision'), str) and bool(request['source_revision']),
            'source revision required')
    binding = request.get('binding', {})
    require(set(binding) == {'files', 'windows', 'endpoints'}
            and set(binding['windows']) == set(TARGETS)
            and set(binding['endpoints']) == set(ENDPOINTS), 'both fixed endpoints/windows required')
    for target, value in binding['windows'].items():
        row = value['selection']
        require((row['episode_id'], row['window_index'], row['source_start']) == matched.FIXED[target],
                'fixed matched training window changed')
    require(binding['endpoints']['parent']['evaluation_request'] == str(matched.SOURCE_EVAL)
            and binding['endpoints']['tinyfit20']['evaluation_request'] == str(_tiny_job('WJ')/'request.json'),
            'fixed endpoint source requests required')
    for path, digest in binding['files'].items():
        contained(path, matched.DATA_ROOT)
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'small record identity required')
    require(not any(key in request for key in ('checkpoint', 'actions', 'action_scale', 'gate_scale',
        'ratio_cap', 'sigma', 'seed', 'training_config', 'lora_strength')), 'no diagnostic overrides allowed')


def _verify_binding(request):
    require(_metadata_binding() == request['binding'], 'frozen diagnostic source binding changed')
    if request['job_id'] == RESCHEDULE_JOB_ID:
        require(_reschedule_binding(request['binding']) == request['reschedule'],
                'v2 CPU preflight or original timeout evidence changed')


def _read_actions(window):
    import torch
    row = window['selection']
    require(_stat(row['shard_path']) == row['shard_stat'], 'selected shard stat changed')
    payload = torch.load(row['shard_path'], map_location='cpu', weights_only=True, mmap=True)
    require(payload['episode_id'] == row['episode_id'] and payload['kind'] == 'action_teacher_window97_features_v1'
            and payload['prompt_policy'] == 'scene_static_only_v1'
            and int(payload['source_starts'][row['window_index']]) == row['source_start'], 'cache identity differs')
    actions = payload['actions'][row['window_index']].detach().float().clone()
    require(torch.equal(actions, matched.actions_from_runs(row))
            and previous._tensor_digest(actions) == window['original_action_sha256_float32']
            and actions.tolist() == window['actual_actions'], 'fixed original actions changed')
    return actions


def prepare(*, output, source_revision, max_seconds=900):
    output = contained(output, ROOT)
    require(not output.exists(), 'new diagnostic directory required; no overwrite/retry')
    binding = _metadata_binding()
    for target in TARGETS:
        _read_actions(binding['windows'][target])
    request = dict(mode=MODE, experiment_revision=REVISION, job_id=output.name,
        source_revision=source_revision, max_seconds=max_seconds, protocol=protocol(), binding=binding,
        preparation=dict(weights_loaded=False, latent_values_read=False, text_values_read=False,
            action_slices_read=2, new_window_selection=False, full_shard_hashes_recomputed=False,
            full_checkpoint_hashes_recomputed=False, gpu_launched=False))
    if output.name == RESCHEDULE_JOB_ID:
        request['reschedule'] = _reschedule_binding(binding)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    return request


def _window(window):
    import torch
    row = window['selection']
    actions = _read_actions(window)
    payload = torch.load(row['shard_path'], map_location='cpu', weights_only=True, mmap=True)
    first, prompt, same_actions = matched._extract_condition(payload, row)
    require(previous._tensor_digest(first) == window['first_latent_sha256_float32']
            and previous._tensor_digest(prompt) == window['prompt_sha256_float32']
            and torch.equal(actions, same_actions), 'diagnostic static input differs from real evaluation')
    clean = payload['clean_latents'][row['window_index']].detach().float().clone()
    require(tuple(clean.shape) == (25,48,30,52) and bool(torch.isfinite(clean).all()),
            'expected finite native97 clean flow target')
    require(not any(key in payload for key in ('clip_fea','y')), 'unexpected extra cache conditioning')
    return clean, prompt.float(), actions


def make_batch(clean, prompt, actions, *, target, sigma):
    import torch
    from training.data.action_dataset import _seed64
    require(target in TARGETS and sigma in SIGMAS, 'fixed target and sigma required')
    seed = _seed64(42, REVISION, target, *matched.FIXED[target])
    noise = torch.randn(clean.shape, generator=torch.Generator(device='cpu').manual_seed(seed))
    noisy, flow = (1-sigma)*clean+sigma*noise, noise-clean
    noisy[0], flow[0] = clean[0], 0
    labels = torch.full((1,clean.shape[0]), sigma*1000., dtype=torch.float32)
    labels[:,0] = 0
    return dict(noisy_latents=noisy[None], target_flow=flow[None], timesteps=labels,
        prompt_embeds=prompt[None], actions=actions[None]), seed


def _components(future, raw, gate, *, cap, eps):
    """One-slot exact production formula, including real BF16 cast and addition."""
    import torch
    with torch.autocast(device_type=future.device.type, enabled=False):
        vf = raw.float()
        denominator = vf.square().mean(dim=1, keepdim=True).clamp_min(eps*eps).sqrt()
        normalized = vf/denominator
        reference = future.detach().float().square().mean(dim=1, keepdim=True).sqrt()
        delta = cap*reference*gate.tanh().view(1,-1,1,1,1)*normalized
        actual = (future+delta.to(future.dtype)).float()-future.float()
    return (vf, normalized, delta, actual)


def signal_statistics(future, raw, gate, *, correct_future, correct_raw, correct_gate, cap=.1, eps=1e-6):
    """All-element reductions with bounded temporary memory, not sampled proxies.

    The correct BF16 feature snapshots may be on CPU. Only one temporal slot is
    copied to the active device at a time; no full FP32 feature copies are kept.
    """
    import torch
    require(future.shape == raw.shape == correct_future.shape == correct_raw.shape
            and raw.ndim == 5 and raw.shape[2] > 0, 'signal geometry differs')
    require(future.dtype == raw.dtype == correct_future.dtype == correct_raw.dtype == torch.bfloat16,
            'measure the actual BF16 inference feature path')
    require(cap == .1 and eps == 1e-6, 'fixed controlled formula required')
    sums = torch.zeros((len(SIGNALS),4), dtype=torch.float64, device=raw.device)
    video_sum = torch.zeros((), dtype=torch.float64, device=raw.device)
    same_video, actual_zero_count = True, 0
    count = 0
    baseline_gate = correct_gate.to(device=raw.device, dtype=torch.float32)
    for slot in range(raw.shape[2]):
        u, v = future[:,:,slot:slot+1], raw[:,:,slot:slot+1]
        cu = correct_future[:,:,slot:slot+1].to(device=raw.device)
        cv = correct_raw[:,:,slot:slot+1].to(device=raw.device)
        same_video &= torch.equal(u, cu)
        require(same_video, 'same noisy input unexpectedly produced different patch features')
        now = _components(u,v,gate,cap=cap,eps=eps)
        before = _components(cu,cv,baseline_gate,cap=cap,eps=eps)
        for index,(value,reference) in enumerate(zip(now,before)):
            require(bool(torch.isfinite(value).all()) and bool(torch.isfinite(reference).all()),
                    'nonfinite diagnostic signal')
            sums[index,0] += value.square().sum(dtype=torch.float64)
            sums[index,1] += reference.square().sum(dtype=torch.float64)
            sums[index,2] += (value-reference).square().sum(dtype=torch.float64)
            sums[index,3] += (value*reference).sum(dtype=torch.float64)
        video_sum += u.float().square().sum(dtype=torch.float64)
        actual_zero_count += int(torch.count_nonzero(now[3] == 0))
        count += v.numel()
    require(bool(torch.isfinite(sums).all()) and bool(torch.isfinite(video_sum)),
            'nonfinite reduced signal statistics')
    video_rms = math.sqrt(float(video_sum)/count)
    result = dict(video_patch_rms=video_rms, video_patch_identical_to_correct=same_video,
        elements_per_signal=count, actual_bf16_delta_zero_fraction=actual_zero_count/count)
    for name,(ss,rr,dd,dot) in zip(SIGNALS,sums.cpu().tolist()):
        rms, reference_rms, difference = math.sqrt(ss/count), math.sqrt(rr/count), math.sqrt(dd/count)
        result[name] = dict(rms=rms, correct_rms=reference_rms, delta_vs_correct_rms=difference,
            delta_vs_correct_ratio=difference/reference_rms if reference_rms else None,
            cosine_vs_correct=dot/math.sqrt(ss*rr) if ss and rr else None)
    for name in ('fp32_controlled_delta','actual_bf16_delta'):
        result[name]['global_rms_ratio_to_video'] = result[name]['rms']/video_rms if video_rms else None
    return result


class SignalTrace:
    """Read-only hooks on the original forward, never replacing module output."""
    def __init__(self, wrapper):
        self.model = wrapper.model
        self.forward_calls = 0
        self.handles = [wrapper.register_forward_pre_hook(self._enter),
                        self.model.patch_embedding.register_forward_hook(self._patch),
                        self.model.act_control_adapter.register_forward_hook(self._action),
                        wrapper.register_forward_hook(self._velocity)]
        self.begin_group()

    def _enter(self, module, inputs):
        self.forward_calls += 1

    def begin_group(self):
        self.correct_patch = self.correct_raw = self.correct_gate = self.correct_velocity = None
        self.begin_condition('correct')

    def begin_condition(self, condition):
        require(condition in CONDITIONS, 'unknown trace condition')
        self.condition, self.patch, self.velocity = condition, None, None
        self.metrics = {}

    def _patch(self, module, inputs, output):
        self.patch = output.detach()

    def _action(self, module, inputs, output):
        require(self.patch is not None and self.patch.shape[2] > output.shape[2],
                'action must follow patch embedding with clean prefix')
        future = self.patch[:,:,self.patch.shape[2]-output.shape[2]:]
        raw, gate = output.detach(), self.model.controlled_action_gate.detach()
        if self.condition == 'correct':
            self.correct_patch = future.cpu().clone()
            self.correct_raw = raw.cpu().clone()
            self.correct_gate = gate.cpu().clone()
        require(self.correct_raw is not None, 'correct condition must precede contrasts')
        self.metrics = signal_statistics(future,raw,gate,correct_future=self.correct_patch,
            correct_raw=self.correct_raw,correct_gate=self.correct_gate,
            cap=self.model.controlled_action_ratio_cap,eps=self.model.controlled_action_eps)
        self.metrics['reference_feature_bytes_cpu'] = sum(
            x.numel()*x.element_size() for x in (self.correct_patch,self.correct_raw,self.correct_gate))
        self.patch = None

    def _velocity(self, module, inputs, output):
        value = output[0] if isinstance(output,tuple) else output
        self.velocity = value[:,1:].detach().float().cpu()
        if self.condition == 'correct':
            self.correct_velocity = self.velocity.clone()
        require(self.correct_velocity is not None, 'missing correct velocity reference')
        rms, delta = previous._rms(self.velocity), previous._rms(self.velocity-self.correct_velocity)
        correct_rms = previous._rms(self.correct_velocity)
        self.metrics['velocity'] = dict(rms=rms,correct_rms=correct_rms,delta_vs_correct_rms=delta,
            delta_vs_correct_ratio=delta/correct_rms if correct_rms else None)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.begin_group()


@contextmanager
def condition_gate(model, condition):
    """Only gate-off is an intervention; even failed forwards restore exactly."""
    import torch
    require(condition in CONDITIONS, 'unknown gate condition')
    gate = model.model.controlled_action_gate
    before = gate.detach().clone()
    with torch.no_grad():
        try:
            if condition == 'gate_off':
                gate.zero_()
            yield
        finally:
            if condition == 'gate_off':
                gate.copy_(before)
            require(torch.equal(gate,before), 'diagnostic changed the learned gate')


def comparisons(records):
    result = {}
    for endpoint in ENDPOINTS:
        local = [row for row in records if row['endpoint'] == endpoint]
        require(len(local) == 16 and len({(r['target'],r['sigma'],r['condition']) for r in local}) == 16,
                'each endpoint requires2 windows x2 sigma x4 conditions')
        result[endpoint] = {}
        for condition in CONDITIONS:
            rows = [row for row in local if row['condition'] == condition]
            result[endpoint][condition] = dict(pairs=len(rows),
                mean_flow_loss=sum(r['flow_loss'] for r in rows)/4,
                mean_loss_delta_vs_correct=sum(r['loss_minus_correct'] for r in rows)/4,
                mean_relative_loss_delta_vs_correct=sum(r['relative_loss_minus_correct'] for r in rows)/4,
                pairs_correct_loss_lower=sum(r['loss_minus_correct'] > 0 for r in rows),
                mean_velocity_delta_vs_correct_rms=sum(r['signals']['velocity']['delta_vs_correct_rms'] for r in rows)/4)
    return result


def _load_endpoint(name, request):
    endpoint = request['binding']['endpoints'][name]
    source = _read(endpoint['evaluation_request'])
    if name == 'parent':
        return paired._load_weights(source)
    return tiny._load_checkpoint(source, request['binding']['endpoints']['parent']['checkpoint'])


class PhaseProgress:
    """Sparse durable phase markers survive a worker timeout without a summary."""
    def __init__(self, directory, job_id, began):
        self.path = Path(directory)/'diagnostic-progress.jsonl'
        self.job_id, self.began, self.last_phase = job_id, began, None
        # Even a failed entry is never silently resumed under the same identity.
        with self.path.open('x', encoding='utf-8'):
            pass

    def emit(self, phase, *, completed_metric_rows, actual_model_forward_calls, **details):
        from training.runtime import append_jsonl
        row = dict(job_id=self.job_id, phase=phase, timestamp_unix=time.time(),
            elapsed_seconds=time.monotonic()-self.began,
            completed_metric_rows=completed_metric_rows,
            actual_model_forward_calls=actual_model_forward_calls, **details)
        append_jsonl(self.path,row)
        self.last_phase = phase
        print('DIAGNOSTIC_PHASE '+json.dumps(row),flush=True)
        return row


def run(request, directory, lease):
    import torch
    import train_action_teacher as teacher
    from training.demo.backend import validate_active_authority
    from training.runtime import append_jsonl
    directory = contained(directory,ROOT)
    validate_request(request,directory)
    request_sha = sha256(directory/'request.json')
    require(_read(directory/'request.json') == request, 'leased request differs')
    require(not (directory/'diagnostic-metrics.jsonl').exists()
            and not (directory/'diagnostic-summary.json').exists()
            and not (directory/'diagnostic-progress.jsonl').exists(), 'no overwrite or automatic retry')
    validate_active_authority(lease)
    teacher._shared_training_gpu(lease,max_seconds=request['max_seconds'],request_sha256=request_sha)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
    began, device, records, model, trace = time.monotonic(), torch.device('cuda',0), [], None, None
    report = dict(status='running',mode=MODE,experiment_revision=REVISION,protocol=protocol(),
        request=request,model_weights_updated=False,gate_restored=None,visual_quality_verified=False,
        action_correctness_verified=False,independent_dev_evaluation=False,endpoint_lineage={},
        actual_model_forward_calls=0)
    progress = PhaseProgress(directory,request['job_id'],began)

    def phase(name, **details):
        return progress.emit(name,completed_metric_rows=len(records),
            actual_model_forward_calls=report['actual_model_forward_calls']+(trace.forward_calls if trace else 0),
            **details)

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    def check():
        require(time.monotonic()-began < request['max_seconds']-20, 'finite diagnostic deadline reached')
        require(matched.directory_bytes(directory) < OUTPUT_BYTES_LIMIT-100_000, 'diagnostic small-output cap reached')
        validate_active_authority(lease)

    try:
        check()
        phase('binding_started')
        _verify_binding(request)
        phase('binding_completed')
        for endpoint in ENDPOINTS:
            check()
            load_started = time.monotonic()
            phase('checkpoint_load_started',endpoint=endpoint)
            state, config, lineage = _load_endpoint(endpoint,request)
            phase('checkpoint_load_completed',endpoint=endpoint)
            phase('model_build_started',endpoint=endpoint)
            model,_ = paired.build_inference_model(config,state,arm='frozen')
            phase('model_build_completed',endpoint=endpoint)
            del state
            gc.collect()
            phase('to_gpu_started',endpoint=endpoint)
            model.to(device)
            phase('to_gpu_completed',endpoint=endpoint)
            require(not any(p.requires_grad or p.grad is not None for p in model.parameters()),
                    'diagnostic must have no trainable parameters/gradients')
            versions = {name:p._version for name,p in model.named_parameters() if name != 'model.controlled_action_gate'}
            learned_gate = model.model.controlled_action_gate.detach().clone()
            report['endpoint_lineage'][endpoint] = dict(lineage=lineage,load_seconds=time.monotonic()-load_started)
            check()
            trace = SignalTrace(model)
            try:
                with torch.inference_mode():
                    for target in TARGETS:
                        clean,prompt,actions = _window(request['binding']['windows'][target])
                        for sigma in SIGMAS:
                            trace.begin_group()
                            batch,seed = make_batch(clean,prompt,actions,target=target,sigma=sigma)
                            correct_loss = None
                            for condition in CONDITIONS:
                                check()
                                trace.begin_condition(condition)
                                local = dict(batch,actions=action_condition(batch['actions'],condition))
                                phase('forward_started',endpoint=endpoint,target=target,sigma=sigma,
                                      condition=condition,noise_seed=seed)
                                torch.cuda.synchronize(device)
                                tick = time.monotonic()
                                with condition_gate(model,condition), torch.autocast('cuda',dtype=torch.bfloat16):
                                    loss = teacher._forward_loss(model,local,config,device)
                                torch.cuda.synchronize(device)
                                elapsed = time.monotonic()-tick
                                require(trace.velocity is not None and bool(torch.isfinite(trace.velocity).all())
                                        and bool(torch.isfinite(loss)), 'nonfinite or missing forward output')
                                if condition == 'gate_off':
                                    require(trace.metrics['fp32_controlled_delta']['rms'] == 0
                                            and trace.metrics['actual_bf16_delta']['rms'] == 0, 'gate-off residual must be exactlyzero')
                                loss_value = float(loss)
                                if condition == 'correct': correct_loss = loss_value
                                row = dict(endpoint=endpoint,target=target,sigma=sigma,condition=condition,
                                    noise_seed=seed,flow_loss=loss_value,loss_minus_correct=loss_value-correct_loss,
                                    relative_loss_minus_correct=(loss_value-correct_loss)/max(correct_loss,1e-30),
                                    condition_action_sha256=previous._tensor_digest(local['actions']),
                                    canonical_action_rms=previous._rms(local['actions']),
                                    target_rms=previous._rms(batch['target_flow'][:,1:].bfloat16()),
                                    forward_seconds_including_hooks=elapsed,signals=trace.metrics)
                                records.append(row)
                                append_jsonl(directory/'diagnostic-metrics.jsonl',row)
                                print(json.dumps(row),flush=True)
                                phase('forward_completed',endpoint=endpoint,target=target,sigma=sigma,
                                      condition=condition,noise_seed=seed)
                            del batch,local
                        del clean,prompt,actions
                require(versions == {name:p._version for name,p in model.named_parameters()
                        if name != 'model.controlled_action_gate'}, 'diagnostic changed non-gate parameter versions')
                require(not any(p.grad is not None for p in model.parameters()), 'diagnostic created parameter gradients')
            finally:
                restored = torch.equal(model.model.controlled_action_gate,learned_gate)
                versions_match = versions == {name:p._version for name,p in model.named_parameters()
                    if name != 'model.controlled_action_gate'}
                report['model_weights_updated'] |= not (restored and versions_match)
                report['endpoint_lineage'][endpoint]['gate_restored'] = restored
                report['endpoint_lineage'][endpoint]['non_gate_parameter_versions_unchanged'] = versions_match
                report['gate_restored'] = all(value.get('gate_restored',False)
                    for value in report['endpoint_lineage'].values())
                report['actual_model_forward_calls'] += trace.forward_calls
                trace.close()
                trace = None
                del learned_gate
                del model
                model = None
                gc.collect()
                torch.cuda.empty_cache()
            check()
        require(len(records) == report['actual_model_forward_calls'] == 32, 'exactly32 fixed forwards required')
        require(report['gate_restored'] is True and report['model_weights_updated'] is False,
                'diagnostic must leave all loaded model weights unchanged')
        for target in TARGETS:
            require(len({r['noise_seed'] for r in records if r['target'] == target}) == 1,
                    'before/after/sigma/condition noise identity differs')
        report.update(status='completed',comparisons=comparisons(records))
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,error=str(error),
                      failure_phase=progress.last_phase)
        raise
    finally:
        # Write before cleanup, which can itself block under shared storage/CUDA pressure.
        last_work_phase = progress.last_phase
        phase('diagnostic_end',status=report['status'],last_work_phase=last_work_phase,
              failure_type=report.get('failure_type'))
        if trace is not None: trace.close()
        if model is not None: del model
        gc.collect()
        torch.cuda.empty_cache()
        report.update(completed_metric_rows=len(records),elapsed_seconds=time.monotonic()-began,
            peak_vram_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            metrics_path=str(directory/'diagnostic-metrics.jsonl'),features_saved=False,
            progress_path=str(progress.path),last_work_phase=last_work_phase)
        write_json(directory/'diagnostic-summary.json',report)
    return report
