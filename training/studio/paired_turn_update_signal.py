"""One B20 real-batch gradient and two isolated first-conv shadow updates; no training continuation."""
from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import math
import os
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import paired_turn_b_eval as b
from training.studio import paired_turn_cfg_signal as cfg

MODE = 'paired_turn_update_signal'
REVISION = 'b20-real-batch152-159-single-shadow-update-v1'
SOURCE_REVISION = 'creator-adapter-realgrad1-v1'
ROOT = b.a.matched.BASE / 'teacher-adapter-update-v1'
JOB_ID = 'cau-realgrad1-v1'
MAX_SECONDS, CLEANUP_SECONDS, STAGE_BYTES = 1800, 60, 100_000_000
PARAMETER = 'model.act_control_adapter.conv.weight'
SHAPE = (3072, 8192, 2, 2)
CHUNK = 262144
METRICS = b.CHECKPOINT.parents[1] / 'metrics.jsonl'
CENSUS = cfg.ROOT / 'adapter-weights-comparison.json'
VENDOR = Path('/DATA_DISK/InteractWorld/venv/lib/python3.10/site-packages/bitsandbytes/optim/optimizer.py')
SEEDS = (10655024954624632928, 8799928946889093185, 13738743180787645673,
         5131272213376099155, 876084163090624031, 209494445856372140,
         7400434388352140693, 4546618386547450976)
TIMESTEPS = (963.0394897460938, 926.8355712890625, 682.849365234375,
             914.3394165039062, 932.0956420898438, 757.8740234375,
             860.34912109375, 954.9102783203125)
read = b.read


def protocol():
    return dict(checkpoint=str(b.CHECKPOINT), checkpoint_stage=b.STAGE, checkpoint_step=20,
        parameter=PARAMETER, shape=list(SHAPE), batch_indices=list(range(152, 160)),
        model_loads=1, loss_forward_calls=8, backward_calls=8, accumulation_divisor=8,
        global_clip_norm=1., original_model_optimizer_steps=0, shadow_optimizer_steps=2,
        shadow_dtypes=['torch.bfloat16', 'torch.float32'], shadow_state_initial_step=20,
        shadow_state_final_step=21, gradient='one_actual_accumulated_clipped_BF16_gradient_exactly_widened',
        optimizer='original_native_AdamW8bit_loader_full_value_dtype_roundtrip_before_model_and_before_step',
        before_weights='same_B20_BF16_forward_representable_values',
        scope='B20_repeat_fixed_last_batch_not_historical_B19_step20_gradient_replay',
        future_clean_latents='supervised_training_target_only_no_generation',
        checkpoint_saved=False, video_generated=False, training_continued=False, automatic_next_stage=False)


def fixed_samples():
    return [dict(absolute_sample_index=i, new_stage_sample_index=i,
        index_namespace='controlled_tinyfit_two_windows_seed42_v1', episode_slot=0,
        episode_id='0328fbd39403bc7955f73eca6a9524f2', window_index=i % 2,
        source_start=0 if i % 2 == 0 else 210, target='J' if i % 2 == 0 else 'L',
        diffusion_seed=SEEDS[i-152], actual_timestep=TIMESTEPS[i-152]) for i in range(152, 160)]


def source_binding():
    reviewed = cfg.source_binding()
    _, _, endpoint = b.completed_endpoint()
    rows = [json.loads(line) for line in METRICS.read_text(encoding='utf-8').splitlines() if line.strip()]
    require([r.get('step') for r in rows] == list(range(1, 21))
            and rows[-1]['micro_samples'] == fixed_samples(), 'actual B last-batch trace differs')
    census = read(CENSUS)
    stat = b.CHECKPOINT.stat()
    actual_stat = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    require(census.get('status') == 'completed' and census.get('B_source', {}).get('path') == str(b.CHECKPOINT)
            and census['B_source'].get('stat') == actual_stat, 'actual B census/stat binding required')
    require(b.checkpoint_digest() == reviewed['model']['checkpoint_sha256'], 'trusted B checkpoint identity changed')
    return dict(reviewed_B=reviewed, checkpoint_stat=actual_stat,
        metrics_path=str(METRICS), metrics_sha256=sha256(METRICS), samples=fixed_samples(),
        census_path=str(CENSUS), census_sha256=sha256(CENSUS),
        endpoint_files={name:sha256(b.ENDPOINT / (name+'.json')) for name in ('request', 'result', 'settlement')},
        parent_lineage=endpoint['execution']['parent_lineage'])


def validate_request(request, directory=None):
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=JOB_ID, max_seconds=MAX_SECONDS, cleanup_seconds=CLEANUP_SECONDS,
        stage_bytes_limit=STAGE_BYTES, protocol=protocol())
    require(set(request) == set(expected) | {'binding'} and all(request.get(k) == v for k, v in expected.items())
            and isinstance(request.get('binding'), dict), 'only fixed one-batch shadow update permitted')
    if directory is not None:
        require(contained(directory, ROOT) == ROOT / JOB_ID, 'fixed unique diagnostic directory required')


def prepare(*, output, source_revision=SOURCE_REVISION, max_seconds=MAX_SECONDS):
    output = contained(output, ROOT)
    require(output == ROOT / JOB_ID and not output.exists(), 'new fixed diagnostic output required')
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision,
        job_id=JOB_ID, max_seconds=max_seconds, cleanup_seconds=CLEANUP_SECONDS,
        stage_bytes_limit=STAGE_BYTES, protocol=protocol(), binding=source_binding())
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request,
        weights_loaded=False, latent_values_read=False, gpu_launched=False, automatic_next_stage=False))
    return request


def equal_tree(left, right, check=lambda: None):
    """Exact dtype/shape/value comparison, ignoring only tensor device; bounded CPU chunks."""
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and left.shape == right.shape):
            return False
        a, z = left.detach().reshape(-1), right.detach().reshape(-1)
        for start in range(0, a.numel(), CHUNK):
            check()
            if not torch.equal(a[start:start+CHUNK].cpu(), z[start:start+CHUNK].cpu()):
                return False
        return True
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right)
                and all(equal_tree(left[k], right[k], check) for k in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_tree(a, z, check) for a, z in zip(left, right))
    return type(left) is type(right) and left == right


def cpu_clone_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_clone_tree(v) for v in value)
    return copy.deepcopy(value)


def first_state(payload):
    """Exact ordered signature/group mapping; never infer parameter names from shapes."""
    import torch
    signature, saved = payload['model_signature'], payload['optimizer']
    groups = saved['param_groups']
    require([g.get('name') for g in groups] == ['action_adapter', 'controlled_action_gate'], 'B optimizer groups changed')
    names = [name for name, row in signature.items() if row['group'] == 'adapter']
    group = groups[0]
    require(len(names) == len(group['params']) == 6 and names[0] == PARAMETER and group['params'][0] == 0,
            'ordered first-conv optimizer mapping differs')
    require(group['lr'] == 1e-5 and group['weight_decay'] == .01 and tuple(group['betas']) == (.9, .95),
            'original AdamW8bit hyperparameters required')
    weight = payload['trainable_model'][PARAMETER]
    require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == SHAPE
            and signature[PARAMETER] == dict(shape=list(SHAPE), dtype='torch.bfloat16', group='adapter'),
            'B first-conv geometry/precision differs')
    state = saved['state'][0]
    quant = state.get('__bnb_optimizer_quant_state__', {})
    require(state.get('step') == 20 and all(k in quant for k in ('state1','state2','qmap1','qmap2','absmax1','absmax2')),
            'complete native step20 serialized quantized state required')
    for key in ('state1', 'state2'):
        require(quant[key].dtype == torch.uint8 and tuple(quant[key].shape) == SHAPE, 'original uint8 moments required')
    for key in ('qmap1', 'qmap2', 'absmax1', 'absmax2'):
        expected = (256,) if key.startswith('qmap') else (weight.numel() // 256,)
        require(quant[key].dtype == torch.float32 and tuple(quant[key].shape) == expected, 'original FP32 maps/scales required')
    one_group = copy.deepcopy(group)
    one_group['params'] = [0]
    if 'param_names' in one_group:
        require(one_group['param_names'] == names, 'saved explicit optimizer names differ')
        one_group['param_names'] = [PARAMETER]
    return weight, dict(state={0:state}, param_groups=[one_group])


def tensor_stats(value, check=lambda: None):
    import torch
    flat, squared, maximum, nonzero = value.detach().reshape(-1), 0., 0., 0
    for start in range(0, flat.numel(), CHUNK):
        check()
        part = flat[start:start+CHUNK].float().cpu()
        require(bool(torch.isfinite(part).all()), 'nonfinite full target tensor')
        squared += float(part.double().square().sum())
        maximum = max(maximum, float(part.abs().max()))
        nonzero += int(torch.count_nonzero(part))
    return dict(numel=flat.numel(), dtype=str(value.dtype), finite=True, nonzero=nonzero,
        rms=math.sqrt(squared / flat.numel()), l2=math.sqrt(squared), abs_max=maximum)


def update_statistics(initial, bf16, fp32, *, post_state_equal, check=lambda: None):
    import torch
    require(initial.dtype == bf16.dtype == torch.bfloat16 and fp32.dtype == torch.float32
            and initial.shape == bf16.shape == fp32.shape, 'matched finite shadow output geometry required')
    totals = {name:dict(changed=0, sum_sq=0., abs_max=0.) for name in ('bf16','fp32','fp32_projected')}
    weight_sq = difference_sq = dot = rounding_sq = 0.
    lost = projection_mismatch = 0
    for start in range(0, initial.numel(), CHUNK):
        check()
        w, lo, hi = [v.reshape(-1)[start:start+CHUNK].float().cpu() for v in (initial, bf16, fp32)]
        require(all(bool(torch.isfinite(v).all()) for v in (w, lo, hi)), 'nonfinite shadow update')
        projected = hi.bfloat16().float()
        dl, dh, dp = lo-w, hi-w, projected-w
        for name, delta in zip(totals, (dl, dh, dp)):
            totals[name]['changed'] += int(torch.count_nonzero(delta))
            totals[name]['sum_sq'] += float(delta.double().square().sum())
            totals[name]['abs_max'] = max(totals[name]['abs_max'], float(delta.abs().max()))
        weight_sq += float(w.double().square().sum())
        difference_sq += float((dh-dl).double().square().sum())
        rounding_sq += float((hi-lo).double().square().sum())
        dot += float((dl.double()*dh.double()).sum())
        lost += int(torch.count_nonzero((dh != 0) & (dl == 0)))
        projection_mismatch += int(torch.count_nonzero(lo != projected))
    n, weight_rms = initial.numel(), math.sqrt(weight_sq / initial.numel())
    for row in totals.values():
        row.update(changed_fraction=row['changed']/n, rms=math.sqrt(row['sum_sq']/n), l2=math.sqrt(row['sum_sq']))
        row['rms_relative_to_weight'] = row['rms']/weight_rms if weight_rms else None
    denom = totals['bf16']['l2'] * totals['fp32']['l2']
    matched = post_state_equal and projection_mismatch == 0
    interpretation = ('kernel_or_state_path_confounded' if not matched else
        'matched_single_step_writeback_rounding_observed' if rounding_sq > 0 else 'no_measurable_single_step_rounding_loss')
    return dict(numel=n, weight_rms=weight_rms, updates=totals, post_state_equal=post_state_equal,
        bf16_vs_recast_fp32_mismatch=projection_mismatch, fp32_nonzero_bf16_unchanged=lost,
        delta_difference_rms=math.sqrt(difference_sq/n), delta_cosine=dot/denom if denom else None,
        fp32_minus_bf16_weight_rms=math.sqrt(rounding_sq/n), rounding_interpretation=interpretation,
        cumulative_training_effect_verified=False, visual_cause_verified=False)


def restored_optimizer(parameter, source, betas, check):
    import bitsandbytes as bnb
    group = copy.deepcopy(source['param_groups'][0])
    group['params'] = [parameter]
    optimizer = bnb.optim.AdamW8bit([group], betas=betas)
    # Native loader deep-copies and unwraps the installed serialization. No manual casting.
    optimizer.load_state_dict(source)
    require(equal_tree(source, optimizer.state_dict(), check), 'state_restore_not_matched')
    return optimizer


def run(request, directory, lease):
    import torch
    import train_paired_turn_teacher as entry
    from bitsandbytes.optim.optimizer import Optimizer8bit
    from training.demo.backend import validate_active_authority
    from training.models.lora import load_trainable_state_dict
    from training.runtime import append_jsonl
    from torch.utils.data import default_collate
    validate_request(request, directory)
    directory = Path(directory)
    require(not (directory / 'diagnostic-result.json').exists(), 'no repeated execution of this identity')
    began, last_guard = time.monotonic(), 0.
    model = loader = batch = loss = named = target = optimizer = parameter = None
    payload = source_state = w0 = gradient = outputs = post_state = None
    report = dict(status='running', mode=MODE, revision=REVISION, request=request,
        model_loads=0, actual_loss_forward_calls=0, actual_backward_calls=0, original_model_optimizer_steps=0,
        shadow_optimizer_steps=0, cpu_native_restore_preflights=0, shadows={}, micro_samples=[],
        counters_exclude_activation_checkpoint_internal_recomputation=True, checkpoint_saved=False,
        video_generated=False, tensors_saved=False, training_continued=False, automatic_next_stage=False,
        future_clean_used_for_supervised_gradient_only=True, quality_improvement_verified=False)

    def phase(name, **extra):
        append_jsonl(directory / 'phases.jsonl', dict(phase=name, elapsed_seconds=time.monotonic()-began, **extra))

    def check():
        nonlocal last_guard
        now = time.monotonic()
        require(now-began < MAX_SECONDS-60, 'finite update diagnostic deadline reached')
        if now-last_guard > 2:
            require(b.a.matched.directory_bytes(ROOT) < STAGE_BYTES-1_000_000, 'independent100MB stage cap reached')
            validate_active_authority(lease)
            last_guard = now

    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
        require(source_binding() == request['binding'], 'actual B inputs/evidence changed')
        vendor = Path(inspect.getsourcefile(Optimizer8bit)).resolve()
        require(vendor == VENDOR and '__bnb_optimizer_quant_state__' in vendor.read_text(), 'installed native state loader source differs')
        report['vendor_source'] = dict(path=str(vendor), sha256=sha256(vendor))
        phase('checkpoint_load_begin')
        config = entry.load_config(b.TRAINING_CONFIG)
        hashes = entry.artifact_hashes(config, b.TRAINING_CONFIG)
        payload = torch.load(b.CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
        entry.validate_checkpoint(payload, config=config, hashes=hashes)
        entry.validate_base_identity(payload, config)
        _, _, endpoint = b.completed_endpoint()
        require(payload['stage'] == b.STAGE and payload['checkpoint_schema'] == b.CHECKPOINT_SCHEMA
                and payload['step'] == 20 and payload['new_stage_micro_batches_consumed'] == 160
                and payload['historical_micro_batches_before_stage'] == 320 and payload['lifetime_micro_batches_consumed'] == 480,
                'actual B20 endpoint required')
        for key in ('parent_lineage','sampling_contract','a_visual_gate','a_reference'):
            require(payload[key] == endpoint['execution'][key], 'actual B endpoint metadata differs: '+key)
        require(payload['metrics']['micro_samples'] == fixed_samples(), 'B checkpoint last-batch trace differs')
        w0, source_state = first_state(payload)
        report['checkpoint_lineage'] = dict(checkpoint=str(b.CHECKPOINT), schema=b.CHECKPOINT_SCHEMA,
            step=20, new_stage_micro_batches_consumed=160, parent_lineage=payload['parent_lineage'])
        report['target_weight_before'] = tensor_stats(w0, check)
        phase('cpu_native_restore_preflight_begin')
        # Fail before loading the 5B model if either dtype silently alters moments/scales.
        for dtype in (torch.bfloat16, torch.float32):
            parameter = torch.nn.Parameter(w0.to(dtype=dtype).clone())
            require(equal_tree(parameter.detach().bfloat16(), w0, check), 'initial represented weight differs')
            optimizer = restored_optimizer(parameter, source_state, config.optimizer.betas, check)
            report['cpu_native_restore_preflights'] += 1
            optimizer = parameter = None
            gc.collect()
        phase('cpu_native_restore_preflight_complete')
        check()
        require(torch.cuda.is_available(), 'guarded gradient diagnostic requires CUDA')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        device = torch.device('cuda', 0)
        phase('model_load_begin')
        model, _ = entry._build_model(config, device)
        load_trainable_state_dict(model, payload['trainable_model'])
        require(entry.pair._model_signature(model, 'frozen') == payload['model_signature'], 'real ordered model signature differs')
        named = dict(model.named_parameters())
        target = named[PARAMETER]
        live_groups = entry.pair._optimizer_groups(model, entry.model_config(config), 'frozen')
        require(live_groups[0]['params'][0] is target, 'actual optimizer first parameter mapping differs')
        del live_groups
        loader = entry.build_paired_turn_dataloader(config=config.data, training=config.training,
                                                   pool=payload['sampling_contract']['pool'])
        entry.teacher._restore_rng_state(payload, require_complete=True)
        model.zero_grad(set_to_none=True)
        report['model_loads'] = 1
        phase('model_load_complete')
        for record in fixed_samples():
            check()
            index = record['absolute_sample_index']
            spec = loader.dataset.sampling_spec(index)
            batch = default_collate([loader.dataset[index]])
            actual = dict(**spec, actual_timestep=float(batch['timesteps'][0, 1]))
            require(actual == record, 'actual real-gradient batch differs from recorded index/noise/timestep')
            phase('micro_forward_begin', sample_index=index)
            report['actual_loss_forward_calls'] += 1
            with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda', dtype=torch.bfloat16):
                loss = entry.teacher._forward_loss(model, batch, config, device)
            require(bool(torch.isfinite(loss)), 'nonfinite real-gradient loss')
            phase('micro_backward_begin', sample_index=index)
            report['actual_backward_calls'] += 1
            (loss / 8).backward()
            report['micro_samples'].append(dict(**actual, loss=float(loss.detach())))
            loss = batch = None
            phase('micro_backward_complete', sample_index=index)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        require(bool(torch.isfinite(norm)) and target.grad is not None and target.grad.dtype == torch.bfloat16,
                'finite original global clipping and BF16 target gradient required')
        report['global_clip'] = dict(norm_before_clip=float(norm), max_norm=1., coefficient=min(1., 1./(float(norm)+1e-6)),
            scope='all_trainable_Adapter_and_gate')
        require(equal_tree(target.detach(), w0, check), 'original model weight was updated')
        gradient = target.grad.detach().cpu().clone()
        report['clipped_gradient'] = tensor_stats(gradient, check)
        require(report['clipped_gradient']['nonzero'] > 0, 'zero actual target gradient is inconclusive')
        # All references to the large model/activation graph must be released before shadows.
        model.zero_grad(set_to_none=True)
        model = target = named = loader = batch = loss = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        phase('full_model_released', allocated_bytes=torch.cuda.memory_allocated())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        outputs, post_state = {}, None
        for label, dtype in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
            check()
            phase('shadow_restore_begin', shadow=label)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            parameter = torch.nn.Parameter(w0.to(device=device, dtype=dtype).clone())
            parameter.grad = gradient.to(device=device, dtype=dtype).clone()
            require(equal_tree(parameter.detach().bfloat16(), w0, check)
                    and equal_tree(parameter.grad.detach().float(), gradient.float(), check), 'shadow initial weights/gradient differ')
            optimizer = restored_optimizer(parameter, source_state, config.optimizer.betas, check)
            phase('shadow_step_begin', shadow=label)
            optimizer.step()
            report['shadow_optimizer_steps'] += 1
            torch.cuda.synchronize()
            actual_state = optimizer.state_dict()
            require(actual_state['state'][0]['step'] == 21, 'one shadow optimizer step20 to21 required')
            report['shadows'][label] = dict(initial_state_step=20, final_state_step=21,
                state_restore_matched=True, initial_weights_match=True, clipped_gradient_match=True,
                updated_weight=tensor_stats(parameter.detach(), check))
            outputs[label] = parameter.detach().cpu().clone()
            if label == 'bf16':
                post_state = cpu_clone_tree(actual_state)
            else:
                report['post_state_equal'] = equal_tree(post_state, actual_state, check)
            actual_state = optimizer = parameter = None
            gc.collect()
            torch.cuda.empty_cache()
            phase('shadow_step_complete', shadow=label)
        require(report['actual_loss_forward_calls'] == report['actual_backward_calls'] == 8
                and report['shadow_optimizer_steps'] == 2 and report['model_loads'] == 1,
                'fixed real-gradient execution counts differ')
        report['update_statistics'] = update_statistics(w0, outputs['bf16'], outputs['fp32'],
            post_state_equal=report['post_state_equal'], check=check)
        report.update(status='completed', state_restore_matched=True, gradient_equal=True,
            rounding_interpretation=report['update_statistics']['rounding_interpretation'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            interpretation='single_layer_single_batch_endpoint_diagnostic;not_video_cause_or_FP32_training_permission')
        check()
    except BaseException as error:
        report.update(status='failed', failure_type=type(error).__name__, error=str(error))
        phase('failed', failure_type=type(error).__name__, error=str(error))
        raise
    finally:
        model = loader = batch = loss = named = target = optimizer = parameter = None
        payload = source_state = w0 = gradient = outputs = post_state = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        report['elapsed_seconds'] = time.monotonic()-began
        require(len(json.dumps(report).encode('utf-8')) < 5_000_000, 'small JSON output limit exceeded')
        write_json(directory / 'diagnostic-result.json', report)
        phase('cleanup_complete')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    print(json.dumps(prepare(**vars(parser.parse_args()))))


if __name__ == '__main__':
    main()
