"""Exclusive fixed64 trajectory export/merge; never a synthetic cache receipt."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import torch

import train_completion_ode_sufficiency as entry
from training import completion_ode_probe_data_v2 as sources

require = entry.require
MANIFEST_SCHEMA = 'completion_ode_cache_manifest_v2'
ROW_SCHEMA = 'completion_ode_completed_trajectory_row_v2'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    return dict(path=str(path), sha256=entry.previous.sha256_file(path))


def pin(path):
    return dict(path=str(path), sha256=entry.previous.sha256_file(path))


def source_hashes():
    from training import completion_ode_wan_v2 as wan
    return {name: entry.previous.sha256_file(path) for name,path in dict(ode_core=entry.core().__file__,
        ode_entry=entry.__file__, ode_sources=sources.__file__, ode_wan=wan.__file__,
        ode_cache=__file__, ode_jobs=Path(__file__).with_name('completion_ode_jobs_v2.py'),
        causal_backbone=Path(__file__).parents[1]/'wan/modules/causal_model.py',
        wan_wrapper=Path(__file__).parents[1]/'utils/wan_wrapper.py').items()}


def prepare_manifest(parent):
    """CPU-only complete source preflight. Invoke before any reservation."""
    require(entry.parent_binding(parent['parent_step']) == parent, 'actual completed formalTF parent required')
    require(not torch.cuda.is_initialized(), 'manifest preparation may not initialize CUDA')
    root = entry.ROOT/'cache'
    path = root/'manifest.json'
    if path.exists():
        manifest = entry.read(path)
        validate_manifest(manifest, parent=parent)
        return pin(path)
    selection = sources.make_selection(parent)
    checked = sources.preflight_selection(parent, selection)
    selection_pin = write_new(root/'selection.json', selection)
    manifest = dict(schema=MANIFEST_SCHEMA, teacher_binding=copy.deepcopy(parent), selection=selection_pin,
        source_hashes=source_hashes(), grid=entry.core().teacher_grid().record(),
        teacher_solver='flow_euler', teacher_steps=40, teacher_cfg_scale=5.,
        teacher_packed_forward_calls_per_trajectory=80, namespace=entry.TRAJECTORY_NAMESPACE,
        saved_indices=[0, 10, 20, 30, 40], reference_enabled=False,
        context_kind='ground_truth_clean_completed_history_not_endpoint',
        target_kind='teacher_ode_endpoint_not_ground_truth', generated_history=False, **checked)
    validate_manifest(manifest, parent=parent)
    return write_new(path, manifest)


def validate_manifest(manifest, *, parent):
    require(manifest.get('schema') == MANIFEST_SCHEMA and manifest.get('teacher_binding') == parent
            and manifest.get('source_hashes') == source_hashes() and manifest.get('CPU_sources_verified') == 64
            and manifest.get('cuda_initialized') is False and manifest.get('model_loaded') is False,
            'fixed64 actualCPU source preflight/actual source hashes required')
    entry.validate_parent_binding(parent)
    selection = entry.small_pin(manifest.get('selection'))
    train = entry.choose_exact_windows(parent['qualified_data_binding'])
    probes = entry.validate_probe_manifest(selection, train)
    rows = manifest.get('rows')
    require(isinstance(rows, list) and len(rows) == 64 and manifest.get('saved_indices') == [0, 10, 20, 30, 40]
            and manifest.get('teacher_packed_forward_calls_per_trajectory') == 80
            and manifest.get('namespace') == entry.TRAJECTORY_NAMESPACE
            and manifest.get('target_kind') == 'teacher_ode_endpoint_not_ground_truth'
            and manifest.get('context_kind') == 'ground_truth_clean_completed_history_not_endpoint'
            and manifest.get('generated_history') is False and manifest.get('reference_enabled') is False,
            'all64 causal teacher endpoint/context rows required')
    entry.core().grid_from_record(manifest['grid'])
    for slot,(row,wanted) in enumerate(zip(rows, train+probes)):
        require(all(row.get(k) == v for k,v in wanted.items()) and row.get('global_slot') == slot
                and row.get('trajectory_seed') == sources.trajectory_seed(wanted)
                and all(entry.digest(row.get(k)) for k in ('clean_context_sha256_float32', 'action_sha256_float32',
                    'positive_text_sha256', 'negative_text_sha256')),
                'exact64 stable source/noise/tensor pins required')
    require(len({r['trajectory_seed'] for r in rows}) == 64, 'trajectory seed collision')
    return manifest


def original_sources(manifest, slots):
    """Known CPU dependencies checked again before a wrapper reserves."""
    parent = manifest['teacher_binding']
    validate_manifest(manifest, parent=parent)
    require(isinstance(slots, list) and slots and len(slots) == len(set(slots))
            and all(type(slot) is int and 0 <= slot < 64 for slot in slots), 'unique explicit fixed64 slots required')
    pool = sources.SourcePool(parent, entry.small_pin(manifest['selection']))
    negative, binding, _ = sources.load_negative(parent)
    require(binding == manifest['negative_binding'] and sources.tensor_sha(negative) == manifest['negative_text_sha256'],
            'actual original negative text asset changed')
    for slot in slots:
        row, value = manifest['rows'][slot], pool.load(manifest['rows'][slot])
        for key,name in (('clean_latents','clean_context_sha256_float32'), ('actions','action_sha256_float32'),
                         ('prompt_embeds','positive_text_sha256')):
            require(sources.tensor_sha(value[key]) == row[name], 'actual original source tensor changed: '+name)
    return pool, negative


def row_record_path(slot): return entry.ROOT/'cache'/f'trajectory-{slot:03d}.json'


def run_cache_shard(manifest_pin, *, slots, device, check, first_probe=False):
    """Only called after the real original guard reserve/check, by jobs_v2."""
    from training.completion_ode_wan_v2 import WanPackedAdapter, build_teacher
    manifest = entry.small_pin(manifest_pin)
    pool, negative = original_sources(manifest, slots)
    require(not first_probe or slots == [0], 'first real profile is exactly slot0, retained for full64')
    require(all(not row_record_path(slot).exists() and
                not (entry.ROOT/'cache'/f'trajectory-{slot:03d}.pt').exists() for slot in slots),
            'existing trajectory is never recomputed/overwritten')
    check(); load_started = time.monotonic()
    model, config, _ = build_teacher(manifest['teacher_binding'], device, check=check)
    model_load_seconds = time.monotonic()-load_started
    adapter = WanPackedAdapter(model, config, device, check=check)
    reports, probe = [], None
    try:
        for slot in slots:
            check(); row = manifest['rows'][slot]; original = pool.load(row)
            clean = original['clean_latents'][None].to(device=device, dtype=torch.bfloat16)
            actions = original['actions'][None].to(device=device, dtype=torch.bfloat16)
            positive = original['prompt_embeds'][None].to(device=device, dtype=torch.bfloat16)
            negative_gpu = negative[None].to(device=device, dtype=torch.bfloat16)
            generator = torch.Generator(device='cpu').manual_seed(row['trajectory_seed'])
            initial = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
            initial[:, :1] = clean[:, :1]
            if first_probe:
                probe = adapter.verify_causality(noisy=initial, clean_context=clean, actions=actions, positive_conditions=positive)
            before, began = adapter.forward_count, time.monotonic()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                trajectory = entry.core().packed_teacher_trajectory(adapter.callback, noisy_initial=initial,
                    clean_context=clean, actions=actions, positive_conditions=positive, negative_conditions=negative_gpu)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic()-began
            require(adapter.forward_count-before == 80, 'genuine packed40Euler two-CFG trajectory must use80 model calls')
            # Separate CPU allocations are intentional: no endpoint-as-GT alias.
            states = torch.stack([value[0].detach().cpu().clone() for value in trajectory.states])
            context = original['clean_latents'].bfloat16().clone()
            payload = dict(schema=entry.CACHE_SHARD_SCHEMA, teacher_binding=manifest['teacher_binding'],
                source_hashes=manifest['source_hashes'], **copy.deepcopy(row), states=states, clean_context=context,
                actions=original['actions'].clone(), prompt_embeds=original['prompt_embeds'].clone(),
                saved_indices=manifest['saved_indices'], grid=manifest['grid'])
            entry.validate_shard(payload, row=row, index=dict(teacher_binding=manifest['teacher_binding'],
                source_hashes=manifest['source_hashes'], grid=manifest['grid']))
            path = entry.ROOT/'cache'/f'trajectory-{slot:03d}.pt'
            with path.open('xb') as stream: torch.save(payload, stream)
            report = dict(schema=ROW_SCHEMA, status='completed', **copy.deepcopy(row),
                shard=dict(path=str(path), sha256=entry.previous.sha256_file(path), stat=entry.stat_identity(path)),
                actual_teacher_forward_calls=80, trajectory_seconds=elapsed, endpoint_context_separate=True,
                source_hashes=manifest['source_hashes'], manifest=manifest_pin)
            write_new(row_record_path(slot), report)
            reports.append(report)
            del trajectory, original, initial, clean, actions, positive, negative_gpu, states, context, payload
        peak = torch.cuda.max_memory_reserved(device)
        require(0 < peak < 23*1024**3, 'actual trajectory reserved memory must be below23GiB')
        result = dict(stage=entry.STAGE, trajectories=reports, slots=slots, model_load_seconds=model_load_seconds,
            actual_teacher_forward_calls=80 if first_probe else 80*len(slots), actual_total_model_forwards=adapter.forward_count,
            endpoint_context_separate=True, peak_reserved_bytes=peak, first_trajectory_profile=first_probe,
            quality_verified=False, realtime_verified=False)
        if probe is not None: result.update(probe)
        return result
    finally:
        del model
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def merge_cache(manifest_pin, *, first_trajectory_profile, cache_job_profiles):
    """CPU merge requires every actual cache-job settlement in the old ledger."""
    manifest = entry.small_pin(manifest_pin)
    validate_manifest(manifest, parent=manifest['teacher_binding'])
    first = entry.actual_profile(first_trajectory_profile, kind='trajectory')
    require(first.get('slots') == [0] and first.get('actual_teacher_forward_calls') == 80
            and first.get('causal_future_perturbation_passed') is True
            and first.get('causal_gradient_probe_passed') is True and first.get('actual_token_mask_audited') is True,
            'real retained slot0 packed causal probe/profile required')
    completed = []
    for profile in [first_trajectory_profile, *cache_job_profiles]:
        execution = entry.actual_profile(profile, kind='trajectory')
        require(execution.get('stage') == entry.STAGE and isinstance(execution.get('slots'), list),
                'completed realODE trajectory shard job required')
        completed.extend(execution['slots'])
    require(sorted(completed) == list(range(64)) and len(set(completed)) == 64,
            'all64 cache slots must have one completed settled producing job')
    rows = []
    for slot,wanted in enumerate(manifest['rows']):
        row = entry.read(row_record_path(slot))
        require(row.get('schema') == ROW_SCHEMA and row.get('status') == 'completed'
                and row.get('manifest') == manifest_pin and all(row.get(k) == v for k,v in wanted.items()),
                'completed trajectory source row differs from immutable selection')
        shard = row['shard']
        require(entry.stat_identity(shard['path']) == shard['stat']
                and entry.previous.sha256_file(shard['path']) == shard['sha256'], 'actual cached trajectory changed')
        payload = torch.load(shard['path'], weights_only=True, map_location='cpu', mmap=True)
        entry.validate_shard(payload, row=row, index=dict(teacher_binding=manifest['teacher_binding'],
            source_hashes=manifest['source_hashes'], grid=manifest['grid']))
        rows.append(row)
    fields = ('teacher_solver','teacher_steps','teacher_cfg_scale','teacher_packed_forward_calls_per_trajectory',
        'namespace','saved_indices','reference_enabled','context_kind','target_kind','generated_history','grid')
    index = dict(schema=entry.CACHE_INDEX_SCHEMA, status='completed', teacher_binding=manifest['teacher_binding'],
        source_hashes=manifest['source_hashes'], selection_sha256=manifest['selection']['sha256'],
        trajectories=rows, **{key:manifest[key] for key in fields})
    index_pin = write_new(entry.ROOT/'cache/index.json', index)
    receipt = dict(schema=entry.CACHE_RECEIPT_SCHEMA, status='completed', index_sha256=index_pin['sha256'],
        teacher_binding=manifest['teacher_binding'], source_hashes=manifest['source_hashes'],
        selection_sha256=manifest['selection']['sha256'], first_trajectory_profile=first_trajectory_profile,
        cache_job_profiles=cache_job_profiles)
    receipt_pin = write_new(entry.ROOT/'cache/receipt.json', receipt)
    binding = dict(index=index_pin, receipt=receipt_pin, selection=manifest['selection'],
        teacher_binding=manifest['teacher_binding'], source_hashes=manifest['source_hashes'])
    entry.validate_cache_binding(binding, teacher_binding=manifest['teacher_binding'])
    entry.validate_probe_supervision(binding, manifest['teacher_binding'])
    write_new(entry.ROOT/'cache/binding.json', binding)
    return binding
