"""Prepared CPU-side ODE80 contract, not permission to run an unfinished cache.

The parent is the real settled formal TF160, unless root has separately recorded
a technical-stop endpoint. ODE starts from its model weights with a NEW optimizer
and namespace; only ODE-internal resumes restore optimizer/RNG/cursor. A genuine
40-Euler CFG5 causal trajectory, not future ground truth, supplies its target.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

import train_completion_causal_tf_sufficiency as tf

previous = tf.previous
require = tf.require
STAGE = 'completion_causal_ode_sufficiency_v2'
CHECKPOINT_SCHEMA = 'completion_causal_ode_sufficiency_checkpoint_v2'
ROOT = tf.ROOT.parent/'ode-sufficiency-v2'
FACTORY = 'train_completion_ode_sufficiency:build_pair_dataloader'
NAMESPACE = 'completion_ode60_endpoint_seed42_v2'
TRAJECTORY_NAMESPACE = 'completion_ode64_trajectory_seed42_v2'
ENDPOINTS = (20, 40, 80)
MAX_STEPS, ACCUMULATION, MICRO_BATCHES = 80, 8, 640
CAP_GPU_SECONDS, CACHE_GPU_SECONDS, MAX_SEGMENT_STEPS = 16*3600, 6*3600, 16
FOLLOWING_GPU_SECONDS = (16+6+4)*3600  # paired DMD, runtime, supplementary evaluation; not new grants.
TRAIN_WINDOWS, PROBE_WINDOWS = 60, 4
CACHE_INDEX_SCHEMA = 'completion_causal_ode_trajectory_index_v2'
CACHE_SHARD_SCHEMA = 'completion_causal_ode_trajectory_shard_v2'
CACHE_RECEIPT_SCHEMA = 'completion_causal_ode_trajectory_receipt_v2'
CORE_MODULE = 'training.completion_ode_core_v2'


def core():
    """One shared mathematical core; no fallback to an imitation/GT flow loss."""
    return importlib.import_module(CORE_MODULE)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 16_000_000,
        'missing/unsafe small ODE record: '+str(path))
    result = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(result, dict), 'ODE JSON object required')
    return result


def digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def small_pin(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and digest(pin['sha256'])
        and Path(pin['path']).is_absolute() and '..' not in Path(pin['path']).parts,
        'absolute small-record SHA256 pin required')
    require(previous.sha256_file(pin['path']) == pin['sha256'], 'ODE small-record SHA256 changed')
    return read(pin['path'])


def stat_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'real non-symlink checkpoint/cache required')
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _technical_stop(step):
    """A narrower completed endpoint is never chosen from visual quality."""
    require(step in (40, 80), 'only explicit completed formal TF40/80 technical stop supported')
    path = ROOT/'parent-technical-stop.json'
    value = read(path)
    expected = dict(schema='completion_ode_root_technical_stop_v1', approved_by='root',
        requested_parent_step=step, default_parent_step=160, reason_class='technical_stop_not_visual_selection')
    require(all(value.get(k) == v for k,v in expected.items()) and isinstance(value.get('reason'), str)
        and value['reason'].strip(), 'root technical-stop record required; not a caller-selected teacher')
    failure = small_pin(value.get('failure_record'))
    require(failure.get('status') in ('failed', 'stopped')
        and isinstance(failure.get('failure_type', failure.get('error')), str),
        'actual failed/stopped technical record required')
    return dict(path=str(path), sha256=previous.sha256_file(path))


def parent_binding(step=160):
    from training.studio import completion_causal_sufficiency_train as studio
    require(type(step) is int and step in (40, 80, 160), 'formal TF parent endpoint required, never defaultTF20')
    approval = None if step == 160 else _technical_stop(step)
    request, execution, _ = studio.completed_endpoint(step)
    paths = studio.paths(step)
    require(execution['step'] == step and execution['stage'] == tf.STAGE
        and execution['checkpoint_schema'] == tf.CHECKPOINT_SCHEMA, 'actual formal TF endpoint required')
    return dict(mode='formal_tf_model_weights_only_new_ode_optimizer_namespace',
        parent_stage=tf.STAGE, parent_checkpoint_schema=tf.CHECKPOINT_SCHEMA, parent_step=step,
        parent_optimizer_internal_step=step, parent_checkpoint=str(paths['checkpoint']),
        parent_checkpoint_stat=stat_identity(paths['checkpoint']), parent_config=str(paths['config']),
        parent_config_sha256=request['config_sha256'], parent_job=str(paths['endpoint']),
        parent_records=studio.record_binding(paths['endpoint']),
        base_identity=copy.deepcopy(execution['base_identity']),
        ancestor_lineage=copy.deepcopy(execution['parent_lineage']),
        tf_schema_migration=copy.deepcopy(execution['schema_migration']),
        qualified_data_binding=copy.deepcopy(request['data_binding']),
        historical_micro_batches_before_stage=execution['lifetime_micro_batches_consumed'],
        parent_sampling_cursor=execution['sampling_cursor'], technical_stop_approval=approval,
        optimizer_restored=False, rng_restored=False, new_optimizer=True, parent_strict_resume=False,
        reference_enabled=False, quality_certified=False)


def validate_parent_binding(binding):
    from training.studio import completion_causal_sufficiency_train as studio
    require(isinstance(binding, dict) and binding.get('parent_step') in (40, 80, 160),
        'formal TF160 or root technical-stop parent required')
    step = binding['parent_step']
    paths = studio.paths(step)
    expected = dict(mode='formal_tf_model_weights_only_new_ode_optimizer_namespace', parent_stage=tf.STAGE,
        parent_checkpoint_schema=tf.CHECKPOINT_SCHEMA, parent_optimizer_internal_step=step,
        parent_checkpoint=str(paths['checkpoint']), parent_config=str(paths['config']), parent_job=str(paths['endpoint']),
        historical_micro_batches_before_stage=tf.HISTORY+step*8, parent_sampling_cursor=step*8,
        optimizer_restored=False, rng_restored=False, new_optimizer=True, parent_strict_resume=False,
        reference_enabled=False, quality_certified=False)
    require(all(binding.get(k) == v for k,v in expected.items()), 'explicit newODE optimizer/namespace lineage differs')
    require((step == 160 and binding.get('technical_stop_approval') is None)
        or (step != 160 and isinstance(binding.get('technical_stop_approval'), dict)),
        'non160 parent requires actual root technical-stop approval')
    require(set(binding.get('parent_checkpoint_stat', {})) == {'bytes', 'mtime_ns'}
        and all(type(v) is int and v > 0 for v in binding['parent_checkpoint_stat'].values())
        and set(binding.get('parent_records', {})) == {n+'_sha256' for n in ('request', 'result', 'settlement')}
        and all(digest(v) for v in [binding.get('parent_config_sha256'), *binding['parent_records'].values()]),
        'full actual parent checkpoint/config/settled pins required')
    previous.validate_lineage(binding.get('ancestor_lineage'))
    tf.validate_bridge_binding(binding.get('tf_schema_migration'))
    require(binding['ancestor_lineage']['qualified_data_binding'] == binding.get('qualified_data_binding')
        and binding['ancestor_lineage']['base_identity'] == binding.get('base_identity'), 'original TF asset pins changed')


def load_parent(binding):
    validate_parent_binding(binding)
    require(parent_binding(binding['parent_step']) == binding, 'actual formal TF parent changed/not completed')
    config = tf.load_config(binding['parent_config'])
    data = binding['qualified_data_binding']
    bridge = binding['tf_schema_migration']
    payload = torch.load(binding['parent_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    tf.validate_checkpoint(payload, config=config,
        hashes=tf.artifact_hashes(config, binding['parent_config'], data, bridge), data_binding=data,
        sampling=tf.sampling_contract(config.training, data), bridge_binding=bridge)
    previous.validate_base_identity(payload, config)
    require(payload['step'] == binding['parent_step'] and payload['sampling_cursor'] == binding['parent_sampling_cursor']
        and stat_identity(binding['parent_checkpoint']) == binding['parent_checkpoint_stat'], 'actual TF source changed')
    # Deliberately no parent optimizer/RNG export: new ODE objective/namespace.
    return dict(trainable_model=payload['trainable_model'], model_signature=payload['model_signature'],
        frozen_action_model=payload['frozen_action_model'], frozen_action_signature=payload['frozen_action_signature'])


@dataclass
class ODEConfig(previous.CompletionConfig):
    def validate(self):
        require(type(self) is ODEConfig, 'independent ODE80 configuration required')
        require(self.data.data_factory == FACTORY and self.training.max_steps == MAX_STEPS
            and self.training.checkpoint_every == 16 and self.training.keep_last == 4
            and self.training.keep_best == 0 and Path(self.training.output_dir) == ROOT/'training',
            'fixed80/newODE factory/output contract required')
        view = previous.CompletionConfig(**{key:copy.deepcopy(getattr(self, key))
            for key in ('model', 'data', 'optimizer', 'training', 'error_recycling')})
        view.data.data_factory = previous.FACTORY
        view.training.max_steps = view.training.checkpoint_every = 20
        view.training.output_dir = str(previous.ROOT/previous.PROFILE_JOB/'training')
        previous.validate_probe_config(view)


def config_from_parent(parent_config, cache_binding):
    parent_config.validate()
    value = ODEConfig(**{key:copy.deepcopy(getattr(parent_config, key))
        for key in ('model', 'data', 'optimizer', 'training', 'error_recycling')})
    value.data.data_factory = FACTORY
    value.data.manifest_path = cache_binding['index']['path']
    value.data.manifest_sha256 = cache_binding['index']['sha256']
    value.training.max_steps, value.training.checkpoint_every = MAX_STEPS, 16
    value.training.output_dir = str(ROOT/'training')
    value.validate()
    return value


def load_config(path):
    import yaml
    source = Path(path).read_text(encoding='utf-8')
    try: raw = json.loads(source)
    except json.JSONDecodeError: raw = yaml.safe_load(source)
    require(isinstance(raw, dict) and set(raw) == {'model', 'data', 'optimizer', 'training', 'error_recycling'},
        'ODE config fields differ')
    raw['data']['canonical_action_keys'], raw['optimizer']['betas'] = tuple(raw['data']['canonical_action_keys']), tuple(raw['optimizer']['betas'])
    value = ODEConfig(model=previous.CausalModelConfig(**raw['model']), data=previous.CausalDataConfig(**raw['data']),
        optimizer=previous.CausalOptimizerConfig(**raw['optimizer']), training=previous.CompletionTrainingConfig(**raw['training']),
        error_recycling=previous.ErrorRecyclingConfig(**raw['error_recycling']))
    value.validate()
    return value


def choose_exact_windows(binding):
    """All sixty pre-reviewed train windows, never an eight-window substitute."""
    selection = previous.validate_data_binding(binding)
    module = previous.common.data_module()
    require(len(selection.rows) == TRAIN_WINDOWS, 'all sixty reviewed training windows required')
    return [dict(**dict(zip(module.IDENTITY_FIELDS, module.window_key(row))), target=row['target'],
        split='train', slot=index) for index,row in enumerate(selection.rows)]


def validate_probe_manifest(manifest, train_rows):
    """Explicit original qualification assets; repeatedly used, not final holdout."""
    require(manifest.get('schema') == 'completion_ode_formal64_selection_v2'
        and manifest.get('training_rows') == train_rows
        and manifest.get('independent_final_test') is False, 'fixed60train/fourdevelopment probe manifest required')
    rows = manifest.get('probe_rows')
    require(isinstance(rows, list) and len(rows) == PROBE_WINDOWS, 'all four fixed qualification probes required')
    from training.studio import balanced_lora_pair_eval as baseline
    wanted = tuple(('pair-j', 'pair-l', 'holdout-j', 'holdout-l'))
    identities = previous.common.data_module().IDENTITY_FIELDS
    train_episodes = {row['episode_id'] for row in train_rows}
    for slot,(row,case) in enumerate(zip(rows,wanted)):
        episode,start,target,_ = baseline.FIXED[case]
        require(row.get('case_id') == case and row.get('episode_id') == episode
            and row.get('source_start') == start and row.get('target') == target
            and row.get('split') == 'probe' and row.get('slot') == slot
            and row['episode_id'] not in train_episodes and all(key in row for key in identities),
            'original fixed qualification source cannot overlap training or be reselected')
        require(isinstance(row.get('source_asset_binding'), dict)
            and row['source_asset_binding'].get('future_GT_supervision_only') is True
            and row['source_asset_binding'].get('future_GT_used_for_generation') is False,
            'separate actual supervision asset binding required, never a generation-loader future leak')
    return rows


def validate_cache_binding(binding, *, teacher_binding):
    validate_parent_binding(teacher_binding)
    require(isinstance(binding, dict) and set(binding) == {'index', 'receipt', 'selection', 'teacher_binding', 'source_hashes'}
        and binding['teacher_binding'] == teacher_binding, 'separate ODE index/receipt/teacher/source binding required')
    index, receipt = small_pin(binding['index']), small_pin(binding['receipt'])
    require(index.get('schema') == CACHE_INDEX_SCHEMA and receipt.get('schema') == CACHE_RECEIPT_SCHEMA
        and index.get('status') == receipt.get('status') == 'completed'
        and receipt.get('index_sha256') == binding['index']['sha256']
        and receipt.get('teacher_binding') == index.get('teacher_binding') == teacher_binding
        and receipt.get('source_hashes') == index.get('source_hashes') == binding['source_hashes']
        and receipt.get('selection_sha256') == index.get('selection_sha256') == binding['selection']['sha256'],
        'completed actual trajectory cache and source pins required')
    require(binding['source_hashes'].get('ode_core') == previous.sha256_file(core().__file__)
        and all(digest(value) for value in binding['source_hashes'].values()), 'actual sharedODE mathcore pin required')
    expected = dict(teacher_solver='flow_euler', teacher_steps=40, teacher_cfg_scale=5.,
        teacher_packed_forward_calls_per_trajectory=80, namespace=TRAJECTORY_NAMESPACE,
        saved_indices=[0, 10, 20, 30, 40], reference_enabled=False,
        context_kind='ground_truth_clean_completed_history_not_endpoint',
        target_kind='teacher_ode_endpoint_not_ground_truth', generated_history=False)
    require(all(index.get(key) == value for key,value in expected.items()), 'causal40/CFG5 endpoint/context contract differs')
    expected_rows = choose_exact_windows(teacher_binding['qualified_data_binding'])
    selection = small_pin(binding['selection'])
    expected_rows += validate_probe_manifest(selection, expected_rows)
    core().grid_from_record(index.get('grid', {}))
    rows = index.get('trajectories')
    require(isinstance(rows, list) and len(rows) == TRAIN_WINDOWS+PROBE_WINDOWS,
        'all60 training+four probe trajectories required, no silent pilot subset')
    identities = previous.common.data_module().IDENTITY_FIELDS
    for row,wanted in zip(rows,expected_rows):
        require(all(row.get(key) == wanted[key] for key in (*identities, 'target', 'split', 'slot'))
            and row.get('status') == 'completed' and row.get('actual_teacher_forward_calls') == 80,
            'exact stable selection/complete80 teacher forwards required')
        require(set(row.get('shard', {})) == {'path', 'sha256', 'stat'} and digest(row['shard']['sha256'])
            and Path(row['shard']['path']).is_absolute()
            and Path(row['shard']['path']).is_relative_to(ROOT/'cache')
            and '..' not in Path(row['shard']['path']).parts, 'new remoteODE cache path/hash required')
    profile = actual_profile(receipt.get('first_trajectory_profile'), kind='trajectory')
    require(profile.get('actual_teacher_forward_calls') == 80
        and profile.get('causal_future_perturbation_passed') is True
        and profile.get('causal_gradient_probe_passed') is True
        and profile.get('actual_token_mask_audited') is True
        and profile.get('endpoint_context_separate') is True
        and 0 < profile.get('peak_reserved_bytes', 0) < 23*1024**3,
        'real first trajectory numerical/resource/settlement profile required')
    return index


def actual_profile(binding, *, kind):
    require(isinstance(binding, dict) and set(binding) == {'request', 'result', 'settlement'},
        'real profile request/result/settlement pins required, not a success declaration')
    directories = {Path(binding[key]['path']).parent for key in ('request', 'result', 'settlement')}
    require(len(directories) == 1 and next(iter(directories)).is_relative_to(ROOT)
        and all(Path(binding[key]['path']).name == key+'.json' for key in binding),
        'actual profile request/result/settlement must belong to one newODE job')
    request,result,settlement = (small_pin(binding[key]) for key in ('request', 'result', 'settlement'))
    fee_overhead=60
    from training import completion_parallel_v4 as v4
    if v4.is_v4(request.get('parallel_budget_binding')):
        manifest=v4.manifest(request['parallel_budget_binding'])
        owned=[r for r in manifest['jobs'] if r['job_id']==request.get('job_id')]
        require(len(owned)==1 and owned[0]['mode']==request.get('mode')
            and owned[0]['max_seconds']==request.get('contract_max_seconds'), 'actual v4 fee scope differs')
        fee_overhead=v4.FEE_OVERHEAD
    require(result.get('status') == 'completed' and settlement.get('status') == 'settled'
        and settlement.get('outcome') == 'completed' and result.get('mode') == request.get('mode')
        and isinstance(request.get('mode'), str) and request['mode']
        and isinstance(request.get('job_id'), str) and request['job_id']
        and 0 < settlement.get('charged_gpu_seconds', 0) <= request.get('max_seconds', 0)+fee_overhead
        and request.get('cleanup_seconds') == 60, 'actual completed/settled boundedODE profile required')
    from training.studio import completion_causal_sufficiency_train as studio
    ledger,_ = studio.ledger_snapshot()
    reservation = ledger['reservations'].get(settlement.get('reservation_id'), {})
    context = reservation.get('context', {})
    directory = next(iter(directories))
    require(reservation.get('status') == 'settled' and context.get('job_id') == request['job_id']
        and context.get('job_directory') == str(directory)
        and context.get('request_sha256') == binding['request']['sha256']
        and reservation.get('charged_seconds') == settlement['charged_gpu_seconds']
        and reservation.get('settlement') == settlement,
        'actualODE profile must match original persistentCreator ledger settlement')
    execution = result.get('execution', {})
    if kind == 'training':
        require(execution.get('stage') == STAGE and execution.get('step') == 2, 'actualODE2 profile required')
        return execution.get('qualification', {})
    if kind == 'evaluation':
        require(execution.get('stage') == STAGE
            and isinstance(execution.get('comparison_identities'), list)
            and execution['comparison_identities']
            and all(isinstance(value, str) for value in execution['comparison_identities'])
            and 60 <= request.get('max_seconds', 0) <= 3600,
            'actual completedODE numerical/video/control record required')
        return execution
    require(kind == 'trajectory' and 60 <= request.get('max_seconds', 0) <= 3600,
        'actual first40Euler packed trajectory profile required')
    return execution


def validate_shard(payload, *, row, index, expected_latent_shape=(25, 48, 30, 52)):
    require(isinstance(payload, dict) and payload.get('schema') == CACHE_SHARD_SCHEMA
        and payload.get('teacher_binding') == index['teacher_binding']
        and payload.get('source_hashes') == index['source_hashes'], 'trajectory shard teacher/source differs')
    identities = previous.common.data_module().IDENTITY_FIELDS
    require(all(payload.get(key) == row[key] for key in (*identities, 'target', 'split', 'slot'))
        and payload.get('trajectory_seed') == row.get('trajectory_seed')
        and type(payload.get('trajectory_seed')) is int, 'trajectory source/noise identity differs')
    states, context = payload.get('states'), payload.get('clean_context')
    require(torch.is_tensor(states) and torch.is_tensor(context)
        and tuple(states.shape) == (5, *expected_latent_shape) and tuple(context.shape) == expected_latent_shape
        and states.dtype == context.dtype == torch.bfloat16
        and states.device.type == context.device.type == 'cpu'
        and states.untyped_storage().data_ptr() != context.untyped_storage().data_ptr()
        and bool(torch.isfinite(states).all()) and bool(torch.isfinite(context).all()),
        'separate finite BF16 teacher states and GT context required')
    require(all(torch.equal(states[slot, :1], context[:1]) for slot in range(5)), 'first latent must stay clean/protected')
    require(payload.get('saved_indices') == [0, 10, 20, 30, 40]
        and payload.get('grid') == index.get('grid'),
        'stored actual integration sigma and model timestep differ from actual teacher grid')
    core().grid_from_record(payload['grid'])
    actions, text = payload.get('actions'), payload.get('prompt_embeds')
    require(torch.is_tensor(actions) and actions.shape == (96, 8) and actions.device.type == 'cpu'
        and bool(torch.isfinite(actions).all()) and bool(((actions == 0)|(actions == 1)).all())
        and torch.is_tensor(text) and text.ndim == 2 and 1 <= text.shape[0] <= 512 and text.shape[1] == 4096
        and text.device.type == 'cpu' and bool(torch.isfinite(text).all()), 'actual96x8 actions/staticpositive text required')
    require(payload.get('positive_text_sha256') == row.get('positive_text_sha256')
        and payload.get('negative_text_sha256') == row.get('negative_text_sha256')
        and all(digest(payload.get(k)) for k in ('positive_text_sha256', 'negative_text_sha256')),
        'positive/negative cache text asset pins required')
    require(not any(key in payload for key in ('ref_latents', 'ref_mask', 'target_flow', 'y', 'clip_fea')),
        'no extra reference/GTflow conditions permitted in ODE cache')
    return payload


class ODETrajectoryDataset(Dataset):
    def __init__(self, cache_binding, teacher_binding, *, purpose='train'):
        require(purpose in ('train', 'probe'), 'explicit ODE split required')
        self.index = validate_cache_binding(cache_binding, teacher_binding=teacher_binding)
        self.cache_binding, self.teacher_binding, self.purpose = cache_binding, teacher_binding, purpose
        self.rows = [row for row in self.index['trajectories'] if row['split'] == purpose]
        require(len(self.rows) == (TRAIN_WINDOWS if purpose == 'train' else PROBE_WINDOWS),
            'fixed60train/fourprobe split required')
        self.num_samples = MICRO_BATCHES if purpose == 'train' else PROBE_WINDOWS*4
        self._verified = set()
        self.source_dataset = tf.build_dataset(teacher_binding['qualified_data_binding'])
        self.source_rows = {previous.common.data_module().window_key(row):row for row in self.source_dataset.episodes}
        self.contract = dict(namespace=NAMESPACE, virtual_samples=self.num_samples,
            window_policy='m_modN_exact_window_state_floor_m_divN_mod4',
            train_windows=TRAIN_WINDOWS, probe_windows=PROBE_WINDOWS, levels=4, trajectories_per_window=1,
            new_optimizer=True, clean_history=True, independent_generalization=False)

    def __len__(self): return self.num_samples

    def sampling_spec(self, index):
        require(type(index) is int and 0 <= index < len(self), 'invalid ODE absolute micro cursor')
        count = len(self.rows)
        slot, level = index%count, (index//count)%4
        row = self.rows[slot]
        identities = previous.common.data_module().IDENTITY_FIELDS
        teacher_index = [0, 10, 20, 30][level]
        grid = core().grid_from_record(self.index['grid'])
        return dict(absolute_sample_index=index, index_namespace=NAMESPACE, window_slot=slot, state_slot=level,
            teacher_trajectory_index=[0, 10, 20, 30][level], **{key:row[key] for key in identities},
            source_target=row['target'], trajectory_seed=row['trajectory_seed'],
            integration_sigma=grid.sigmas[teacher_index], model_timestep=grid.model_timesteps[teacher_index])

    def __getitem__(self, index):
        spec = self.sampling_spec(index)
        row = self.rows[spec['window_slot']]
        shard = row['shard']; path = Path(shard['path'])
        require(stat_identity(path) == shard['stat'], 'actual trajectory shard changed')
        if path not in self._verified:
            require(previous.sha256_file(path) == shard['sha256'], 'trajectory states/context asset hash differs')
            self._verified.add(path)
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        validate_shard(payload, row=row, index=self.index)
        if self.purpose == 'train':
            key = previous.common.data_module().window_key(row)
            require(key in self.source_rows, 'ODE cached window absent from actual parent sixty-window data')
            source_row = self.source_rows[key]
            source = self.source_dataset._cached_window(source_row, source_row['window_index'])
            require(torch.equal(source['clean_latents'].bfloat16(), payload['clean_context'])
                and torch.equal(source['actions'].float(), payload['actions'].float())
                and torch.equal(source['prompt_embeds'].float(), payload['prompt_embeds'].float()),
                'actual cachedGT context/actions/positive text differ from original reviewed assets')
            del source
        else:
            from training.completion_ode_probe_data_v2 import load_probe_window
            source = load_probe_window(row, self.teacher_binding)
            require(torch.equal(source['clean_latents'].bfloat16(), payload['clean_context'])
                and torch.equal(source['actions'].float(), payload['actions'].float())
                and torch.equal(source['prompt_embeds'].float(), payload['prompt_embeds'].float()),
                'independent original probeGT/context/actions/text changed')
            del source
        # GT context and ODE endpoint remain different objects and names. Only
        # the supervised training core can accept clean_context, never generate.
        return dict(trajectory_states=payload['states'].clone(), clean_context=payload['clean_context'].clone(),
            actions=payload['actions'].float().clone(), prompt_embeds=payload['prompt_embeds'].float().clone(),
            round_index=spec['state_slot'])


def build_pair_dataloader(*, config, training, cache_binding, teacher_binding):
    from torch.utils.data import DataLoader
    require(config.data_factory == FACTORY and training.max_steps == MAX_STEPS,
        'independent ODE80 loader contract required')
    return DataLoader(ODETrajectoryDataset(cache_binding, teacher_binding), batch_size=1,
        shuffle=False, num_workers=0, drop_last=True)


def counters(step, parent):
    validate_parent_binding(parent)
    require(type(step) is int and 0 <= step <= MAX_STEPS, 'invalid ODE optimizer step')
    history = parent['historical_micro_batches_before_stage']
    return dict(step=step, optimizer_internal_step=step, micro_batches_consumed=step*8,
        new_stage_micro_batches_consumed=step*8, sampling_cursor=step*8,
        historical_micro_batches_before_stage=history, lifetime_micro_batches_consumed=history+step*8)


def sampling_contract(dataset, parent):
    require(dataset.purpose == 'train' and len(dataset) == MICRO_BATCHES, 'fixed640 ODE training draws required')
    return dict(namespace=NAMESPACE, dataset=copy.deepcopy(dataset.contract), first_stage_sample_index=0,
        last_stage_sample_index=639, virtual_samples=640, sampling_restarts_at_each_segment=False,
        teacher_binding=copy.deepcopy(parent), cache_binding=copy.deepcopy(dataset.cache_binding),
        parent_strict_resume=False, new_optimizer=True, reference_enabled=False,
        train_windows=TRAIN_WINDOWS, probe_windows=PROBE_WINDOWS,
        mean_window_exposures_at20=160/TRAIN_WINDOWS, mean_window_exposures_at40=320/TRAIN_WINDOWS,
        mean_window_exposures_at80=640/TRAIN_WINDOWS, exposure_counts_are_not_complete_epochs=True)


def method_contract():
    return dict(method=STAGE, objective='future_only_teacher_ode_endpoint_MSE',
        prediction='z_s_float_minus_actual_integration_sigma_times_velocity_float', target='teacher_endpoint_not_GT',
        inference_sampler='four_endpoint_predictions_with_flow_renoise_not_Euler4',
        training_context='causal_ground_truth_clean_completed_blocks', cfg_baked_from_teacher=5.,
        student_training_positive_branch_only=True, student_inference_external_cfg=1.,
        reference_enabled=False, action_adapter_frozen=True, fp32_gate_frozen=True, wan_backbone_frozen=True,
        fp32_rank16_alpha16_lora=True, lora_lr=2e-6, new_optimizer=True, parent_strict_resume=False,
        strict_internal_resume=True, self_rollout_training=False, distribution_matching_distillation=False,
        quality_validated=False, training_length_adequacy_proven=False, realtime_validated=False)


def next_segment(start, slowest):
    require(type(start) is int and 2 <= start < MAX_STEPS
        and type(slowest) in (int, float) and math.isfinite(slowest) and slowest > 0,
        'actual completedODE cursor and measured step time required')
    endpoint = next(value for value in ENDPOINTS if value > start)
    maximum = min(MAX_SEGMENT_STEPS, endpoint-start)
    count = next((n for n in range(maximum, 0, -1) if math.ceil(slowest*n*1.35+180) <= 3600), None)
    require(count is not None, 'oneODE update cannot fit3600; no extension')
    return dict(start_step=start, stop_after_step=start+count, new_steps=count,
        next_comparison_endpoint=endpoint, measured_slowest_step_seconds=float(slowest),
        max_seconds=math.ceil(slowest*count*1.35+180))


def validate_budget_contract(value, max_seconds, *, current_start_step, current_stop_step):
    fields = {'scope', 'stage_cap_gpu_seconds', 'settled_gpu_seconds',
        'remaining_planned_gpu_seconds', 'current_job_cleanup_seconds', 'ledger_sha256', 'remaining_schedule'}
    require(isinstance(value, dict) and set(value) in (fields, fields|{'parallel_budget_binding'})
        and value['scope'] == 'completion_ode_cache_train_validation_eval_combined'
        and value['stage_cap_gpu_seconds'] == CAP_GPU_SECONDS and value['current_job_cleanup_seconds'] == 60,
        'original16GPUh combinedODE cache/train/eval accounting required')
    charged, remaining = value['settled_gpu_seconds'], value['remaining_planned_gpu_seconds']
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (charged, remaining))
        and type(max_seconds) is int and 60 <= max_seconds <= 3600
        and charged+max_seconds+60+remaining <= CAP_GPU_SECONDS and digest(value['ledger_sha256']),
        'ODE finite budget insufficient; no new grant')
    schedule = small_pin(value['remaining_schedule'])
    require(schedule.get('schema') == 'completion_ode_complete_remaining_schedule_v2'
        and schedule.get('root') == str(ROOT) and schedule.get('final_optimizer_step') == MAX_STEPS
        and schedule.get('fixed_training_comparison_endpoints') == list(ENDPOINTS)
        and schedule.get('fixed_cache_trajectory_count') == TRAIN_WINDOWS+PROBE_WINDOWS
        and schedule.get('fixed_probe_count') == PROBE_WINDOWS,
        'actual immutable fullODE80/cache64/probe4 schedule required, not zero remaining-cost declaration')
    require(schedule.get('current_start_step') == current_start_step
        and schedule.get('current_stop_step') == current_stop_step
        and type(current_start_step) is int and type(current_stop_step) is int
        and 0 <= current_start_step < current_stop_step <= MAX_STEPS,
        'immutable schedule must bind the actual currentODE segment')
    rows = schedule.get('remaining_jobs')
    require(isinstance(rows, list) and all(isinstance(row, dict)
        and type(row.get('max_seconds')) is int and 60 <= row['max_seconds'] <= 3600
        and row.get('cleanup_seconds') == 60 for row in rows)
        and sum(row['max_seconds']+60 for row in rows) == remaining,
        'remainingODE worst cost must equal actual complete finite schedule')
    cursor = current_stop_step
    for row in rows:
        require(row.get('kind') in ('training', 'numeric_validation', 'video_evaluation', 'untrained_control'),
            'remainingODE work must declare its real stage, not an unidentified cost')
        if row['kind'] == 'training':
            stop = row.get('stop_after_step')
            require(row.get('start_step') == cursor and type(stop) is int
                and cursor < stop <= min(cursor+MAX_SEGMENT_STEPS, MAX_STEPS)
                and not any(cursor < endpoint < stop for endpoint in ENDPOINTS),
                'full remainingODE trajectory must preserve every update and comparison endpoint')
            cursor = stop
    require(cursor == MAX_STEPS, 'zero/partial remaining train cost cannot omitODE80')
    # Evaluations may already have settled before a later segment. Bind those
    # actual records, rather than treating a metadata claim as paid work.
    completed = schedule.get('completed_evaluations')
    require(isinstance(completed, list), 'actual settled comparison records must be explicit')
    covered = []
    for row in completed:
        require(isinstance(row, dict) and row.get('kind') in
            ('numeric_validation', 'video_evaluation', 'untrained_control'), 'invalid priorODE comparison')
        profile = actual_profile(row.get('records'), kind='evaluation')
        require(profile.get('comparison_identities') == row.get('comparison_identities'),
            'actual settledODE comparison identity changed')
        covered += row['comparison_identities']
    for row in rows:
        if row['kind'] != 'training':
            require(isinstance(row.get('comparison_identities'), list)
                and row['comparison_identities']
                and all(isinstance(value, str) for value in row['comparison_identities']),
                'actual remainingODE numeric/video/control identities required')
            covered += row['comparison_identities']
    cases = ('pair-j-original', 'pair-j-jl_swap', 'holdout-l-original', 'holdout-l-jl_swap')
    required = [f'ode{step}-numeric-probe4-level4' for step in ENDPOINTS]
    required += [f'ode{step}-{case}' for step in ENDPOINTS for case in cases]
    required += ['untrained-'+case for case in cases]
    require(len(covered) == len(set(covered)) and set(covered) == set(required),
        'complete fixed20/40/80 numerical/videos and same4round untrained control must remain funded')


def reconcile_real_budget(value, lease, *, max_seconds):
    """No caller-supplied zero-history value can stand in for the old ledger."""
    from training.creator.guard import GuardConfig, CreatorGuard
    from training.studio.controlled_action_train import RUNTIME
    cfg = GuardConfig.load(RUNTIME/'guard-creator100-gpu2.json')
    require(cfg.ledger == RUNTIME/'budget.json' and cfg.max_round_gpu_hours == 100
        and cfg.budget_scope == 'creator_100h_500gb_20261003' and cfg.supplemental_gpu_authority is None,
        'unchanged originalCreator100 ledger authority required')
    guard = CreatorGuard(cfg); guard._authority(); ledger = guard._ledger()
    if value.get('parallel_budget_binding') is not None:
        from training.completion_ode_jobs_v2 import reconcile_parallel_budget
        # The unchanged guard writes an entire ledger atomically. This single
        # current snapshot, not preparation's stale digest/paid subtotal, is the
        # accounting source: another declared owned GPU may legally settle or
        # reserve while the current worker validates its CPU assets. Recompute
        # all real paid/active/pending work; never omit the other reservation.
        return reconcile_parallel_budget(value['parallel_budget_binding'], ledger=ledger, lease=lease,
            max_seconds=max_seconds)
    require(previous.sha256_file(cfg.ledger) == value['ledger_sha256'], 'actual ledger changed since boundedODE preparation')
    charged, global_charged, active = 0., 0., []
    for rid, record in ledger['reservations'].items():
        if record['status'] == 'settled':
            amount = record.get('charged_seconds')
            require(type(amount) in (int, float) and math.isfinite(amount) and amount >= 0, 'invalid actual historicalGPU fee')
            global_charged += amount
            directory = Path(record['context']['job_directory']).resolve()
            if directory == ROOT.resolve() or ROOT.resolve() in directory.parents: charged += amount
        else: active.append((rid, record))
    require(charged == value['settled_gpu_seconds'] and len(active) == 1
        and active[0][0] == lease.get('reservation_id') and active[0][1]['lease'] == lease
        and active[0][1]['reserved_seconds'] == max_seconds+60,
        'realODE fees/current lease differ; no reset or other activeCreator work permitted')
    require(global_charged+max_seconds+60+value['remaining_planned_gpu_seconds']+FOLLOWING_GPU_SECONDS <= 100*3600
        and ledger['storage_bytes_upper'] <= 500_000_000_000, 'actual original100h/500GB insufficient')


def validate_probe_supervision(cache_binding, parent):
    """Public CPU gate for a wrapper to call BEFORE reserving a GPU lease.

    The defensive call inside launch is only pre-CUDA: launch already receives
    an active lease. It cannot substitute for the wrapper's pre-reserve check.
    """
    dataset = ODETrajectoryDataset(cache_binding, parent, purpose='probe')
    for index in range(PROBE_WINDOWS):
        # Each shard contains all five states; the four source windows are read
        # here, not merely accepted from a success boolean on their receipts.
        row = dataset[index]
        require(set(row) == {'trajectory_states', 'clean_context', 'actions', 'prompt_embeds', 'round_index'},
            'independent original-GT probe loader returned inference/leaking fields')
    return dict(probe_windows=PROBE_WINDOWS, independently_verified_GT_sources=True,
        original_GT_used_for_generation=False, numerical_generalization_proven=False)


def artifact_hashes(config, config_path, parent, cache_binding):
    config.validate(); validate_parent_binding(parent)
    validate_cache_binding(cache_binding, teacher_binding=parent)
    require(config.data.manifest_path == cache_binding['index']['path']
        and config.data.manifest_sha256 == cache_binding['index']['sha256'], 'actual ODE index config binding differs')
    return dict(training_config=previous.sha256_file(config_path), entry_source=previous.sha256_file(__file__),
        math_core=previous.sha256_file(core().__file__), tf_entry=previous.sha256_file(tf.__file__),
        causal_tf_source=previous.sha256_file(importlib.import_module('training.causal_tf').__file__),
        parent_binding=copy.deepcopy(parent), cache_binding=copy.deepcopy(cache_binding))


def validate_checkpoint(payload, *, config, hashes, parent, sampling):
    config.validate(); validate_parent_binding(parent)
    expected = dict(format_version=2, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
        config=config.to_dict(), manifest_hashes=hashes, parent_lineage=parent, base_identity=parent['base_identity'],
        method_contract=method_contract(), sampling_contract=sampling,
        frozen_action_signature=previous.frozen_action_signature())
    require(all(payload.get(key) == value for key,value in expected.items()), 'strictODE checkpoint contract differs')
    step = payload.get('step')
    require(type(step) is int and 0 < step <= MAX_STEPS
        and all(payload.get(key) == value for key,value in counters(step, parent).items()), 'nativeODE cursor/history differs')
    previous.validate_frozen_action(payload.get('frozen_action_model'))
    signature, state = payload['model_signature'], payload['trainable_model']
    previous.validate_lora_signature(signature)
    require(set(signature) == set(state), 'full FP32ODE LoRA state required')
    for name, tensor in state.items():
        require(torch.is_tensor(tensor) and tensor.dtype == torch.float32
            and list(tensor.shape) == signature[name]['shape'] and bool(torch.isfinite(tensor).all()), 'invalidODE FP32LoRA')
    require(payload['optimizer_mapping'] == previous.optimizer_mapping(signature, payload['optimizer'], step),
        'full nativeODE optimizer step/moments/mapping required')
    tf.validate_rng(payload)
    return payload


def _checkpoint(model, frozen, optimizer, *, config, hashes, parent, sampling, step, metrics):
    signature, native = previous.model_signature(model), optimizer.state_dict()
    return dict(format_version=2, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, config=config.to_dict(),
        manifest_hashes=hashes, parent_lineage=parent, base_identity=parent['base_identity'],
        method_contract=method_contract(), sampling_contract=sampling, model_signature=signature,
        frozen_action_signature=previous.frozen_action_signature(), frozen_action_model=frozen,
        trainable_model=previous.trainable_state_dict(model), optimizer=native,
        optimizer_mapping=previous.optimizer_mapping(signature, native, step), **counters(step, parent), metrics=metrics,
        torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def native_optimizer_roundtrip(optimizer, signature, step, *, check=lambda:None):
    """Compare an independent CPU snapshot, not aliased state_dict references."""
    from training.studio.paired_turn_update_signal import cpu_clone_tree
    native = optimizer.state_dict()
    previous.optimizer_mapping(signature, native, step)
    snapshot = cpu_clone_tree(native)
    check(); optimizer.load_state_dict(native)
    require(previous.master.equal_tree(snapshot, optimizer.state_dict(), check),
        'ODE native full-state restore differs')
    return True


def packed_wan_callback(model, config, device):
    """The same concrete packed Wan adapter is used for cache and training."""
    from training.completion_ode_wan_v2 import WanPackedAdapter
    return WanPackedAdapter(model, config, device, positive_only=True).callback


def endpoint_loss(model, batch, config, device):
    shared = core()
    states = batch['trajectory_states'].to(device=device, dtype=torch.bfloat16)
    require(states.ndim == 6 and states.shape[0] == 1 and states.shape[1] == 5,
        'microbatch1 and five actual sampled teacher states required')
    trajectory = shared.PackedTrajectory(shared.teacher_grid(), tuple(states[:, index] for index in range(5)))
    level = batch['round_index']
    require(torch.is_tensor(level) and level.numel() == 1, 'one exact cached teacher level per micro required')
    return shared.packed_student_loss(packed_wan_callback(model, config, device), trajectory=trajectory,
        clean_context=batch['clean_context'].to(device=device, dtype=torch.bfloat16),
        actions=batch['actions'].to(device=device, dtype=torch.bfloat16),
        positive_conditions=batch['prompt_embeds'].to(device=device, dtype=torch.bfloat16), round_index=int(level.item()))


def forward_backward(model, dataset, *, step, config, device, profiler, check):
    losses, samples = [], []
    for offset in range(8):
        check(); index = (step-1)*8+offset
        profiler.set_phase('data_wait')
        batch = default_collate([dataset[index]])
        require(set(batch) == {'trajectory_states', 'clean_context', 'actions', 'prompt_embeds', 'round_index'},
            'only explicit separate ODE trajectory/context supervised tensors permitted')
        profiler.set_phase('forward_backward')
        saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if device.type == 'cuda' else nullcontext()
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with saved, amp:
            loss = endpoint_loss(model, batch, config, device)
        require(torch.is_tensor(loss) and loss.ndim == 0 and bool(torch.isfinite(loss)), 'finite scalar trueODE endpoint loss required')
        (loss/8).backward()
        losses.append(float(loss.detach()))
        samples.append(dict(**dataset.sampling_spec(index), loss=losses[-1], target='teacher_endpoint_not_GT'))
        del loss, batch
    return sum(losses)/8, samples


def launch(config, config_path, *, parent, cache_binding, start_step, stop_after_step, resume,
        measured_slowest_step_seconds, budget_contract, shared_lease, shared_max_seconds, shared_request_sha256,
        profile_binding=None):
    """Prepared execution only; requires missing actual parent/cache/resource pins."""
    from training.demo.backend import validate_active_authority
    config.validate(); validate_parent_binding(parent)
    require(parent_binding(parent['parent_step']) == parent, 'actual completed TF endpoint required')
    validate_budget_contract(budget_contract, shared_max_seconds,
        current_start_step=start_step, current_stop_step=stop_after_step)
    reconcile_real_budget(budget_contract, shared_lease, max_seconds=shared_max_seconds)
    output = Path(config.training.output_dir)
    require(type(start_step) is int and type(stop_after_step) is int, 'explicitODE start/stop required')
    if start_step == 0:
        require(stop_after_step == 2 and resume is None and shared_max_seconds == 1800
            and profile_binding is None and (not output.exists() or not any(output.iterdir())),
            'firstODE real2step profile requires new output/optimizer, not fake prior success')
    else:
        segment = next_segment(start_step, measured_slowest_step_seconds)
        require(stop_after_step == segment['stop_after_step'] and shared_max_seconds == segment['max_seconds']
            and Path(resume or '') == output/f'checkpoints/step-{start_step:07d}.pt', 'strict boundedODE predecessor required')
        profile = actual_profile(profile_binding, kind='training')
        require(profile.get('native_state_restore_matched') is True and profile.get('lora_updated') is True
            and profile.get('frozen_action_has_gradients') is False
            and 0 < profile.get('peak_reserved_bytes', 0) < 23*1024**3,
            'real2step ODE resource/update/native-state profile required before continuing')
    dataset = ODETrajectoryDataset(cache_binding, parent)
    validate_probe_supervision(cache_binding, parent)
    sampling = sampling_contract(dataset, parent)
    hashes = artifact_hashes(config, config_path, parent, cache_binding)
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(state, config=config, hashes=hashes, parent=parent, sampling=sampling)
        require(state['step'] == start_step, 'actualODE predecessor differs')
    else:
        state = load_parent(parent)
    gpu = previous.teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
        request_sha256=shared_request_sha256)
    allowed=budget_contract.get('parallel_budget_binding',{}).get('allowed_gpu_indices',[1,2])
    require(shared_lease.get('gpu_index') in allowed and torch.cuda.is_available(), 'actual owned Creator lease only')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    previous.seed_everything(42)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, config=config.to_dict(), manifest_hashes=hashes, parent_lineage=parent,
        sampling_contract=sampling, method_contract=method_contract(), resume=resume, resume_step=start_step,
        stop_after_step=stop_after_step, budget_contract=budget_contract, gpu_state=gpu.as_dict(),
        git_revision=previous.git_revision(Path(__file__).parent), started_unix=time.time(),
        shared_guard=dict(lease=shared_lease, max_seconds=shared_max_seconds, request_sha256=shared_request_sha256))
    if not resume:
        with (output/'run_metadata.json').open('x', encoding='utf-8') as stream: json.dump(metadata, stream, indent=2)
    previous.append_jsonl(output/'executions.jsonl', metadata)
    profiler = previous.OptimizerStepProfiler(device, output_dir=output/'profile_output', gpu_uuid=shared_lease['gpu_uuid'])
    began, last_check = time.monotonic(), 0.

    def check():
        nonlocal last_check
        now = time.monotonic()
        require(now-began < shared_max_seconds-30, 'finiteODE deadline reached')
        if now-last_check >= 2: validate_active_authority(shared_lease); last_check = now

    try:
        frozen = state['frozen_action_model']
        with profiler.phase('model_load'):
            model, summary = previous.build_model(config, device, frozen_action_model=frozen,
                trainable_model=state['trainable_model'])
            signature = previous.model_signature(model)
            require(signature == state['model_signature'], 'ODE LoRA identity differs from formalTF parent')
            optimizer = previous.make_optimizer(model, state['optimizer'] if resume else None, check=check)
        if resume: previous.teacher._restore_rng_state(state, require_complete=True)
        else: previous.seed_everything(42)
        del state
        frozen_versions = {name:p._version for name,p in model.named_parameters() if not p.requires_grad}
        model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
        for step in range(start_step+1, stop_after_step+1):
            check(); before = previous.sampled_lora(model); profiler.begin_step(step)
            loss, samples = forward_backward(model, dataset, step=step, config=config, device=device,
                profiler=profiler, check=check)
            gradients = previous.gradient_evidence(model, step)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            require(bool(torch.isfinite(norm)), 'nonfiniteODE gradient norm')
            profiler.set_phase('optimizer_update'); optimizer.step()
            after = previous.sampled_lora(model)
            require(all(bool(torch.isfinite(value).all()) for value in after.values()), 'nonfiniteODE LoRA update')
            changed = sum(int(torch.count_nonzero(after[name]-before[name])) for name in before)
            require(changed > 0 and all(p.grad is None and p._version == frozen_versions[name]
                for name,p in model.named_parameters() if name in frozen_versions), 'ODE did not updateLoRA or modified frozen action')
            optimizer.zero_grad(set_to_none=True)
            native_roundtrip_verified = False
            if step == stop_after_step:
                native_roundtrip_verified = native_optimizer_roundtrip(optimizer, signature, step, check=check)
            timing = profiler.end_step(step); memory = timing['memory']
            require(memory['cumulative_peak_reserved_bytes'] < 23*1024**3, 'ODE reservedmemory exceeds23GiB')
            latest = dict(loss=loss, loss_scope='future_only_teacher_ode_endpoint_MSE', grad_norm=float(norm),
                optimizer_step_seconds=timing['seconds'], task_gradient_evidence=gradients,
                lora_changed_sampled_elements=changed, adapter_unchanged=True, gate_unchanged=True,
                native_state_restore_matched=native_roundtrip_verified,
                micro_samples=samples, sample_namespace=NAMESPACE, step_profile=timing,
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'], peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'])
            previous.append_jsonl(output/'metrics.jsonl', dict(**counters(step, parent), **latest))
            print(json.dumps(dict(**counters(step, parent), **latest)), flush=True)
        profiler.begin_checkpoint(stop_after_step)
        target = previous.master.b.pair._save_checkpoint(output, _checkpoint(model, frozen, optimizer,
            config=config, hashes=hashes, parent=parent, sampling=sampling, step=stop_after_step, metrics=latest))
        profiler.end_checkpoint(stop_after_step, checkpoint_path=str(target))
        result = dict(status='completed', stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
            **counters(stop_after_step, parent), parent_lineage=parent, base_identity=parent['base_identity'],
            sampling_contract=sampling, method_contract=method_contract(), model_signature=signature,
            frozen_action_signature=previous.frozen_action_signature(), resume_step=start_step, checkpoint_path=str(target),
            configured_max_steps=MAX_STEPS, trainable_parameters=summary.trainable_parameters, **latest,
            qualification=dict(lora_updated=True, gradients_finite=True, adapter_unchanged=True, gate_unchanged=True,
                native_state_restore_matched=True, frozen_action_has_gradients=False,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']), quality_improvement_verified=False,
            training_length_adequacy_proven=False, realtime_verified=False, automatic_next_stage=False)
        previous.append_jsonl(output/'execution_segments.jsonl', result)
        (output/'execution_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    except BaseException as error:
        profiler.close(error=error)
        previous.append_jsonl(output/'failures.jsonl', dict(status='failed', stage=STAGE, failure_type=type(error).__name__,
            error=str(error), resume_step=start_step, timestamp_unix=time.time()))
        raise
    finally: profiler.close()
