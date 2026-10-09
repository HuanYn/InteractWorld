"""A-visual-gated B: same parent/optimizer/noise, new two-offset training pool.

Independent bounded stage. Reuses the unchanged tinyfit optimizer/flow loop,
not its old data identity. Never initialize from A or alter running entries.
"""
from __future__ import annotations

import copy
import json
from operator import index as integer_index
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_action_teacher as teacher
import train_controlled_lora_pair as pair
import train_controlled_tinyfit as reference
from training.studio import paired_turn_cache as cache
from training.studio import paired_turn_eval as a_eval
from training.config import load_config as _load_config
from training.data.action_dataset import _seed64, cache_index_path
from training.data.action_resampled import ResampledActionDataset
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file

STAGE = STAGE_NAME = 'controlled_action_paired_turn_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_paired_turn_checkpoint_v1'
FACTORY = 'train_paired_turn_teacher:build_paired_turn_dataloader'
NAMESPACE = 'controlled_tinyfit_two_windows_seed42_v1'
MAX_STEPS, MICRO_BATCHES, HISTORY = 20, 160, 320
GATE_LR, RATIO_CAP = pair.GATE_LR, pair.RATIO_CAP
ROOT = cache.ROOT
DATA = ROOT / 'data'
REVIEW = ROOT / 'a/review-result.json'
PARENT_CHECKPOINT = a_eval.PARENT_CHECKPOINT
REFERENCE_ROOT = reference.ROOT / 'teacher-tinyfit-v1'
FIXED = (('J', cache.FIXED_EPISODE, 0, 0), ('L', cache.FIXED_EPISODE, 1, 210))
POOL_SCHEMA = 'controlled_paired_turn_pool_v1'
_require = pair._require
base_identity = pair.base_identity
validate_base_identity = pair.validate_base_identity
install_parent_weights = pair.install_parent_weights


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_config(config):
    """Compatibility view solely for unchanged model construction validators."""
    result = copy.deepcopy(config)
    result.data.data_factory = pair.FACTORY
    result.data.manifest_sha256 = pair.MANIFEST_SHA256
    return result


def validate_probe_config(config):
    _require(config.data.data_factory == FACTORY, 'explicit tiny two-window loader required')
    _require(isinstance(config.data.manifest_sha256, str) and len(config.data.manifest_sha256) == 64,
             'new two-window manifest digest required')
    pair.validate_probe_config(model_config(config), arm='frozen')
    _require(config.training.seed == 42 and config.training.micro_batch_size == 1
        and config.training.gradient_accumulation_steps == 8 and config.data.num_workers == 0,
        'fixed seed42 micro1 accumulate8 serial sampling required')


def load_config(path):
    config = _load_config(path)
    validate_probe_config(config)
    return config


def validate_config(config, *, data_root, output_dir, model_dir):
    validate_probe_config(config)
    # The old validator is reused for unchanged model/optimizer/path rules only.
    # Actual new cache/manifest identity is checked separately, never old SHA.
    pair.validate_config(model_config(config), data_root=data_root, output_dir=output_dir,
                         model_dir=model_dir, arm='frozen')
    _require(pair.parent._absolute(data_root) == pair.parent._absolute(DATA),
             'B assets must use the stage-specific exact two-window cache')


def artifact_hashes(config, config_path):
    validate_probe_config(config)
    values = teacher._manifest_hashes(config, config_path)
    binding = cache.validate_completed_cache()
    _require(str(cache_index_path(config.data.manifest_path)) == binding['cache_index'],
             'B feature index differs from completed paired cache')
    values.update(paired_turn_cache_result=sha256_file(ROOT / cache.JOB_ID / 'result.json'),
                  paired_turn_parent_request=sha256_file(reference.SOURCE_EVAL),
                  paired_turn_a_review=sha256_file(REVIEW),
                  paired_turn_a_metrics=sha256_file(REFERENCE_ROOT / 'ctf-profile2-v1/training/metrics.jsonl'))
    return values


def visual_gate():
    """Fail closed on actual A review plus the exact four execution records.

    The main assistant writes the review only after viewing the videos. This
    function validates provenance, not visual semantics; it never fabricates
    a review or changes its decision based on metrics.
    """
    review = _read(REVIEW)
    required = dict(schema='paired-turn-a-visual-review-v1', arm='A_old_pool',
        execution_contract_valid=True, all_four_completed_and_settled=True,
        all_four_decoded_97=True, reviewed_by='main_assistant',
        review_scope='fixed_0_to_6_seconds_and_prespecified_turn_timestamps',
        visually_reviewed=True, b_training_allowed=True)
    _require(all((review.get(k) is v) if type(v) is bool else (review.get(k) == v)
                 for k, v in required.items()),
             'actual technically valid A visual review must explicitly allow B')
    score, appearance = review.get('paired_pass_count'), review.get('appearance_pass')
    _require(type(score) is int and 0 <= score <= 2 and type(appearance) is bool,
             'A paired score and appearance decision missing')
    _require(not (score == 2 and appearance), 'A already passed: B must not run')
    cases = review.get('cases')
    _require(isinstance(cases, list) and len(cases) == 4
        and {c.get('job_id') for c in cases} == set(a_eval.CASES.values()),
        'A review must bind exactly all four fixed jobs')
    observed = []
    for (target, condition), job_id in a_eval.CASES.items():
        row = next(c for c in cases if c['job_id'] == job_id)
        _require(row.get('target') == target and row.get('condition') == condition
                 and row.get('decoded_frames') == 97 and row.get('visually_reviewed') is True,
                 'per-case A decoding/review evidence missing')
        paths = {name: ROOT / job_id / (name + '.json')
                 for name in ('request', 'result', 'settlement')}
        documents = {}
        for name, path in paths.items():
            _require(row.get(name + '_path') == str(path)
                     and row.get(name + '_sha256') == sha256_file(path),
                     'A review record binding changed: ' + job_id + '/' + name)
            documents[name] = _read(path)
        request, result, settlement = (documents[k] for k in ('request', 'result', 'settlement'))
        a_eval.validate_request(request)
        _require(request['job_id'] == job_id and request['target'] == target
                 and request['condition'] == condition and result.get('request') == request
                 and result.get('status') == 'completed' and result.get('mode') == a_eval.MODE
                 and result.get('frames_written') == 97
                 and result.get('actual_model_forward_calls') == 80
                 and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
                 'technical A failure cannot authorize B training')
        observed.append({key: row[key] for key in ('job_id', 'target', 'condition',
            'request_path', 'request_sha256', 'result_path', 'result_sha256',
            'settlement_path', 'settlement_sha256', 'decoded_frames', 'visually_reviewed')})
    return dict(review_path=str(REVIEW), review_sha256=sha256_file(REVIEW),
                paired_pass_count=score, appearance_pass=appearance, b_training_allowed=True,
                cases=observed, verification='actual_main_review_and_small_execution_records;not_a_visual_judge')


def reference_a_contract(config):
    """Check equal settings, actual160 A samples and P lineage, without A weights."""
    source, result = a_eval.source_evaluation()
    a_config = reference.load_config(source['training_config'])
    before, after = a_config.to_dict(), config.to_dict()
    _require(before['model'] == after['model'] and before['optimizer'] == after['optimizer'],
             'B must preserve A model and optimizer hyperparameters')
    for section, changed in (('data', {'manifest_path', 'manifest_sha256', 'data_factory'}),
                             ('training', {'output_dir'})):
        _require({k: v for k, v in before[section].items() if k not in changed}
                 == {k: v for k, v in after[section].items() if k not in changed},
                 'B may only replace pool paths/factory and output: ' + section)
    metadata = result['lineage']['checkpoint_lineage']
    _require(metadata['stage'] == reference.STAGE and metadata['step'] == 20,
             'A must be the completed old-pool tinyfit20 endpoint')
    _validate_lineage(metadata['parent_lineage'])
    reference._validate_pool(metadata['sampling_contract']['pool'])
    expected_policy = reference.sampling_contract(a_config.training, metadata['sampling_contract']['pool'])
    _require(metadata['sampling_contract'] == expected_policy and NAMESPACE == reference.NAMESPACE,
             'A noise/virtual-index mechanism changed')
    metrics_path = Path(source['training_config']).parent / 'training/metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    _require([m.get('step') for m in metrics] == list(range(1, 21)),
             'A must have exactly20 actual optimizer-step records, no duplicate/omitted steps')
    samples = []
    for step, row in enumerate(metrics, 1):
        _require(row.get('micro_batches_consumed') == step * 8
                 and row.get('sample_namespace') == NAMESPACE and len(row.get('micro_samples', [])) == 8,
                 'A actual microbatch exposure record incomplete')
        for sample in row['micro_samples']:
            index = len(samples)
            target, episode, window, start = reference.FIXED[index % 2]
            _require(sample.get('new_stage_sample_index') == sample.get('absolute_sample_index') == index
                and sample.get('index_namespace') == NAMESPACE
                and (sample.get('target'), sample.get('episode_id'), sample.get('window_index'), sample.get('source_start'))
                    == (target, episode, window, start)
                and sample.get('diffusion_seed') == _seed64(_seed64(42, NAMESPACE), 'diffusion', index)
                and type(sample.get('actual_timestep')) in (int, float)
                and 0 < sample['actual_timestep'] <= 1000,
                'A recorded sample/seed/timestep differs from the fixed160 exposure protocol')
            samples.append(dict(new_stage_sample_index=index, diffusion_seed=sample['diffusion_seed'],
                                actual_timestep=sample['actual_timestep']))
    _require(len(samples) == 160, 'exactly160 A exposures required')
    record = dict(reference_arm='A_old_pool', training_config=source['training_config'],
        training_config_sha256=source['training_config_sha256'], metrics_path=str(metrics_path),
        metrics_sha256=sha256_file(metrics_path), parent_lineage=metadata['parent_lineage'],
        source_evaluation=str(a_eval.SOURCE_EVAL), source_evaluation_sha256=sha256_file(a_eval.SOURCE_EVAL),
        source_result_sha256=sha256_file(a_eval.SOURCE_RESULT), actual_micro_samples=samples,
        optimizer_steps=20, micro_batches=160, each_pool_window_exposures=80,
        namespace=NAMESPACE, uses_A_weights=False, noise_tensor_equality_not_directly_measured=True,
        comparison='same_draw_algorithm_shape_seed;verify_actual_timestep_each_B_micro')
    return a_config, record


def pool_contract():
    binding = cache.validate_completed_cache()
    from training.data.action_dataset import _read_jsonl
    rows = _read_jsonl(Path(binding['cache_index']))
    _require(len(rows) == 1 and rows[0]['episode_id'] == cache.FIXED_EPISODE
             and rows[0]['num_windows'] == 2 and len(rows[0]['shards']) == 1,
             'B must use only the exact completed one-episode two-window index')
    windows = []
    for target, episode, window, start in FIXED:
        row = binding['windows'][target]
        _require((row['episode_id'], row['window_index'], row['source_start']) == (episode, window, start)
                 and row['annotation_cache_actions_equal'] is True, 'new paired-turn window identity changed')
        windows.append(dict(target=target, episode_id=episode, window_index=window, source_start=start,
            shard_path=row['shard_path'], shard_stat=row['shard_stat'],
            recorded_shard_sha256=rows[0]['shards'][0]['sha256'],
            original_actions=row['future_actions_96x8'],
            first_latent_sha256_float32=row['first_latent_sha256_float32'],
            prompt_sha256_float32=binding['prompt_sha256_float32'],
            original_action_sha256_float32=row['original_action_sha256_float32']))
    result = dict(schema=POOL_SCHEMA, cache_index=binding['cache_index'],
        cache_index_sha256=binding['cache_index_sha256'], cache_receipt=binding['cache_receipt'],
        cache_receipt_sha256=binding['cache_receipt_sha256'], cache_binding_sha256=a_eval.binding_digest(binding),
        windows=windows, distinct_windows=2, each_new_exposures=80, total_new_exposures=160,
        training_condition='original_recorded_actions_not_counterfactual_labels',
        same_episode=True, true_same_state_counterfactual=False, supports_generalization=False,
        cache_verification='completed_new_offset_cache_receipt_index_and_stat_no_old_latent_reuse')
    _validate_pool(result)
    return result


def _validate_pool(pool):
    _require(isinstance(pool, dict) and pool.get('schema') == POOL_SCHEMA
        and pool.get('distinct_windows') == 2 and pool.get('each_new_exposures') == 80
        and pool.get('total_new_exposures') == 160, 'fixed paired-turn two-window pool required')
    rows = pool.get('windows', [])
    _require(len(rows) == 2 and tuple((r['target'], r['episode_id'], r['window_index'], r['source_start'])
        for r in rows) == FIXED, 'B pool identities/order changed')
    # Exact preselected 96x8 labels, not merely binary shapes or key counts.
    runs = {'J': [(34, 'J'), (3, ''), (48, 'W'), (11, 'WJ')],
            'L': [(32, 'W'), (15, 'WL'), (2, 'W'), (2, ''), (35, 'L'), (2, ''), (8, 'W')]}
    keys = 'WASDIJKL'
    for row in rows:
        expected = [[float(key in pressed) for key in keys]
                    for count, pressed in runs[row['target']] for _ in range(count)]
        _require(row.get('original_actions') == expected,
                 'B training labels must remain exact original source0/210 actions')


def sampling_contract(training, pool):
    _validate_pool(pool)
    _require(training.seed == 42 and training.max_steps == 20
        and training.micro_batch_size == 1 and training.gradient_accumulation_steps == 8,
        'tinyfit sampler requires fixed20x8')
    return dict(namespace=NAMESPACE, index_policy='new_stage_index_alternates_J_L_exact_offsets_same_A_noise_namespace',
        training_seed=42, data_seed=_seed64(42, NAMESPACE), virtual_samples=160,
        first_stage_sample_index=0, last_stage_sample_index=159, exposures_per_window=80,
        each_optimizer_step=dict(J=4, L=4), historical_micro_batches_before_stage=HISTORY,
        original_absolute_dataset_cursor_continued=False,
        diffusion_seed_policy='seed64(seed64(42,namespace),diffusion,new_stage_index)',
        noise_and_timestep='fresh_deterministic_random_draw_per_exposure;not_fixed_noise',
        pool=pool)


class PairedTurnDataset(ResampledActionDataset):
    def __init__(self, index_path, *, pool, **kwargs):
        _validate_pool(pool)
        super().__init__(index_path, num_samples=MICRO_BATCHES, verify_hashes=False, **kwargs)
        self.pool = pool
        self.data_seed = _seed64(42, NAMESPACE)
        self.fixed_slots = []
        self._verified_windows = set()
        for row in pool['windows']:
            choices = [i for i, episode in enumerate(self.episodes) if episode['episode_id'] == row['episode_id']]
            _require(len(choices) == 1, 'selected window absent from existing train index')
            slot = choices[0]
            episode = self.episodes[slot]
            _require(episode['num_windows'] == 2 and len(episode['shards']) == 1,
                     'new exact two-window episode expected')
            shard = episode['shards'][0]
            _require((self.index_path.parent / shard['path']).resolve() == Path(row['shard_path']).resolve()
                and shard['sha256'] == row['recorded_shard_sha256'], 'trusted selected shard index differs')
            self.fixed_slots.append(slot)

    def sampling_spec(self, index):
        _require(not isinstance(index, bool), 'sample index cannot be bool')
        index = integer_index(index)
        if not 0 <= index < MICRO_BATCHES:
            raise IndexError(index)
        row = self.pool['windows'][index % 2]
        return dict(absolute_sample_index=index, new_stage_sample_index=index,
            index_namespace=NAMESPACE, episode_slot=self.fixed_slots[index % 2],
            episode_id=row['episode_id'], window_index=row['window_index'], source_start=row['source_start'],
            target=row['target'], diffusion_seed=_seed64(self.data_seed, 'diffusion', index))

    def _cached_window(self, episode, window_index):
        row = next((r for r in self.pool['windows'] if r['episode_id'] == episode['episode_id']
                    and r['window_index'] == window_index), None)
        _require(row is not None, 'attempt to train on a window outside the fixed tiny pool')
        value = super()._cached_window(episode, window_index)
        identity = (row['episode_id'], window_index)
        if identity not in self._verified_windows:
            from training.studio.causal_native_eval import tensor_sha
            _require(torch.equal(value['actions'], torch.tensor(row['original_actions'], dtype=torch.float32))
                and tensor_sha(value['clean_latents'][:1][None]) == row['first_latent_sha256_float32']
                and tensor_sha(value['prompt_embeds']) == row['prompt_sha256_float32'],
                'selected conditioning/actions differ from matched generation')
            self._verified_windows.add(identity)
        return value


def build_paired_turn_dataloader(*, config, training, pool=None):
    pool = pool_contract() if pool is None else pool
    sampling_contract(training, pool)
    dataset = PairedTurnDataset(cache_index_path(config.manifest_path), pool=pool,
        manifest_path=config.manifest_path, split='train', seed=42, timestep_shift=5.,
        prompt_cache_path=None, num_frames=97)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=True)


def _validate_lineage(lineage):
    _require(isinstance(lineage, dict) and lineage.get('mode') == 'weights_only_new_stage'
        and lineage.get('parent_stage') == pair.STAGE and lineage.get('parent_arm') == 'frozen'
        and lineage.get('parent_step') == 20 and lineage.get('parent_micro_batches_consumed') == 320
        and lineage.get('parent_new_stage_micro_batches_consumed') == 160
        and lineage.get('historical_micro_batches_before_stage') == HISTORY
        and lineage.get('optimizer_restored') is False and lineage.get('rng_restored') is False
        and lineage.get('r005_inherited') is False and lineage.get('teacher64_inherited') is False,
        'completed frozen-pair weights-only parent, not a strict resume or fresh base, required')
    for key in ('parent_checkpoint', 'parent_config', 'parent_job'):
        pair.parent._absolute(lineage[key])
    _require(Path(lineage['parent_checkpoint']).name == 'step-0000020.pt'
        and isinstance(lineage.get('base_identity'), dict), 'fixed parent endpoint/base identity required')
    _require(lineage['parent_checkpoint'] == str(PARENT_CHECKPOINT),
             'B must initialize from common P frozen20, never A tinyfit20')


def load_parent(config):
    # Validate the original A-side loader/config independently, then transplant
    # only P Adapter/gate. A weights are never a source.
    a_config, reference_record = reference_a_contract(config)
    state, lineage = reference.load_parent(a_config)
    _validate_lineage(lineage)
    _require(lineage == reference_record['parent_lineage'],
             'P lineage must be exactly the same as the completed A training parent')
    return state, lineage


def precision_contract():
    return pair.precision_contract('frozen')


def method_contract():
    values = pair.parent.method_contract()
    values.update(method=STAGE, gate_initialization='inherited_frozen_pair20_weights_only',
        hyperparameters_source='project_fixed_equal_pool_replacement_paired_turn_v1', precision=precision_contract(),
        new_optimizer=True, parent_strict_resume=False, fresh_from_official_base=False,
        sampling='two_preselected_native97_windows_alternating_80_new_exposures_each',
        generalization_experiment=False, data_coverage_causal_proof=False,
        same_episode_different_first_frames=True, true_same_state_counterfactual=False,
        requires_valid_negative_A_visual_review=True)
    return values


def initialization_contract(config, lineage):
    _validate_lineage(lineage)
    return dict(mode='frozen_pair20_adapter_gate_weights_only_new_optimizer',
        base_model_path=config.model.base_model_path, source_checkpoint=lineage['parent_checkpoint'],
        source_checkpoint_sha256=lineage['parent_checkpoint_sha256'],
        adapter_initialization='inherited_frozen_pair20', gate_initialization='inherited_frozen_pair20_FP32',
        lora_initialization='same_FP32_seed42_A_zero_B_frozen;zero_functional_residual',
        lora_trainable=False, optimizer_restored=False, rng_restored=False, start_step=0,
        new_stage_micro_batches_consumed=0, historical_micro_batches_before_stage=HISTORY,
        training_seed=42, parent_strict_resume=False, fresh_from_base=False)


def _build_model(config, device):
    validate_probe_config(config)
    return pair._build_model(model_config(config), device, arm='frozen')


def _checkpoint(model, optimizer, *, config, hashes, lineage, pool, step, metrics, a_visual_gate, a_reference):
    return dict(format_version=1, checkpoint_schema=CHECKPOINT_SCHEMA, stage=STAGE, arm='frozen',
        config=config.to_dict(), manifest_hashes=hashes, base_identity=lineage['base_identity'], parent_lineage=lineage,
        a_visual_gate=a_visual_gate, a_reference=a_reference,
        initialization=initialization_contract(config, lineage), method_contract=method_contract(),
        precision_contract=precision_contract(), sampling_contract=sampling_contract(config.training, pool),
        model_signature=pair._model_signature(model, 'frozen'), trainable_model=trainable_state_dict(model),
        optimizer=optimizer.state_dict(), step=step, micro_batches_consumed=step * 8,
        new_stage_micro_batches_consumed=step * 8, historical_micro_batches_before_stage=HISTORY,
        lifetime_micro_batches_consumed=HISTORY + step * 8, parent_optimizer_steps=20,
        ancestor_optimizer_steps=20, metrics=metrics, torch_rng_state=torch.get_rng_state(),
        cuda_rng_state_all=torch.cuda.get_rng_state_all(), python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes):
    validate_probe_config(config)
    _require(isinstance(payload, dict) and payload.get('format_version') == 1
        and payload.get('checkpoint_schema') == CHECKPOINT_SCHEMA and payload.get('stage') == STAGE
        and payload.get('arm') == 'frozen', 'new paired-turn B stage checkpoint required')
    lineage = payload.get('parent_lineage')
    _validate_lineage(lineage)
    _require(isinstance(hashes.get('paired_turn_a_review'), str)
             and len(hashes['paired_turn_a_review']) == 64
             and isinstance(hashes.get('paired_turn_a_metrics'), str)
             and len(hashes['paired_turn_a_metrics']) == 64
             and payload.get('a_visual_gate', {}).get('review_sha256') == hashes['paired_turn_a_review']
             and payload['a_visual_gate'].get('b_training_allowed') is True
             and payload.get('a_reference', {}).get('metrics_sha256') == hashes.get('paired_turn_a_metrics')
             and payload['a_reference'].get('parent_lineage') == lineage,
             'B checkpoint must preserve actual A review and equal-training provenance')
    pool = payload.get('sampling_contract', {}).get('pool')
    _validate_pool(pool)
    expected = dict(config=config.to_dict(), manifest_hashes=hashes,
        base_identity=lineage['base_identity'], initialization=initialization_contract(config, lineage),
        method_contract=method_contract(), precision_contract=precision_contract(),
        sampling_contract=sampling_contract(config.training, pool))
    for key, value in expected.items():
        _require(payload.get(key) == value, 'paired-turn strict checkpoint changed: ' + key)
    step = payload.get('step')
    _require(type(step) is int and step in (2, 20) and payload.get('micro_batches_consumed') == step * 8
        and payload.get('new_stage_micro_batches_consumed') == step * 8
        and payload.get('historical_micro_batches_before_stage') == HISTORY
        and payload.get('lifetime_micro_batches_consumed') == HISTORY + step * 8,
        'new-stage cursor must not be confused with original absolute sampling')
    state, signature = payload.get('trainable_model'), payload.get('model_signature')
    _require(isinstance(state, dict) and bool(state) and isinstance(signature, dict)
        and set(state) == set(signature), 'tinyfit state/signature differ')
    groups = set()
    for name, value in state.items():
        group = pair.parent._group(name)
        groups.add(group)
        dtype = torch.float32 if group == 'gate' else torch.bfloat16
        _require(torch.is_tensor(value) and value.dtype == dtype and bool(torch.isfinite(value).all())
            and signature[name] == dict(shape=list(value.shape), dtype=str(dtype), group=group),
            'tinyfit state tensor precision/shape differs')
        if group == 'gate':
            _require(value.shape == (3072,), 'FP32 gate must retain3072 channels')
    _require(groups == {'adapter', 'gate'}, 'only Adapter/gate state permitted; no LoRA inherited')
    return payload


def validate_resume(path, *, config, hashes):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(payload, config=config, hashes=hashes)
    optimizer = payload.get('optimizer', {})
    groups = optimizer.get('param_groups', [])
    _require([g.get('name') for g in groups] == ['action_adapter', 'controlled_action_gate'],
             'same two optimizer groups required')
    ids = []
    for group, name, lr, decay in zip(groups, ('adapter', 'gate'), (1e-5, GATE_LR), (.01, 0.)):
        count = sum(v['group'] == name for v in payload['model_signature'].values())
        _require(group.get('lr') == lr and group.get('weight_decay') == decay
            and tuple(group.get('betas', ())) == (.9, .95) and len(group.get('params', [])) == count,
            'strict same-stage optimizer settings changed')
        ids.extend(group['params'])
    _require(isinstance(optimizer.get('state'), dict) and len(ids) == len(set(ids))
        and set(ids) == set(optimizer['state']) and all(bool(optimizer['state'][i]) for i in ids),
        'complete optimizer state required')
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    random.Random().setstate(payload['python_rng_state'])
    np.random.RandomState().set_state(payload['numpy_rng_state'])
    rng = payload.get('cuda_rng_state_all')
    _require(isinstance(rng, list) and len(rng) == 1 and torch.is_tensor(rng[0])
        and rng[0].dtype == torch.uint8 and rng[0].ndim == 1 and rng[0].numel(), 'single CUDA RNG required')
    return payload


def launch(config, config_path, *, data_root, output_dir, model_dir, parent_lineage, pool,
           stop_after_step, a_visual_gate, a_reference, resume=None, shared_lease, shared_max_seconds, shared_request_sha256):
    validate_config(config, data_root=data_root, output_dir=output_dir, model_dir=model_dir)
    _validate_lineage(parent_lineage)
    _validate_pool(pool)
    _require(type(stop_after_step) is int and stop_after_step in (2, 20), 'only2/20 endpoints permitted')
    _require(type(shared_max_seconds) is int and 60 <= shared_max_seconds <= 3600, 'bounded job required')
    output = Path(output_dir).resolve()
    if resume is None:
        _require(stop_after_step == 2 and shared_max_seconds <= 900, 'first measure exactly two optimizer steps')
        _require(not output.exists() or not any(output.iterdir()), 'new stage output must be empty')
    else:
        _require(Path(resume).resolve().parent == output / 'checkpoints'
            and Path(resume).name == 'step-0000002.pt' and stop_after_step == 20, 'strict same-stage2->20 only')
    _require(visual_gate() == a_visual_gate, 'A visual gate changed after preparation')
    _require(reference_a_contract(config)[1] == a_reference, 'A training equivalence changed')
    _require(pool_contract() == pool, 'fixed paired pool changed after preparation')
    hashes = artifact_hashes(config, config_path)
    _require(base_identity(config) == parent_lineage['base_identity'], 'same frozen official base required')
    restored = validate_resume(resume, config=config, hashes=hashes) if resume else None
    state = None
    if restored:
        _require(restored['step'] == 2 and restored['parent_lineage'] == parent_lineage
            and restored['a_visual_gate'] == a_visual_gate and restored['a_reference'] == a_reference
            and restored['sampling_contract'] == sampling_contract(config.training, pool), 'strict lineage/pool differs')
    else:
        state, observed = load_parent(config)
        _require(observed == parent_lineage, 'weights-only parent changed after CPU preparation')
    start = restored['step'] if restored else 0
    gpu = teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
        request_sha256=shared_request_sha256)
    _require(torch.cuda.is_available(), 'leased tinyfit training requires CUDA')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, config=config.to_dict(), manifest_hashes=hashes,
        base_identity=parent_lineage['base_identity'], parent_lineage=parent_lineage,
        initialization=initialization_contract(config, parent_lineage), method_contract=method_contract(),
        sampling_contract=sampling_contract(config.training, pool), a_visual_gate=a_visual_gate,
        a_reference=a_reference, command=sys.argv,
        git_revision=git_revision(Path(__file__).parent), entry_sha256=sha256_file(__file__),
        gpu_state=gpu.as_dict(), shared_guard=dict(lease=shared_lease, max_seconds=shared_max_seconds,
            request_sha256=shared_request_sha256, watchdog_owner='external_guard_worker'),
        resume=resume, resume_step=start, stop_after_step=stop_after_step, started_unix=time.time())
    if not resume:
        with (output / 'run_metadata.json').open('x', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
    append_jsonl(output / 'executions.jsonl', metadata)
    profiler = OptimizerStepProfiler(device, output_dir=output / 'profile_output', gpu_uuid=shared_lease['gpu_uuid'])
    began = time.monotonic()
    try:
        with profiler.phase('model_load'):
            model, summary = _build_model(config, device)
        if state is not None:
            install_parent_weights(model, state)
            del state
        optimizer = pair._build_optimizer(model, model_config(config), 'frozen')
        if restored:
            _require(restored['model_signature'] == pair._model_signature(model, 'frozen'), 'model resume signature differs')
            load_trainable_state_dict(model, restored['trainable_model'])
            optimizer.load_state_dict(restored['optimizer'])
        loader = build_paired_turn_dataloader(config=config.data, training=config.training, pool=pool)
        _require(len(loader) == MICRO_BATCHES, 'tinyfit must expose exactly160 new samples')
        _, iterator = teacher._iterator_at_micro_batch(loader, start * 8)
        if restored:
            teacher._restore_rng_state(restored, require_complete=True)
            del restored
        model.zero_grad(set_to_none=True)
        latest = {}
        for step in range(start + 1, stop_after_step + 1):
            before = pair._sample_parameters(model)
            profiler.begin_step(step)
            losses, samples = [], []
            for offset in range(8):
                sample_index = (step - 1) * 8 + offset
                spec = loader.dataset.sampling_spec(sample_index)
                profiler.set_phase('data_wait')
                batch = next(iterator)
                reference_sample = a_reference['actual_micro_samples'][sample_index]
                _require(spec['diffusion_seed'] == reference_sample['diffusion_seed']
                    and float(batch['timesteps'][0, 1]) == reference_sample['actual_timestep'],
                    'B actual noise seed/timestep differs from A at the same micro index')
                samples.append(dict(**spec, actual_timestep=float(batch['timesteps'][0, 1])))
                profiler.set_phase('forward')
                with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = teacher._forward_loss(model, batch, config, device)
                _require(bool(torch.isfinite(loss)), 'nonfinite tinyfit loss')
                profiler.set_phase('backward')
                (loss / 8).backward()
                losses.append(float(loss.detach()))
                del loss
            precision = pair._gradient_precision(model, 'frozen')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            _require(bool(torch.isfinite(norm)), 'nonfinite tinyfit gradient')
            precision['gradients_finite'] = True
            profiler.set_phase('optimizer_update')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            timing = profiler.end_step(step)
            updates = pair._update_evidence(model, before, step=step, arm='frozen')
            del before
            memory = timing['memory']
            latest = dict(loss=sum(losses) / 8, grad_norm=float(norm), precision_summary=precision,
                update_evidence=updates, gate_statistics=pair._gate_statistics(model),
                optimizer_step_seconds=timing['seconds'], step_profile=timing,
                new_stage_sample_first=(step - 1) * 8, new_stage_sample_last=step * 8 - 1,
                sample_namespace=NAMESPACE, micro_samples=samples,
                exposures_this_step=dict(J=4, L=4), exposures_since_stage_start=dict(J=step * 4, L=step * 4),
                elapsed_seconds=time.monotonic() - began, peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],
                host_peak_rss_bytes=memory.get('host_peak_rss_bytes'), samples_per_second=8 / timing['seconds'])
            append_jsonl(output / 'metrics.jsonl', dict(step=step, micro_batches_consumed=step * 8, **latest))
            print(json.dumps(dict(step=step, **latest)), flush=True)
            if step in (2, 20):
                profiler.begin_checkpoint(step)
                target = pair._save_checkpoint(output, _checkpoint(model, optimizer, config=config, hashes=hashes,
                    lineage=parent_lineage, pool=pool, step=step, metrics=latest,
                    a_visual_gate=a_visual_gate, a_reference=a_reference))
                profiler.end_checkpoint(step, checkpoint_path=str(target))
        result = dict(status='completed', stage=STAGE, arm='frozen', step=stop_after_step,
            micro_batches_consumed=stop_after_step * 8, new_stage_micro_batches_consumed=stop_after_step * 8,
            historical_micro_batches_before_stage=HISTORY, lifetime_micro_batches_consumed=HISTORY + stop_after_step * 8,
            parent_lineage=parent_lineage, sampling_contract=sampling_contract(config.training, pool),
            a_visual_gate=a_visual_gate, a_reference=a_reference,
            resume_step=start, checkpoint_path=str(target), configured_max_steps=20,
            trainable_parameters=summary.trainable_parameters, **latest,
            qualification=dict(adapter_updated=True, gate_updated=True, lora_trainable=False, gradients_finite=True,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']), quality_improvement_verified=False,
            generalization_verified=False, data_coverage_causal_proof=False)
        append_jsonl(output / 'execution_segments.jsonl', result)
        (output / 'execution_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    except BaseException as exc:
        profiler.close(error=exc)
        append_jsonl(output / 'failures.jsonl', dict(status='failed', stage=STAGE,
            failure_type=type(exc).__name__, error=str(exc), timestamp_unix=time.time(), resume_step=start))
        raise
    finally:
        profiler.close()
