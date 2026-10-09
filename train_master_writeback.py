"""Two finite precision-migration arms; unchanged BF16 forward, native 8-bit state.

Only the round arm discards FP32 Adapter master low bits after each update.
Old B checkpoints are validated as old B, never relabelled as this stage.
"""
from __future__ import annotations

import copy
import json
import math
from operator import index as integer_index
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

import train_paired_turn_teacher as b
from training.config import load_config as _load_config
from training.data.action_dataset import _seed64, cache_index_path
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file
from training.studio import paired_turn_b_eval as source
from training.studio.paired_turn_update_signal import equal_tree

STAGE = 'controlled_action_master_writeback_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_master_writeback_checkpoint_v1'
FACTORY = 'train_master_writeback:build_master_dataloader'
ROOT = b.ROOT.parent / 'teacher-master-writeback-v1'
DATA = b.DATA
ARMS = ('round', 'retain')
HISTORY, SAMPLE_OFFSET, MICRO_BATCHES = 480, 160, 160
NAMESPACE = b.NAMESPACE
GATE = 'model.controlled_action_gate'
PREFIX = 'model.act_control_adapter.'
ADAPTER_SHAPES = {
    PREFIX + 'conv.weight': (3072, 8192, 2, 2),
    PREFIX + 'conv.bias': (3072,),
    PREFIX + 'residual_blocks.0.conv1.weight': (3072, 3072, 3, 3),
    PREFIX + 'residual_blocks.0.conv1.bias': (3072,),
    PREFIX + 'residual_blocks.0.conv2.weight': (3072, 3072, 3, 3),
    PREFIX + 'residual_blocks.0.conv2.bias': (3072,),
}
require = b._require
validate_base_identity = b.validate_base_identity


def arm_for_config(config, arm=None):
    name = Path(config.training.output_dir).parent.name
    matches = [a for a in ARMS if name == f'cmaster-{a}-profile2-v1']
    require(len(matches) == 1 and (arm is None or arm == matches[0]), 'fixed arm output identity required')
    return matches[0]


def b_config(config):
    """Model/data-config view only; never used to relabel a checkpoint payload."""
    result = copy.deepcopy(config)
    result.data.data_factory = b.FACTORY
    return result


def model_config(config):
    return b.model_config(b_config(config))


def validate_probe_config(config, arm=None):
    arm_for_config(config, arm)
    require(config.data.data_factory == FACTORY, 'explicit master-writeback sampler required')
    b.validate_probe_config(b_config(config))
    require(config.data.manifest_path == str(DATA / 'manifests/train.jsonl'), 'same exact B cache only')


def load_config(path):
    config = _load_config(path)
    validate_probe_config(config)
    return config


def validate_config(config, *, output_dir, arm):
    validate_probe_config(config, arm)
    expected = ROOT / f'cmaster-{arm}-profile2-v1/training'
    require(Path(output_dir).resolve() == expected.resolve()
        and Path(config.training.output_dir).resolve() == expected.resolve(), 'fixed new training directory required')
    b.validate_config(b_config(config), data_root=DATA, output_dir=output_dir,
                      model_dir=Path(config.model.base_model_path))


def parent_binding():
    _, request, result = source.completed_endpoint()
    execution = result['execution']
    stat = source.CHECKPOINT.stat()
    return dict(mode='precision_migration_native_optimizer_state', parent_stage=b.STAGE,
        parent_checkpoint_schema=b.CHECKPOINT_SCHEMA, parent_step=20, parent_optimizer_step=20,
        parent_checkpoint=str(source.CHECKPOINT), parent_checkpoint_sha256=source.checkpoint_digest(),
        parent_checkpoint_stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        parent_config=str(source.TRAINING_CONFIG), parent_config_sha256=request['config_sha256'],
        parent_job=str(source.ENDPOINT),
        parent_records={n: sha256_file(source.ENDPOINT / (n+'.json')) for n in ('request','result','settlement')},
        base_identity=execution['parent_lineage']['base_identity'], ancestor_lineage=execution['parent_lineage'],
        optimizer_restored=True, rng_restored=True, parent_strict_resume=False, new_optimizer=False,
        historical_micro_batches_before_stage=HISTORY, r005_inherited=False, teacher64_inherited=False)


def validate_lineage(lineage):
    require(isinstance(lineage, dict) and lineage.get('mode') == 'precision_migration_native_optimizer_state'
        and lineage.get('parent_stage') == b.STAGE and lineage.get('parent_checkpoint_schema') == b.CHECKPOINT_SCHEMA
        and lineage.get('parent_checkpoint') == str(source.CHECKPOINT)
        and lineage.get('parent_config') == str(source.TRAINING_CONFIG)
        and lineage.get('parent_step') == lineage.get('parent_optimizer_step') == 20
        and lineage.get('historical_micro_batches_before_stage') == HISTORY
        and lineage.get('optimizer_restored') is True and lineage.get('rng_restored') is True
        and lineage.get('parent_strict_resume') is False and lineage.get('new_optimizer') is False
        and lineage.get('r005_inherited') is False and lineage.get('teacher64_inherited') is False,
        'only genuine B20 precision migration with original optimizer state permitted')


def artifact_hashes(config, config_path):
    validate_probe_config(config)
    result = b.artifact_hashes(b_config(config), config_path)
    result.update(master_parent_config=sha256_file(source.TRAINING_CONFIG),
                  master_parent_request=sha256_file(source.ENDPOINT / 'request.json'),
                  master_parent_result=sha256_file(source.ENDPOINT / 'result.json'),
                  master_parent_settlement=sha256_file(source.ENDPOINT / 'settlement.json'))
    return result


def load_parent(lineage):
    require(parent_binding() == lineage, 'B20 source metadata/stat changed')
    original = b.load_config(source.TRAINING_CONFIG)
    hashes = b.artifact_hashes(original, source.TRAINING_CONFIG)
    # Original B validator, original B configuration and original B hashes.
    payload = b.validate_resume(source.CHECKPOINT, config=original, hashes=hashes)
    b.validate_base_identity(payload, original)
    require(payload['step'] == 20 and payload['lifetime_micro_batches_consumed'] == HISTORY,
            'actual completed B20, not an A/P or relabelled payload, required')
    _, _, result = source.completed_endpoint()
    for key in ('parent_lineage', 'sampling_contract', 'a_visual_gate', 'a_reference'):
        require(payload[key] == result['execution'][key], 'B endpoint payload changed: '+key)
    optimizer_mapping(payload['model_signature'], payload['optimizer'], internal_step=20)
    return payload


def precision_contract():
    return dict(backbone_parameter_dtype='bfloat16', adapter_parameter_dtype='bfloat16',
        adapter_gradient_dtype='bfloat16', gate_parameter_dtype='float32', gate_gradient_dtype='float32',
        frozen_lora_parameter_dtype='float32', forward_autocast_dtype='bfloat16',
        master_parameter_dtype='float32', separate_master_copy=True,
        master_gradient='exact_float32_widening_of_globally_clipped_BF16_model_gradient',
        optimizer='native_AdamW8bit_original_serialized_moments_maps_scales',
        no_FP32_forward_or_FP32_backprop=True)


def method_contract(arm):
    require(arm in ARMS, 'fixed round/retain arm required')
    return dict(method=STAGE, arm=arm, intervention='discard_master_low_bits_after_step' if arm == 'round'
        else 'retain_master_low_bits_across_steps', ratio_cap=.1, adapter_lr=1e-5, gate_lr=.01,
        adapter_weight_decay=.01, gate_weight_decay=0., betas=[.9,.95], global_clip_norm=1.,
        lora_trainable=False, seed=42, max_new_steps=20, precision=precision_contract(),
        parent_strict_resume=False, fresh_from_official_base=False, intervention_parameters=6,
        gate_and_optimizer_moments_projected=False, generalization_experiment=False,
        reference_memory_implemented=False, automatic_downstream_training=False)


def sampling_contract(training, pool):
    b._validate_pool(pool)
    require(training.seed == 42 and training.max_steps == 20 and training.micro_batch_size == 1
            and training.gradient_accumulation_steps == 8, 'fixed20x8 sampling required')
    return dict(namespace=NAMESPACE, data_seed=_seed64(42, NAMESPACE), training_seed=42,
        virtual_samples=160, first_stage_sample_index=0, last_stage_sample_index=159,
        first_absolute_sample_index=160, last_absolute_sample_index=319, exposures_per_window=80,
        historical_micro_batches_before_stage=HISTORY, each_optimizer_step=dict(J=4,L=4),
        diffusion_seed_policy='seed64(seed64(42,namespace),diffusion,160+new_stage_index)',
        index_policy='new_stage0..159_maps_original_sampler160..319', pool=pool)


class MasterDataset(b.PairedTurnDataset):
    def sampling_spec(self, index):
        require(not isinstance(index, bool), 'sample index cannot be bool')
        index = integer_index(index)
        if not 0 <= index < MICRO_BATCHES:
            raise IndexError(index)
        absolute = SAMPLE_OFFSET + index
        row = self.pool['windows'][absolute % 2]
        return dict(absolute_sample_index=absolute, new_stage_sample_index=index, index_namespace=NAMESPACE,
            episode_slot=self.fixed_slots[absolute % 2], episode_id=row['episode_id'],
            window_index=row['window_index'], source_start=row['source_start'], target=row['target'],
            diffusion_seed=_seed64(self.data_seed, 'diffusion', absolute))


def build_master_dataloader(*, config, training, pool=None):
    pool = b.pool_contract() if pool is None else pool
    sampling_contract(training, pool)
    dataset = MasterDataset(cache_index_path(config.manifest_path), pool=pool, manifest_path=config.manifest_path,
        split='train', seed=42, timestep_shift=5., prompt_cache_path=None, num_frames=97)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)


def _build_model(config, device, arm=None):
    validate_probe_config(config, arm)
    return b._build_model(b_config(config), device)


def optimizer_mapping(signature, state, *, internal_step):
    groups = state.get('param_groups', [])
    expected_names = list(ADAPTER_SHAPES) + [GATE]
    require(set(signature) == set(expected_names), 'all six Adapter tensors and original gate required')
    names = [n for n, s in signature.items() if s['group'] == 'adapter'] + [GATE]
    require(names == expected_names, 'original optimizer name ordering changed')
    require([g.get('name') for g in groups] == ['action_adapter','controlled_action_gate'], 'original optimizer groups required')
    mapping, ids = [], []
    for group, group_names, lr, decay in zip(groups, (names[:-1], [GATE]), (1e-5,.01), (.01,0.)):
        require(group.get('lr') == lr and group.get('weight_decay') == decay
            and tuple(group.get('betas', ())) == (.9,.95) and len(group.get('params', [])) == len(group_names),
            'native optimizer hyperparameters/order changed')
        if 'param_names' in group:
            require(group['param_names'] == group_names, 'saved optimizer names differ')
        for name, identifier in zip(group_names, group['params']):
            require(type(identifier) is int and identifier == len(ids), 'fixed original optimizer IDs required')
            slot = state.get('state', {}).get(identifier)
            require(isinstance(slot, dict) and slot.get('step') == internal_step, 'optimizer internal step differs')
            quant = slot.get('__bnb_optimizer_quant_state__')
            require(isinstance(quant, dict) and all(k in quant for k in ('state1','state2')),
                    'native serialized BNB state required, not generic AdamW')
            shape = ADAPTER_SHAPES[name] if name != GATE else (3072,)
            dtype = 'torch.bfloat16' if name != GATE else 'torch.float32'
            require(signature[name] == dict(shape=list(shape), dtype=dtype, group='adapter' if name != GATE else 'gate'),
                    'original forward parameter signature changed')
            large = math.prod(shape) >= 4096
            for key in ('state1','state2'):
                require(torch.is_tensor(quant[key]) and tuple(quant[key].shape) == shape
                    and quant[key].dtype == (torch.uint8 if large else torch.float32), 'native moment precision/geometry differs')
            if large:
                for key in ('qmap1','qmap2','absmax1','absmax2'):
                    length = 256 if key.startswith('qmap') else math.ceil(math.prod(shape)/256)
                    require(torch.is_tensor(quant.get(key)) and quant[key].dtype == torch.float32
                        and tuple(quant[key].shape) == (length,), 'native map/scale precision changed')
            mapping.append(dict(name=name, optimizer_id=identifier, group=group['name'],
                forward_dtype=dtype, optimizer_parameter_dtype='torch.float32', shape=list(shape),
                separate_master=name != GATE))
            ids.append(identifier)
    require(set(state.get('state', {})) == set(ids), 'missing/extra optimizer slots')
    return mapping


def make_masters(model, saved=None):
    named = dict(model.named_parameters())
    require(saved is None or set(saved) == set(ADAPTER_SHAPES), 'complete master state required')
    result = {}
    for name, shape in ADAPTER_SHAPES.items():
        parameter = named[name]
        require(parameter.requires_grad and parameter.dtype == torch.bfloat16 and tuple(parameter.shape) == shape,
                'BF16 forward parameter geometry changed')
        value = parameter.detach().float() if saved is None else saved[name].to(device=parameter.device)
        require(value.dtype == torch.float32 and value.shape == parameter.shape
            and torch.equal(value.bfloat16(), parameter.detach()), 'master projection differs from forward weight')
        result[name] = torch.nn.Parameter(value.clone())
    return result


def make_optimizer(model, masters, source_state, *, check=lambda: None):
    import bitsandbytes as bnb
    named = dict(model.named_parameters())
    groups = copy.deepcopy(source_state['param_groups'])
    groups[0]['params'] = list(masters.values())
    groups[1]['params'] = [named[GATE]]
    optimizer = bnb.optim.AdamW8bit(groups, betas=(.9,.95))
    optimizer.load_state_dict(source_state)
    require(equal_tree(source_state, optimizer.state_dict(), check), 'state_restore_not_matched')
    return optimizer


def copy_clipped_gradients(model, masters):
    named = dict(model.named_parameters())
    for name, master in masters.items():
        grad = named[name].grad
        require(grad is not None and grad.dtype == torch.bfloat16, 'original BF16 clipped gradient missing')
        require(bool(torch.isfinite(grad).all()), 'nonfinite complete Adapter gradient')
        master.grad = grad.detach().float()
    gate_grad = named[GATE].grad
    require(gate_grad is not None and gate_grad.dtype == torch.float32
        and bool(torch.isfinite(gate_grad).all()), 'finite FP32 gate gradient required')


def writeback(model, masters, arm):
    require(arm in ARMS, 'round/retain intervention only')
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, master in masters.items():
            require(bool(torch.isfinite(master).all()), 'nonfinite master update')
            projected = master.bfloat16()
            if arm == 'round':
                master.copy_(projected.float())
            named[name].copy_(projected)


def storage_evidence(model, masters, arm):
    named = dict(model.named_parameters())
    for name, master in masters.items():
        require(master.dtype == torch.float32 and named[name].dtype == torch.bfloat16
            and torch.equal(master.detach().bfloat16(), named[name].detach()), 'model/master projection mismatch')
        if arm == 'round':
            require(torch.equal(master.detach(), master.detach().bfloat16().float()), 'round arm retained low bits')
    return dict(model_master_projection_equal=True, master_parameters=6, master_dtype='torch.float32',
        forward_parameter_dtype='torch.bfloat16', round_projection_applied=arm == 'round')


def sampled_masters(masters):
    return {n: p.detach().reshape(-1)[::max(1, math.ceil(p.numel()/4096))][:4096].float().cpu().clone()
            for n, p in masters.items()}


def model_update_evidence(model, before):
    after = b.pair._sample_parameters(model)
    require(set(before) == set(after), 'forward parameter sample names changed')
    groups = {}
    for name, value in after.items():
        require(bool(torch.isfinite(value).all()), 'nonfinite forward parameter sample')
        group = b.pair.parent._group(name)
        row = groups.setdefault(group, dict(sampled_elements=0, changed_elements=0))
        row['sampled_elements'] += value.numel()
        row['changed_elements'] += int(torch.count_nonzero(value.float()-before[name].float()))
    require(groups['gate']['changed_elements'] > 0, 'no actual FP32 gate update')
    # A retained master can change without this step crossing a BF16 bin.
    # That is the intervention being tested, not a reason to stop it early.
    return dict(groups=groups, sampled_not_full_parameter_census=True, gate_updated=True,
        adapter_forward_updated=groups['adapter']['changed_elements'] > 0)


def _checkpoint(model, masters, optimizer, *, config, hashes, arm, lineage, pool, step, metrics):
    state = optimizer.state_dict()
    signature = b.pair._model_signature(model, 'frozen')
    return dict(format_version=1, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, arm=arm,
        config=config.to_dict(), manifest_hashes=hashes, base_identity=lineage['base_identity'], parent_lineage=lineage,
        method_contract=method_contract(arm), precision_contract=precision_contract(),
        sampling_contract=sampling_contract(config.training, pool), model_signature=signature,
        trainable_model=trainable_state_dict(model), master_parameters={n:p.detach().cpu() for n,p in masters.items()},
        master_mapping=optimizer_mapping(signature, state, internal_step=20+step), optimizer=state,
        step=step, optimizer_internal_step=20+step, micro_batches_consumed=step*8,
        new_stage_micro_batches_consumed=step*8, historical_micro_batches_before_stage=HISTORY,
        lifetime_micro_batches_consumed=HISTORY+step*8, sampling_cursor=SAMPLE_OFFSET+step*8,
        metrics=metrics, torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes, arm=None):
    arm = arm_for_config(config, arm)
    validate_probe_config(config, arm)
    require(payload.get('format_version') == 1 and payload.get('stage') == STAGE
        and payload.get('checkpoint_schema') == CHECKPOINT_SCHEMA and payload.get('arm') == arm,
        'independent master-writeback stage/arm required')
    lineage = payload['parent_lineage']; validate_lineage(lineage)
    pool = payload.get('sampling_contract', {}).get('pool'); b._validate_pool(pool)
    expected = dict(config=config.to_dict(), manifest_hashes=hashes, base_identity=lineage['base_identity'],
        method_contract=method_contract(arm), precision_contract=precision_contract(),
        sampling_contract=sampling_contract(config.training, pool))
    for key, value in expected.items():
        require(payload.get(key) == value, 'strict master stage contract changed: '+key)
    step = payload.get('step')
    require(type(step) is int and step in (2,20)
        and payload.get('optimizer_internal_step') == 20+step
        and payload.get('micro_batches_consumed') == payload.get('new_stage_micro_batches_consumed') == step*8
        and payload.get('historical_micro_batches_before_stage') == HISTORY
        and payload.get('lifetime_micro_batches_consumed') == HISTORY+step*8
        and payload.get('sampling_cursor') == SAMPLE_OFFSET+step*8, 'sampling/history/optimizer cursor differs')
    state, masters, signature = payload['trainable_model'], payload['master_parameters'], payload['model_signature']
    require(set(state) == set(signature) == set(ADAPTER_SHAPES) | {GATE}
        and set(masters) == set(ADAPTER_SHAPES), 'complete BF16 export and six masters required')
    for name, value in state.items():
        shape = ADAPTER_SHAPES[name] if name != GATE else (3072,)
        dtype = torch.bfloat16 if name != GATE else torch.float32
        require(torch.is_tensor(value) and value.dtype == dtype and tuple(value.shape) == shape
            and bool(torch.isfinite(value).all()), 'invalid forward export tensor')
        if name != GATE:
            master = masters[name]
            require(master.dtype == torch.float32 and master.shape == value.shape
                and bool(torch.isfinite(master).all()) and torch.equal(master.bfloat16(), value), 'invalid saved master/projection')
            if arm == 'round':
                require(torch.equal(master, value.float()), 'round checkpoint retains forbidden low bits')
    require(payload['master_mapping'] == optimizer_mapping(signature, payload['optimizer'], internal_step=20+step),
            'saved master/optimizer name mapping changed')
    return payload


def validate_resume(path, *, config, hashes, arm):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(payload, config=config, hashes=hashes, arm=arm)
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    random.Random().setstate(payload['python_rng_state']); np.random.RandomState().set_state(payload['numpy_rng_state'])
    rng = payload.get('cuda_rng_state_all')
    require(isinstance(rng,list) and len(rng) == 1 and rng[0].dtype == torch.uint8
        and rng[0].ndim == 1 and rng[0].numel() > 0, 'complete single-device RNG required')
    return payload


def launch(config, config_path, *, arm, lineage, pool, stop_after_step, resume=None,
           shared_lease, shared_max_seconds, shared_request_sha256):
    from training.demo.backend import validate_active_authority
    validate_config(config, output_dir=config.training.output_dir, arm=arm)
    validate_lineage(lineage); b._validate_pool(pool)
    require(stop_after_step in (2,20) and 60 <= shared_max_seconds <= (900 if stop_after_step == 2 else 3600),
            'finite2/20 job required')
    output = Path(config.training.output_dir)
    if resume is None:
        require(stop_after_step == 2 and (not output.exists() or not any(output.iterdir())), 'new profile only')
    else:
        require(Path(resume) == output/'checkpoints/step-0000002.pt' and stop_after_step == 20, 'strict same-arm2->20 only')
    require(parent_binding() == lineage and b.pool_contract() == pool, 'parent/cache binding changed')
    hashes = artifact_hashes(config, config_path)
    saved = validate_resume(resume, config=config, hashes=hashes, arm=arm) if resume else load_parent(lineage)
    require(saved['parent_lineage'] == lineage if resume else saved['stage'] == b.STAGE, 'wrong resume/source lineage')
    start = saved['step'] if resume else 0
    gpu = b.teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds, request_sha256=shared_request_sha256)
    require(torch.cuda.is_available(), 'leased CUDA required')
    device = torch.device('cuda',0); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42); output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, arm=arm, config=config.to_dict(), manifest_hashes=hashes, parent_lineage=lineage,
        method_contract=method_contract(arm), sampling_contract=sampling_contract(config.training,pool),
        resume=resume, resume_step=start, stop_after_step=stop_after_step, git_revision=git_revision(Path(__file__).parent),
        entry_sha256=sha256_file(__file__), gpu_state=gpu.as_dict(), started_unix=time.time(),
        shared_guard=dict(lease=shared_lease,max_seconds=shared_max_seconds,request_sha256=shared_request_sha256))
    if not resume:
        with (output/'run_metadata.json').open('x',encoding='utf-8') as f: json.dump(metadata,f,indent=2)
    append_jsonl(output/'executions.jsonl',metadata)
    profiler = OptimizerStepProfiler(device,output_dir=output/'profile_output',gpu_uuid=shared_lease['gpu_uuid'])
    began = time.monotonic(); last_authority_check = 0.
    def check():
        nonlocal last_authority_check
        now = time.monotonic()
        require(now-began < shared_max_seconds-30, 'finite training deadline reached')
        if now-last_authority_check >= 2:
            validate_active_authority(shared_lease)
            last_authority_check = now
    try:
        with profiler.phase('model_load'):
            model, summary = _build_model(config,device,arm)
            load_trainable_state_dict(model,saved['trainable_model'])
            require(b.pair._model_signature(model,'frozen') == saved['model_signature'], 'forward signature differs')
            masters = make_masters(model,saved['master_parameters'] if resume else None)
            optimizer_mapping(saved['model_signature'], saved['optimizer'],internal_step=20+start)
            optimizer = make_optimizer(model,masters,saved['optimizer'],check=check)
        loader = build_master_dataloader(config=config.data,training=config.training,pool=pool)
        b.teacher._restore_rng_state(saved,require_complete=True)
        del saved
        model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
        latest = {}
        for step in range(start+1,stop_after_step+1):
            check(); before = sampled_masters(masters); before_model = b.pair._sample_parameters(model)
            profiler.begin_step(step); losses,samples = [],[]
            for offset in range(8):
                check(); index=(step-1)*8+offset; profiler.set_phase('data_wait')
                batch=default_collate([loader.dataset[index]])
                samples.append(dict(**loader.dataset.sampling_spec(index),actual_timestep=float(batch['timesteps'][0,1])))
                profiler.set_phase('forward')
                with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=b.teacher._forward_loss(model,batch,config,device)
                require(bool(torch.isfinite(loss)), 'nonfinite loss')
                profiler.set_phase('backward'); (loss/8).backward(); losses.append(float(loss.detach()))
                samples[-1]['loss'] = losses[-1]
                del loss,batch
            precision=b.pair._gradient_precision(model,'frozen')
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            require(bool(torch.isfinite(norm)), 'nonfinite global gradient norm')
            copy_clipped_gradients(model,masters)
            precision['gradients_finite'] = True
            profiler.set_phase('optimizer_update'); optimizer.step(); writeback(model,masters,arm)
            optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
            storage=storage_evidence(model,masters,arm)
            after=sampled_masters(masters)
            changed=sum(int(torch.count_nonzero(after[n]-before[n])) for n in before)
            residuals=torch.cat([(v-v.bfloat16().float()).reshape(-1) for v in after.values()])
            require(changed>0, 'no actual sampled master update')
            model_updates=model_update_evidence(model,before_model)
            del before,after,before_model
            timing=profiler.end_step(step); memory=timing['memory']
            latest=dict(loss=sum(losses)/8,grad_norm=float(norm),precision_summary=precision,
                update_evidence=dict(master_changed_sampled_elements=changed,master_updated=True,
                    sampled_not_full_census=True,model_updates=model_updates,
                    includes_weight_decay_and_historical_momentum=True,parameter_change_is_not_quality=True,
                    master_residual_sampled_rms=float(residuals.square().mean().sqrt()),
                    master_residual_sampled_nonzero=int(torch.count_nonzero(residuals))),storage_evidence=storage,
                native_state_restore_matched=True,gate_statistics=b.pair._gate_statistics(model),
                optimizer_step_seconds=timing['seconds'],step_profile=timing, micro_samples=samples,
                sample_namespace=NAMESPACE,new_stage_sample_first=(step-1)*8,new_stage_sample_last=step*8-1,
                absolute_sample_first=SAMPLE_OFFSET+(step-1)*8,absolute_sample_last=SAMPLE_OFFSET+step*8-1,
                exposures_since_stage_start=dict(J=step*4,L=step*4),
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],elapsed_seconds=time.monotonic()-began)
            append_jsonl(output/'metrics.jsonl',dict(step=step,optimizer_internal_step=20+step,**latest))
            print(json.dumps(dict(step=step,arm=arm,**latest)),flush=True)
            if step in (2,20):
                profiler.begin_checkpoint(step)
                target=b.pair._save_checkpoint(output,_checkpoint(model,masters,optimizer,config=config,hashes=hashes,
                    arm=arm,lineage=lineage,pool=pool,step=step,metrics=latest))
                profiler.end_checkpoint(step,checkpoint_path=str(target))
        result=dict(status='completed',stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,step=stop_after_step,
            optimizer_internal_step=20+stop_after_step,micro_batches_consumed=stop_after_step*8,
            new_stage_micro_batches_consumed=stop_after_step*8,historical_micro_batches_before_stage=HISTORY,
            lifetime_micro_batches_consumed=HISTORY+stop_after_step*8,sampling_cursor=SAMPLE_OFFSET+stop_after_step*8,
            parent_lineage=lineage,sampling_contract=sampling_contract(config.training,pool),method_contract=method_contract(arm),
            precision_contract=precision_contract(),resume_step=start,checkpoint_path=str(target),configured_max_steps=20,
            trainable_parameters=summary.trainable_parameters,**latest,
            qualification=dict(master_updated=True,adapter_updated=model_updates['adapter_forward_updated'],gate_updated=True,gradients_finite=True,
                native_state_restore_matched=True,model_master_projection_equal=True,lora_trainable=False,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']),quality_improvement_verified=False)
        append_jsonl(output/'execution_segments.jsonl',result)
        (output/'execution_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException as exc:
        profiler.close(error=exc)
        append_jsonl(output/'failures.jsonl',dict(status='failed',stage=STAGE,arm=arm,
            failure_type=type(exc).__name__,error=str(exc),resume_step=start,timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
