"""Native4-step streaming adapters and truthful RTX3090 runtime row registry.

Only instrumentation/CPU contracts have been tested here. Real checkpoints,
the unchanged Creator lease, native CUDA probes and browser ACKs are mandatory
before reporting GPU throughput or input-to-visible-frame latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time


LATENT_CHANNELS = 48
LATENTS_PER_CHUNK = 3
FRAMES_PER_CHUNK = 12
SAMPLING_STEPS = 4
ACTION_KEYS = ('W', 'A', 'S', 'D', 'I', 'J', 'K', 'L')
DATA_ROOT = Path('/DATA_DISK/InteractWorld')
STUDENT_STAGES = ('completion_causal_ode_sufficiency_v2', 'completion_longforcing_selfrollout_dmd_v2')


class RuntimeContractError(RuntimeError): pass


def require(ok, message):
    if not ok: raise RuntimeContractError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def native_bf16_autocast(device):
    import torch
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def request_allowed_cards(lease):
    """Card selection comes from the actual pinned request, not a widened constant.

    The original backend validates the all-card supplemental authority. Runtime
    additionally checks its own immutable coordinator membership before CUDA.
    Legacy two-card runtime requests keep their original restriction.
    """
    directory=Path(lease.get('job_directory',''))
    if not directory.is_absolute() or not (directory/'request.json').is_file():
        return (1,2)
    request=_read(directory/'request.json')
    binding=request.get('parallel_budget_binding')
    if binding is None:return (1,2)
    from training.completion_ode_jobs_v2 import parallel_manifest
    manifest=parallel_manifest(binding)
    row=next((r for r in manifest['jobs'] if r['job_id']==lease.get('job_id')),None)
    require(row is not None and row['directory']==str(directory)
        and row['gpu_index']==lease.get('gpu_index')==request.get('gpu_index'),
        'runtime card must match exact immutable completion job')
    return tuple(binding['allowed_gpu_indices'])


def _pin(pin, *, parse=True):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and Path(pin['path']).is_absolute(),
            'actual small record absolute path/hash required')
    path = Path(pin['path'])
    require(path.is_file() and path.stat().st_size <= 20_000_000 and file_sha256(path) == pin['sha256'],
            'actual producer record is missing or changed')
    return _read(path) if parse else path


def validate_student_binding(binding, *, ledger_path, fixture_root=None):
    """CPU fail-closed prepare: actual settled self-trained4step producer, no fake parent.

    This checks records/stat only. A native model loader MUST independently
    validate the full checkpoint schema/model/config before using its tensors.
    """
    root = Path(fixture_root).resolve() if fixture_root is not None else DATA_ROOT
    require(isinstance(binding, dict) and binding.get('schema') == 'completion_runtime_actual_student_v2'
            and binding.get('sampling_steps') == 4 and binding.get('external_cfg_scale') == 1.
            and binding.get('latent_channels') == 48, 'self-trained4step CFG1 Wan2.2 student binding required')
    records = binding.get('records')
    require(isinstance(records, dict) and set(records) == {'request', 'result', 'settlement'},
            'actual producer request/result/settlement pins required')
    request, result, settlement = (_pin(records[k]) for k in ('request', 'result', 'settlement'))
    directory = Path(records['request']['path']).parent.resolve()
    require(root in directory.parents
            and all(Path(records[k]['path']).resolve() == directory/f'{k}.json' for k in records),
            'producer small records must share one actual data job directory')
    execution = result.get('execution', {})
    require(request.get('job_id') == binding.get('producer_job_id') == directory.name
            and result.get('status') == 'completed' and result.get('job_id') == request['job_id']
            and result.get('request_sha256') == records['request']['sha256']
            and result.get('mode') == request.get('mode')
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and execution.get('stage') in STUDENT_STAGES and execution.get('step') == binding.get('endpoint_step'),
            'completed settled actualODE/DMD endpoint required, not teacher/officialstudent/TODO')
    ledger_path = Path(ledger_path).resolve()
    require(root in ledger_path.parents, 'same data-root original ledger required')
    require(fixture_root is not None or ledger_path==DATA_ROOT/'creator-runtime/budget.json',
            'runtime cannot substitute another project ledger or reset accounting')
    ledger = _read(ledger_path)
    row = ledger.get('reservations', {}).get(settlement.get('reservation_id'), {})
    require(ledger.get('version') == 1 and row.get('status') == 'settled'
            and row.get('settlement') == settlement and row.get('charged_seconds') == settlement.get('charged_gpu_seconds')
            and row.get('context', {}).get('job_id') == request['job_id']
            and Path(row['context']['job_directory']).resolve() == directory
            and row['context'].get('request_sha256') == records['request']['sha256'],
            'actual endpoint settlement must match the unchanged persistent ledger')
    checkpoint = binding.get('checkpoint', {})
    path = Path(checkpoint.get('path', '')).resolve()
    require(root in path.parents and path.is_file(), 'actual student checkpoint does not yet exist')
    stat = path.stat()
    require(checkpoint.get('bytes') == stat.st_size and checkpoint.get('mtime_ns') == stat.st_mtime_ns
            and execution.get('checkpoint_path') == str(path), 'actual checkpoint stat/path changed')
    config = binding.get('config')
    _pin(config, parse=False)  # Native configs may be YAML rather than JSON.
    require(root in Path(config['path']).resolve().parents, 'existing data-root model config required')
    return dict(producer_job_id=request['job_id'], stage=execution['stage'], step=execution['step'],
                records_verified=True, checkpoint_stat_verified=True,
                full_checkpoint_schema_verified=False, gpu_profile_verified=False)


def student_binding_from_job(directory, *, ledger_path=DATA_ROOT/'creator-runtime/budget.json'):
    """Derive pins only after an actual finalODE80 or finalDMD32 settlement."""
    directory=Path(directory)
    require(directory.is_absolute() and DATA_ROOT in directory.parents,'actual data-root producer required')
    pins={key:dict(path=str(directory/f'{key}.json'),sha256=file_sha256(directory/f'{key}.json'))
          for key in ('request','result','settlement')}
    request,result=_read(directory/'request.json'),_read(directory/'result.json')
    execution=result.get('execution',{});stage,step=execution.get('stage'),execution.get('step')
    require((stage==STUDENT_STAGES[0] and step==80) or (stage==STUDENT_STAGES[1] and step==32),
            'fixed finalODE80/DMD32 endpoint only, not intermediate best-checkpoint selection')
    path=Path(execution.get('checkpoint_path',''));require(path.is_file(),'actual final student not yet available')
    if stage==STUDENT_STAGES[0]:config=Path(request['config_path']);config_hash=request['config_sha256']
    else:
        parent=request['actual_parent_binding'];config=Path(parent['config']);config_hash=parent['config_sha256']
    stat=path.stat()
    binding=dict(schema='completion_runtime_actual_student_v2',producer_job_id=request['job_id'],endpoint_step=step,
        sampling_steps=4,external_cfg_scale=1.,latent_channels=48,records=pins,
        checkpoint=dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
        config=dict(path=str(config),sha256=config_hash))
    validate_student_binding(binding,ledger_path=ledger_path)
    return binding


def load_actual_student_model(binding,device,check):
    """Real project native loaders; no officialstudent or loose state_dict fallback."""
    validate_student_binding(binding,ledger_path=DATA_ROOT/'creator-runtime/budget.json')
    producer=_pin(binding['records']['request']);result=_pin(binding['records']['result'])['execution']
    if result['stage']==STUDENT_STAGES[0]:
        from training import completion_ode_jobs_v2 as jobs
        request=copy.deepcopy(producer)
        arguments=request['resolved_arguments']
        arguments.update(endpoint_step=binding['endpoint_step'],endpoint_profile=copy.deepcopy(binding['records']))
        request['endpoint_binding']=jobs._endpoint(arguments,parent=arguments['parent'],binding=arguments['cache_binding'])
        model,config,_=jobs._load_model(request,device,check)
    else:
        import train_completion_dmd_sufficiency as dmd
        request=copy.deepcopy(producer);request['arguments']['endpoint_step']=binding['endpoint_step']
        request['endpoint_checkpoint_stat']={key:binding['checkpoint'][key] for key in ('bytes','mtime_ns')}
        model,banks,config=dmd._endpoint_model(request,device,check)
        require(banks.current=='generator','actualDMD generator bank required, not frozenreal/fake score')
        model.requires_grad_(False).eval()
    require(not model.training,'actual runtime student must be frozen inference model')
    return model,config


def load_actual_wan_vae(config,device):
    """Existing official48channel fullVAE, strict key/shape check atop old wrapper."""
    import torch
    from utils.wan_wrapper import WanVAEWrapper
    from wan.modules.vae2_2 import _normalize_vae_state_dict
    path=Path(config.model.base_model_path)/'Wan2.2_VAE.pth'
    require(DATA_ROOT in path.parents and path.is_file() and not path.is_symlink(),
            'existing data-root officialWan2.2VAE required, no random/missing LightVAE fallback')
    state=_normalize_vae_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True))
    wrapper=WanVAEWrapper(vae_type='Wan2.2_VAE',pretrained_path=str(path))
    native=wrapper.model.state_dict()
    require(set(state)==set(native) and all(state[key].shape==native[key].shape for key in native),
            'strict fullVAE keys/shapes required; upstream strict=False logging is insufficient')
    del state,native
    return wrapper.requires_grad_(False).eval().to(device=device,dtype=torch.bfloat16)


def kv_storage_bytes(*, layers, heads, head_dim, height=30, width=52,
                     window_latents=7, dtype_bytes=2, cross_tokens=512):
    """Exact K/V tensor bytes only; NOT a peak-VRAM prediction or launch approval."""
    values=(layers,heads,head_dim,height,width,window_latents,dtype_bytes,cross_tokens)
    require(all(type(v) is int and v>0 for v in values) and height%2==width%2==0,
            'positive native token/cache dimensions required')
    tokens=height*width//4
    return dict(video_kv_bytes=layers*2*window_latents*tokens*heads*head_dim*dtype_bytes,
                cross_kv_bytes=layers*2*cross_tokens*heads*head_dim*dtype_bytes,
                excludes_weights_vae_activations_allocator=True)


def validate_native_gate(gate,binding,*,ledger_path,local_window_latents,fixture_root=None):
    """Actual settled same-student runtime probe, not hand-written pass booleans."""
    require(isinstance(gate,dict) and gate.get('schema')=='completion_runtime_native_gate_v2',
            'actual native runtime gate required')
    pins=gate.get('actual_profile_records')
    require(isinstance(pins,dict) and set(pins)=={'request','result','settlement'},'actual native probe pins required')
    request,result,settlement=(_pin(pins[k]) for k in ('request','result','settlement'))
    directory=Path(pins['request']['path']).parent.resolve()
    root=Path(fixture_root).resolve() if fixture_root else DATA_ROOT
    require(root in directory.parents and all(Path(pins[k]['path']).resolve()==directory/f'{k}.json' for k in pins)
        and directory.name==request.get('job_id') and request.get('mode')=='completion_runtime_v2_probe'
        and request.get('schema')=='completion_runtime_probe_request_v2'
        and request.get('student_binding')==binding and request.get('local_window_latents')==local_window_latents
        and result.get('status')=='completed' and result.get('request_sha256')==pins['request']['sha256']
        and result.get('job_id')==request['job_id'] and result.get('mode')==request['mode']
        and settlement.get('status')=='settled' and settlement.get('outcome')=='completed',
        'actual bounded completed settled same-student/method probe required')
    ledger=_read(ledger_path);row=ledger.get('reservations',{}).get(settlement.get('reservation_id'),{})
    require(row.get('status')=='settled' and row.get('settlement')==settlement
        and row.get('charged_seconds')==settlement.get('charged_gpu_seconds')
        and row.get('context',{}).get('job_id')==request['job_id']
        and row['context'].get('job_directory')==str(directory)
        and row['context'].get('request_sha256')==pins['request']['sha256'],'native probe must match original persistentledger')
    value={k:v for k,v in gate.items() if k!='actual_profile_records'}
    require(result.get('execution',{}).get('native_runtime_gate')==value
        and value.get('producer_job_id')==binding['producer_job_id']
        and value.get('runtime_source_sha256')==file_sha256(__file__)
        and value.get('checkpoint_schema_verified') is True and value.get('installed_causality_verified') is True
        and value.get('vae48_continuous_verified') is True
        and finite(value.get('peak_reserved_bytes')) and 0<value['peak_reserved_bytes']<23*2**30,
        'native gate fields/source/resource cannot be invented or changed after real probe')
    return value


def runtime_rows(*, inventory=None):
    """Official Table2 coverage, NOT copied paper timings or fabricated project numbers."""
    inventory = inventory or {}
    measured = dict(dit_ms=None, kv_commit_ms=None, vae_ms=None, transfer_ms=None, encode_ms=None,
                    generation_fps=None, browser_presented_fps=None, peak_reserved_bytes=None,
                    input_to_presented_p50_ms=None, input_to_presented_p95_ms=None)
    rows = [
        dict(row_id='base', components=['bf16_dit', 'wan22_full_vae'], status='pending_gpu_profile'),
        dict(row_id='sage2', components=['bf16_dit', 'sageattention2', 'wan22_full_vae'],
             status='pending_dependency', reason='sageattention distribution/version2 not installed/verified'),
        dict(row_id='sage2_lightvae', components=['bf16_dit', 'sageattention2', 'lightvae48'],
             status='pending_assets', reason='actual matching48channel pruned decoder andSage2 not verified'),
        dict(row_id='sage2_lightvae_fp8', components=['native_fp8_dit', 'sageattention2', 'lightvae48'],
             status='hardware_unsupported', reason='RTX3090 sm86 has no nativeFP8 matrix instruction path'),
        dict(row_id='sage2_lightvae_fp8_fastrope', components=['native_fp8_dit', 'sageattention2', 'lightvae48', 'fast_rope'],
             status='hardware_unsupported', reason='nativeFP8 row is not BF16 emulation'),
        dict(row_id='sage2_lightvae_mxfp6_fastrope', components=['native_mxfp6_dit', 'sageattention2', 'lightvae48', 'fast_rope'],
             status='hardware_unsupported', reason='BlackwellMXFP6 native path unavailable onRTX3090'),
        dict(row_id='sage2_lightvae_mxfp4_fastrope', components=['native_mxfp4_dit', 'sageattention2', 'lightvae48', 'fast_rope'],
             status='hardware_unsupported', reason='BlackwellMXFP4/Sage3sm120 path unavailable onRTX3090'),
        dict(row_id='project_stream_cached_vae', components=['bf16_dit', 'positive_cfg1', 'cached_wan22_vae', 'browser_ack'],
             status='implemented_cpu_pending_native_profile', paper_equivalent=False),
        dict(row_id='project_local_kv7', components=['bf16_dit', 'absolute_rope', 'sink1_local7', 'cached_wan22_vae', 'browser_ack'],
             status='implemented_cpu_pending_native_causality_and_resource_probe', paper_equivalent=False,
             semantic_change='retainfirst latent+last6 positions; evictedhistory information is lost'),
        dict(row_id='project_bf16_fastrope', components=['bf16_dit', 'fast_rope', 'cached_wan22_vae', 'browser_ack'],
             status='pending_numeric_equivalence_and_native_profile', paper_equivalent=False),
        dict(row_id='project_module_schedule', components=['bf16_dit', 'scheduled_module_transfer', 'cached_wan22_vae'],
             status='pending_implementation_and_profile', paper_equivalent=False),
    ]
    for row in rows:
        row.update(data_origin='project_runtime_plan_not_measured', hardware='RTX3090',
                   sampling_steps=4, external_cfg_scale=1., resolution=[832, 480], fps=16,
                   latent_channels=48, latents_per_chunk=3, frames_per_chunk=12,
                   same_student_and_input_required=True, measurements=dict(measured), inventory=inventory)
    return rows


class PhaseTimer:
    """Synchronized server monotonic wall time, not unsynchronized GPU launch time."""
    def __init__(self, *, clock=time.perf_counter, synchronize=lambda: None):
        self.clock, self.synchronize = clock, synchronize

    def call(self, function):
        self.synchronize(); start = self.clock()
        value = function()
        self.synchronize(); stop = self.clock()
        require(finite(stop-start) and stop >= start, 'invalid monotonic timing')
        return value, (stop-start)*1000


class LocalWanPositiveKV:
    """Explicit project local-window inference variant, not full-history equivalence.

    It reuses the actual CausalWan absolute-RoPE/cache-roll implementation. Every
    layer's global position remains absolute; its local index is independently
    bounded. A new native mask/numerical/peak-memory gate is still required.
    """
    branch_count, external_cfg_scale = 1, 1.

    def __init__(self, model, config, *, window_latents=7, check=lambda: None, cache_initializers=None):
        require(type(window_latents) is int and window_latents >= 7 and (window_latents-1)%3 == 0,
                'whole3-latent blocks plus immutablefirst sink required')
        backbone = model.model
        require(backbone.independent_first_frame and backbone.num_frame_per_block == 3
                and not backbone.use_relative_rope, 'absoluteRoPE independent-first causal3 model required')
        self.model, self.config, self.check, self.window = model, config, check, window_latents
        self.initializers, self.kv, self.cross = cache_initializers, None, None
        self.position = self.forward_count = self.evictions = 0
        self.pending = self.actions = self.text = None
        self.restores = []
        for target in [backbone, *(block.self_attn for block in backbone.blocks)]:
            for name, value in [('local_attn_size', window_latents), ('sink_size', 1)]:
                self.restores.append((target, name, getattr(target, name)))
                setattr(target, name, value)
            if hasattr(target, 'max_attention_size'):
                self.restores.append((target, 'max_attention_size', target.max_attention_size))
                # Exact token count set once after first latent spatial shape is known.

    def initialize(self, first, conditions):
        import torch
        require(self.kv is None and self.position == self.forward_count == 0
                and first.shape[0:3] == (1, 1, 48) and first.dtype == torch.bfloat16,
                'fresh48channel BF16 localKV session required')
        self.text = conditions.squeeze(0).to(first)
        require(self.text.ndim == 2 and self.text.shape[-1] == 4096, 'fixed positive embedding required')
        backbone = self.model.model
        self.tokens = first.shape[-2]*first.shape[-1]//(backbone.patch_size[1]*backbone.patch_size[2])
        for block in backbone.blocks: block.self_attn.max_attention_size = self.window*self.tokens
        initializers = self.initializers
        if initializers is None:
            from pipeline.causal_inference import CausalInferencePipeline
            initializers = (CausalInferencePipeline._initialize_kv_cache, CausalInferencePipeline._initialize_crossattn_cache)
        holder = SimpleNamespace(generator=self.model, local_attn_size=self.window, frame_seq_length=self.tokens,
            num_transformer_blocks=len(backbone.blocks), args=SimpleNamespace(image_or_video_shape=[1, self.window]))
        initializers[0](holder, 1, first.dtype, first.device); initializers[1](holder, 1, first.dtype, first.device)
        self.kv, self.cross = holder.kv_cache1, holder.crossattn_cache
        self._call(first, 0, 0, None); self.position = 1

    def _call(self, value, timestep, start, action):
        import torch
        self.check(); require(not torch.is_grad_enabled(), 'native localKV is inference-only')
        old_global = [int(row['global_end_index']) for row in self.kv]
        old_local = [int(row['local_end_index']) for row in self.kv]
        times = torch.as_tensor(timestep, device=value.device, dtype=torch.float32).reshape(-1, 1).expand(1, value.shape[1])
        with native_bf16_autocast(value.device):
            result = self.model.model(value.permute(0, 2, 1, 3, 4), t=times, context=[self.text], seq_len=None,
                act_context=action, act_context_scale=self.config.model.action_scale, kv_cache=self.kv,
                crossattn_cache=self.cross, current_start=start*self.tokens)
        velocity = (result[0] if isinstance(result, tuple) else result).permute(0, 2, 1, 3, 4)
        end = (start+value.shape[1])*self.tokens
        for index, row in enumerate(self.kv):
            require(int(row['global_end_index']) == end and 0 < int(row['local_end_index']) <= self.window*self.tokens,
                    'native absolute/global vs localKV bounded positions differ')
            if end > old_global[index] and old_local[index]+end-old_global[index] > self.window*self.tokens:
                self.evictions += 1
        require(velocity.shape == value.shape and velocity.dtype == value.dtype and bool(torch.isfinite(velocity).all()),
                'native localKV invalid velocity')
        self.forward_count += 1
        return velocity

    def velocity(self, *, noisy_block, history, action_block, timestep, block_index,
                 branch, cfg_scale, conditions, **unused):
        import torch
        from training.models.action_adapter import build_action_context
        require(branch == 'positive' and cfg_scale == 1. and not unused.keys()-{'sigma'}
                and self.position == 1+3*block_index == history.shape[1]
                and noisy_block.shape[1] == 3 and torch.equal(conditions.squeeze(0).to(self.text), self.text),
                'same own-history positive-only ordered3-latent block required')
        if self.pending != block_index:
            require(self.pending is None, 'interleaved localKV blocks prohibited')
            self.pending = block_index
            self.actions = build_action_context(action_block, height=self.config.data.height,
                width=self.config.data.width, device=noisy_block.device, dtype=noisy_block.dtype)
        return self._call(noisy_block, timestep, self.position, self.actions)

    def commit(self, endpoint, block_index):
        require(self.pending == block_index and self.position == 1+3*block_index, 'unsolved localKV block cannot commit')
        self._call(endpoint, 0, self.position, self.actions)
        self.position += 3; self.pending = self.actions = None

    def close(self):
        self.kv = self.cross = self.actions = self.text = None
        for target, name, value in reversed(self.restores): setattr(target, name, value)
        self.restores.clear()


class NativeCausalActionEngine:
    """One newly generated block per online snapshot; never video/cache branches."""
    def __init__(self, backend, first, conditions, *, seed=42, max_chunks=8, timer=None):
        import torch
        require(first.ndim == 5 and first.shape[0:3] == (1, 1, 48) and first.dtype == torch.bfloat16
                and bool(torch.isfinite(first).all()), 'nativeBF16 first latentB1,T1,C48 required')
        require(type(max_chunks) is int and 1 <= max_chunks <= 80 and backend.branch_count == 1
                and backend.external_cfg_scale == 1. and backend.forward_count == 0,
                'freshpositive CFG1 finite4step session required')
        if hasattr(backend, 'capacity'):
            require(1+3*max_chunks <= backend.capacity, 'full-history engine cannot pretend bounded60s support')
        self.backend, self.first, self.conditions = backend, first.detach().clone(), conditions
        self.history, self.index, self.seed, self.max_chunks = self.first, 0, seed, max_chunks
        self.timer = timer or PhaseTimer()
        self.initialized = False

    def initialize(self):
        import torch
        require(not self.initialized, 'initial latent KV commit occurs once')
        with torch.no_grad(),native_bf16_autocast(self.first.device):
            _, elapsed = self.timer.call(lambda:self.backend.initialize(self.first, self.conditions))
        require(self.backend.forward_count == 1, 'initial clean commit must be a real model forward')
        self.initialized = True
        return dict(first_commit_ms=elapsed, total_forwards=1)

    def generate_chunk(self, *, action_id, action_bits):
        import torch
        from training.completion_ode_core_v2 import fixed_noise_streams, sample_endpoint_block
        require(self.initialized and self.index < self.max_chunks, 'session is not primed or finite chunk bound reached')
        require(isinstance(action_id, str) and action_id and len(action_bits) == 8
                and all(type(bit) is int and bit in (0, 1) for bit in action_bits), 'exact latest8-key action snapshot required')
        actions = torch.tensor(action_bits, device=self.first.device, dtype=torch.float32).reshape(1,1,8).repeat(1,12,1)
        streams = fixed_noise_streams(seed=self.seed, block_index=self.index, shape=(1,3,*self.first.shape[2:]))
        before = self.backend.forward_count
        with torch.no_grad(),native_bf16_autocast(self.first.device):
            endpoint, dit_ms = self.timer.call(lambda:sample_endpoint_block(self.backend.velocity,
                initial_noise=streams.initial.to(device=self.first.device, dtype=torch.bfloat16),
                renoise=streams.renoise, history=self.history, action_block=actions,
                block_index=self.index, positive_conditions=self.conditions))
            require(self.backend.forward_count == before+4, '4steps must mean4actual positive DiT forwards')
            _, commit_ms = self.timer.call(lambda:self.backend.commit(endpoint, self.index))
        require(self.backend.forward_count == before+5 and endpoint.shape[1:3] == (3,48),
                'one actual clean commit plus3native latent positions required')
        self.history = torch.cat((self.history,endpoint.detach()), dim=1)
        row = dict(chunk_id=self.index, action_id=action_id, action_bits=list(action_bits), dit_ms=dit_ms,
            kv_commit_ms=commit_ms, denoising_forwards=4, clean_commit_forwards=1,
            noise_seeds=list(streams.seeds), history_source='own_committed_outputs', future_clean_latents_supplied=False,
            branch_count=1, external_cfg_scale=1., latents_per_chunk=3, new_frame_count=12)
        self.index += 1
        return endpoint, row

    def close(self): self.backend.close(); self.history = None


class ContinuousWanDecoder:
    """Warm up thefirst latent once, then decode3→12 without clearing caches."""
    def __init__(self, wrapper, *, height=480, width=832, timer=None):
        require(getattr(wrapper, 'z_dim', None) == 48 and callable(getattr(wrapper,'decode_to_pixel',None))
                and callable(getattr(getattr(wrapper,'model',None),'clear_cache',None)),
                'actual48channel Wan2.2 cached_decode wrapper required, not16channel checkpoint')
        self.wrapper, self.height, self.width = wrapper,height,width
        self.timer = timer or PhaseTimer()
        self.primed = self.closed = False; self.chunks = 0

    def _decode(self, latent, expected_frames):
        import torch
        require(latent.ndim == 5 and latent.shape[0] == 1 and latent.shape[2] == 48
                and bool(torch.isfinite(latent).all()), '48channel native latent tensor required')
        with torch.no_grad(),native_bf16_autocast(latent.device):
            output, milliseconds = self.timer.call(lambda:self.wrapper.decode_to_pixel(latent,use_cache=True,return_in_cpu=False))
        require(isinstance(output,torch.Tensor) and output.shape == (1,expected_frames,3,self.height,self.width)
                and bool(torch.isfinite(output).all()), 'continuous VAE returnedNone/wrong frame count/invalid pixels')
        return output,milliseconds

    def warmup(self, first):
        require(not self.primed and not self.closed and first.shape[1] == 1, 'first latent decoded once per freshsession')
        self.wrapper.model.clear_cache()
        output, milliseconds = self._decode(first,1)
        self.primed = True
        return output,dict(vae_warmup_ms=milliseconds, frames=1, excluded_from_steady_state=True)

    def decode_chunk(self, latent):
        require(self.primed and not self.closed and latent.shape[1] == 3, 'primefirst then exactly3latents per12RGB')
        before = getattr(self.wrapper.model,'_feat_map',None)
        output,milliseconds = self._decode(latent,12)
        require(before is None or getattr(self.wrapper.model,'_feat_map',None) is before,
                'VAE feature cache was replaced/reset between streaming blocks')
        self.chunks += 1
        return output,dict(vae_ms=milliseconds, new_frame_count=12, continuous_cache=True)

    def close(self):
        if not self.closed: self.wrapper.model.clear_cache()
        self.closed = True


def build_native_runtime(*, binding, ledger_path, lease, request_sha256, max_seconds,
                         first, conditions, model_loader=None, vae_loader=None, max_chunks=8,
                         local_window_latents=None, native_gate=None, check=lambda:None):
    """Called only inside a new original-guard worker, never duringCPU preparation."""
    validate_student_binding(binding,ledger_path=ledger_path)
    validate_native_gate(native_gate,binding,ledger_path=ledger_path,local_window_latents=local_window_latents)
    require(local_window_latents is not None or max_chunks<=8,
            'unprofiled full-history long stream is prohibited, use finite8chunks or an auditedlocal variant')
    from training.demo.backend import validate_lease
    validate_lease(lease,max_seconds=max_seconds,job_id=lease.get('job_id'),request_sha256=request_sha256)
    allowed=request_allowed_cards(lease)
    require(lease.get('gpu_index') in allowed, 'actual guarded completion card required')
    import os
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==lease.get('gpu_uuid'),
            'original worker must isolate the leased physicalUUID before importingCUDA')
    import torch
    require(torch.cuda.is_available(), 'actualCUDA not available; do not substitute CPU fixture as runtime result')
    check(); device=torch.device('cuda',0); torch.cuda.set_device(device)
    model,config=(model_loader(binding,device) if model_loader else load_actual_student_model(binding,device,check))
    timer=PhaseTimer(synchronize=lambda:torch.cuda.synchronize(device))
    if local_window_latents is None:
        from training.completion_ode_wan_v2 import WanSingleBranchKV
        backend=WanSingleBranchKV(model,config,capacity=1+3*max_chunks,check=check)
    else: backend=LocalWanPositiveKV(model,config,window_latents=local_window_latents,check=check)
    engine=NativeCausalActionEngine(backend,first.to(device=device,dtype=torch.bfloat16),conditions,
        seed=42,max_chunks=max_chunks,timer=timer)
    try:
        vae=vae_loader(device) if vae_loader else load_actual_wan_vae(config,device)
        decoder=ContinuousWanDecoder(vae,height=480,width=832,timer=timer)
    except BaseException:
        engine.close()
        raise
    return engine,decoder


def run_native_probe(request,directory,lease,*,check):
    """Finite4chunk genuine CUDA numerical/cache/VAE/resource probe; not run here.

    It verifies inference cache visibility by perturbing allocated uncommitted
    future K/V slots, not by supplying futureGT. This is NOT a training-mask VJP
    or visual/control test. Original worker owns reserve/watchdog/settlement.
    """
    directory=Path(directory)
    require(DATA_ROOT in directory.parents and directory.name==request.get('job_id')
        and request.get('schema')=='completion_runtime_probe_request_v2'
        and request.get('mode')=='completion_runtime_v2_probe' and request.get('probe_chunks')==4
        and _read(directory/'request.json')==request and callable(check)
        and type(request.get('max_seconds')) is int and 60<=request['max_seconds']<=1200,
        'exact finite guarded new native probe required')
    binding=request['student_binding'];validate_student_binding(binding,ledger_path=DATA_ROOT/'creator-runtime/budget.json')
    from training.demo.backend import validate_lease
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],request_sha256=file_sha256(directory/'request.json'))
    import os
    require(lease.get('gpu_index') in request_allowed_cards(lease)
            and os.environ.get('CUDA_VISIBLE_DEVICES')==lease.get('gpu_uuid'),
            'same original guarded completion UUID isolation required')
    import torch
    from training.completion_ode_jobs_v2 import _generation_inputs
    from training.completion_ode_wan_v2 import WanSingleBranchKV
    from training.completion_ode_core_v2 import fixed_noise_streams,teacher_grid
    check();require(torch.cuda.is_available(),'native CUDA required; probe cannot be CPUfixture')
    device=torch.device('cuda',0);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    model,config=load_actual_student_model(binding,device,check)
    first,positive,*unused=_generation_inputs(request['case_id'],request['baseline_binding'],'original')
    first=first.to(device=device,dtype=torch.bfloat16);positive=positive.to(device=device,dtype=torch.bfloat16)
    window=request.get('local_window_latents')
    require(window is None or window==7,'fixed full-history13 or explicit local7/sink1 probe only')
    backend=WanSingleBranchKV(model,config,capacity=13,check=check) if window is None else LocalWanPositiveKV(model,config,window_latents=7,check=check)
    timer=PhaseTimer(synchronize=lambda:torch.cuda.synchronize(device));engine=NativeCausalActionEngine(backend,first,positive,max_chunks=4,timer=timer)
    decoder=None
    try:
        decoder=ContinuousWanDecoder(load_actual_wan_vae(config,device),height=480,width=832,timer=timer)
        warmup=engine.initialize();_,vae_warmup=decoder.warmup(first)
        streams=fixed_noise_streams(seed=42,block_index=0,shape=(1,3,48,30,52))
        state=streams.initial.to(device=device,dtype=torch.bfloat16)
        actions=torch.tensor([1,0,0,0,0,1,0,0],device=device,dtype=torch.float32).reshape(1,1,8).repeat(1,12,1)
        sigma,timestep=teacher_grid().levels(state,0)
        kwargs=dict(noisy_block=state,history=first,action_block=actions,sigma=sigma,timestep=timestep,
            block_index=0,branch='positive',cfg_scale=1.,conditions=positive)
        with torch.no_grad(),native_bf16_autocast(device):
            one=backend.velocity(**kwargs)
            changed=0
            for row in backend.kv:
                for key in ('k','v'):
                    future=row[key][:,4*backend.tokens:];changed+=future.numel();future.fill_(123.)
            require(changed>0,'real uncommitted allocatedfuture KV region required for perturbation check')
            two=backend.velocity(**kwargs)
        max_error=float((one.float()-two.float()).abs().max())
        require(torch.equal(one,two),'actual inference read allocated uncommitted futureKV or is numerically unstable')
        rows=[]
        for index in range(4):
            bits=[1,0,0,0,0,1,0,0] if index%2==0 else [1,0,0,0,0,0,0,1]
            latent,dit=engine.generate_chunk(action_id=f'fixed-probe-{index}',action_bits=bits)
            _,vae=decoder.decode_chunk(latent);rows.append(dit|vae)
        peak=torch.cuda.max_memory_reserved(device)
        require(peak<23*2**30,'native DiT+KV+continuousVAE exceeds fixed23GiB resource gate')
        if window is not None:
            require(backend.evictions>0 and all(row['k'].shape[1]==7*backend.tokens
                and int(row['local_end_index'])<=7*backend.tokens for row in backend.kv),
                'actual localKV must evict and stay bounded, not rename fullhistory')
        gate=dict(schema='completion_runtime_native_gate_v2',producer_job_id=binding['producer_job_id'],
            local_window_latents=window,runtime_source_sha256=file_sha256(__file__),
            checkpoint_schema_verified=True,installed_causality_verified=True,
            causality_scope='inference_uncommitted_future_KV_perturbation_and_exact_positions_not_trainingVJP',
            future_kv_perturbed_elements=changed,future_kv_velocity_max_error=max_error,
            vae48_continuous_verified=True,decoded_new_frames=48,peak_reserved_bytes=peak,
            local_history_loss_declared=window is not None,visual_control_verified=False,realtime_verified=False)
        return dict(native_runtime_gate=gate,probe_rows=rows,warmup=warmup,vae_warmup=vae_warmup,
            actual_model_forward_calls=backend.forward_count,gpu_measurements_only_no_browser=True)
    finally:
        engine.close()
        if decoder is not None:decoder.close()
