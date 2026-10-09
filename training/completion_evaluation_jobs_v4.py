"""Actual continuous sixty-second evaluation, not prerecorded frame playback.

Six finite batches retain all18 prescribed clips. The fixed1800s ceiling is
an operating limit, not a measured speed claim; a failure preserves partial
outputs and stops the owning queue. Full241-latent history is CPU-layer parked
without eviction, explicitly a slow memory fallback and not an acceleration.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

MODE='completion_evaluation_v4_60s'
REVISION='creator-completion-deadline-v4'
ROOT=Path('/DATA_DISK/InteractWorld/creator-runtime/abot-completion-20261007/paired60s-evaluation-v2')
LEDGER=ROOT.parent.parent/'budget.json'
SEEDS=(42,43,44)
CASES=('pair-j','holdout-l')
MODELS=('ode80','short32','long32')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def jobs(producers,*,gpu_indices=(0,1,2)):
    """Exact final producer identities, not fabricated future checkpoint pins."""
    require(set(producers)==set(MODELS) and len(set(gpu_indices))==3,'three actual final producers/cards required')
    rows=[]
    for model,card in zip(MODELS,gpu_indices):
        for case in CASES:
            identity=f'd30-eval60-{model}-{case}-v4'
            rows.append(dict(job_id=identity,mode=MODE,group='evaluation',gpu_index=card,
                max_seconds=1800,cleanup_seconds=60,source_revision=REVISION,
                directory=str(ROOT/identity),depends_on=[producers[model]],
                arguments=dict(model_key=model,case_id=case,producer_job_id=producers[model],
                    seeds=list(SEEDS),blocks=80,frames=961,action_policy='repeat_original96_ten_times')))
    return rows


def periodic_actions(original):
    import numpy as np
    original=np.asarray(original)
    require(original.shape==(96,8) and np.isfinite(original).all()
        and np.isin(original,[0,1]).all(),'exact original96x8 binary action sequence required')
    return np.tile(original,(10,1)).astype(np.float32)


def _contract(spec,plan):
    require(spec.get('mode')==MODE and spec.get('group')=='evaluation'
        and spec.get('source_revision')==REVISION and spec.get('max_seconds')==1800
        and spec.get('cleanup_seconds')==60,'exact finite sixty-second batch required')
    args=spec.get('arguments',{})
    require(args.get('model_key') in MODELS and args.get('case_id') in CASES
        and args.get('seeds')==list(SEEDS) and args.get('blocks')==80 and args.get('frames')==961
        and args.get('action_policy')=='repeat_original96_ten_times'
        and args.get('producer_job_id') in spec.get('depends_on',[]),'full18-clip fixed protocol required')
    rows={row['job_id']:row for row in plan['jobs']}
    row=rows.get(spec.get('job_id'))
    require(row is not None and all(spec.get(k)==v for k,v in row.items()),'immutable evaluation node changed')
    producer=rows[args['producer_job_id']]
    if args['model_key']=='ode80':
        require(producer['mode']=='completion_ode_v2_train'
            and producer['arguments'].get('stop_after_step')==80,'actual finalODE80 producer required')
    else:
        require(producer['mode']=='completion_dmd_v2_train'
            and producer['arguments'].get('stop_after_step')==32
            and producer['arguments'].get('arm')==args['model_key'][:-2],'actual same-arm finalDMD32 required')
    require(Path(spec['directory'])==ROOT/spec['job_id'],'exact evaluation output root required')
    return producer


def prepare_job(spec,directory):
    """CPU only: actual completed producer and z0-only baseline before reserve."""
    from training import completion_parallel_v4 as bridge
    from training.completion_runtime_v2 import student_binding_from_job
    from training.studio import balanced_lora_pair_eval as baseline
    plan=bridge.manifest(spec['parallel_budget_binding']);producer=_contract(spec,plan)
    directory=Path(directory)
    require(directory.is_dir() and not (directory/'request.json').exists(),'new CPU request only')
    binding=student_binding_from_job(producer['directory'],ledger_path=LEDGER)
    request=copy.deepcopy(spec)
    request.update(schema='completion_evaluation_request_v4',contract_max_seconds=1800,
        derived_timeout=dict(schema='creator_measured_timeout_v1',kind='fixed',max_seconds=1800),
        student_binding=binding,baseline_binding=baseline.baseline_records(spec['arguments']['case_id'])[2],
        evaluation_source_sha256=sha(__file__),native_long_profile_completed=False,
        original_actions_are_periodic_not_counterfactual_GT=True)
    write(directory/'request.json',request)
    return directory/'request.json'


def decode60(vae,latents,initial,*,device,check,writer):
    """One uninterrupted VAE feature cache, one actual initial RGB+960 new RGB."""
    import torch
    from training.completion_dmd_native_v2 import native_amp
    from training.studio.completion_causal_eval import close_video_preserving
    require(tuple(latents.shape)==(1,241,48,30,52),'complete own-history241 latent sequence required')
    require(initial.shape==(480,832,3),'exact initial display RGB required')
    vae.model.clear_cache();written=0;began=time.monotonic();finalized=False
    try:
        for start,stop,count in [(0,1,1),*[(1+3*i,4+3*i,12) for i in range(80)]]:
            check()
            with torch.no_grad(),native_amp(device):
                decoded=vae.decode_to_pixel(latents[:,start:stop].to(device,torch.bfloat16),
                    use_cache=True,return_in_cpu=True)
            require(tuple(decoded.shape)==(1,count,3,480,832) and bool(torch.isfinite(decoded).all()),
                'continuous48-channel VAE shape/finite mismatch')
            rgb=decoded[0].permute(0,2,3,1).add(1).mul(127.5).round().clamp(0,255).byte().numpy()
            if start==0:rgb[0]=initial
            writer.write(rgb);written+=count
        close_video_preserving(writer);finalized=True
        require(written==961,'exact961 actual decoded frames required')
        return dict(frames_written=written,decode_seconds=time.monotonic()-began,
            vae_continuous_decode=True,history_reset_between_blocks=False,exact_initial_RGB_display_only=True)
    finally:
        vae.model.clear_cache()
        if not finalized:
            try:close_video_preserving(writer)
            except Exception:pass


def run_job(request,directory,lease):
    import numpy as np
    import torch
    from training import completion_parallel_v4 as bridge
    from training.completion_runtime_v2 import load_actual_student_model,load_actual_wan_vae,validate_student_binding
    from training.completion_ode_jobs_v2 import _generation_inputs
    from training.completion_ode_core_v2 import sample_causal_clip
    from training.completion_dmd_native_v2 import PositiveHistory,native_amp
    from training.demo.backend import validate_lease,validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    directory=Path(directory)
    require(read(directory/'request.json')==request and request.get('schema')=='completion_evaluation_request_v4'
        and request.get('evaluation_source_sha256')==sha(__file__),'prepared source/request changed')
    plan=bridge.manifest(request['parallel_budget_binding'])
    spec={k:request[k] for k in next(row for row in plan['jobs'] if row['job_id']==request['job_id'])}
    _contract(spec,plan)
    validate_lease(lease,max_seconds=1800,job_id=request['job_id'],request_sha256=sha(directory/'request.json'))
    require(lease['gpu_index']==request['gpu_index'] in plan['allowed_gpu_indices']
        and os.environ.get('CUDA_VISIBLE_DEVICES')==lease['gpu_uuid']
        and os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1',
        'original shared lease/offline physical UUID isolation required')
    bridge.reconcile(request['parallel_budget_binding'],ledger=read(LEDGER),lease=lease,max_seconds=1800)
    validate_student_binding(request['student_binding'],ledger_path=LEDGER)
    require(torch.cuda.is_available(),'actual CUDA required; no fixture evaluation claim')
    began=time.monotonic()
    def check():
        require(time.monotonic()-began<1770,'finite full-batch watchdog reached; no silent sample removal')
        validate_active_authority(lease)
    device=torch.device('cuda',0);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    model,config=load_actual_student_model(request['student_binding'],device,check)
    model.requires_grad_(False).eval()
    values=_generation_inputs(request['arguments']['case_id'],request['baseline_binding'],'original')
    first,positive,original,initial,negative,lineage=values
    first=first.to(device,torch.bfloat16);positive=positive.to(device,torch.bfloat16)
    actual=periodic_actions(original);actions=torch.as_tensor(actual).unsqueeze(0).to(device)
    np.save(directory/'actions.npy',actual,allow_pickle=False)
    write(directory/'inputs.json',dict(source_input_lineage=lineage,actual_actions=actual.tolist(),
        frames=961,fps=16,blocks=80,seeds=list(SEEDS),condition='periodic_original',
        future_clean_latents_read=False,future_clean_latents_supplied=False,
        development_cases_reused=True,independent_final_holdout=False))
    generated=[];load_seconds=time.monotonic()-began
    for seed in SEEDS:
        check();target=directory/f'seed-{seed}';target.mkdir(exist_ok=False)
        backend=PositiveHistory(model,config,capacity=241,check=check,cpu_layer_offload=True)
        tick=time.monotonic()
        try:
            with torch.no_grad(),native_amp(device):
                sample=sample_causal_clip(backend,first=first,actions=actions,
                    positive_conditions=positive,seed=seed,blocks=80)
            latents=sample.pop('latents').cpu()
            require(sample['total_forwards']==401 and backend.position==241,'full80-block actual causal rollout required')
            generation=dict(sample,generation_seconds=time.monotonic()-tick,
                cpu_layer_KV_offload=True,CPU_KV_transfer_bytes=backend.host_transfer_bytes,
                CPU_KV_transfer_seconds=backend.host_transfer_seconds,slow_resource_fallback_not_acceleration=True)
            write(target/'latent-generation.json',generation);generated.append((seed,latents,generation))
        finally:backend.close()
    model.to('cpu');del model;gc.collect();torch.cuda.empty_cache();check()
    vae=load_actual_wan_vae(config,device);rows=[]
    try:
        for seed,latents,generation in generated:
            target=directory/f'seed-{seed}';writer=ffmpeg_writer_factory(target/'raw.mp4',832,480,16)
            decoded=decode60(vae,latents,initial,device=device,check=check,writer=writer)
            row=dict(seed=seed,model_key=request['arguments']['model_key'],case_id=request['arguments']['case_id'],
                raw_video=str(target/'raw.mp4'),generation=generation,**decoded,
                quality_verified=False,action_correctness_verified=False,realtime_verified=False)
            write(target/'result.json',row);rows.append(row)
        peak=torch.cuda.max_memory_reserved(device)
        require(peak<23*2**30,'true long rollout/decode exceeds fixed23GiB gate')
        return dict(status='completed',execution=dict(stage='completion_continuous60s_evaluation_v4',
            clips=rows,clips_completed=3,total_frames_written=2883,model_load_seconds=load_seconds,
            peak_reserved_bytes=peak,sampling_steps=4,external_cfg_scale=1.,
            phase_order='three_real80block_rollouts_then_release_model_then_continuous_decode',
            training_or_history_reset=False,future_clean_latents_supplied=False,
            quality_evaluation='awaiting_real_video_review',quality_verified=False,realtime_verified=False),
            elapsed_seconds=time.monotonic()-began)
    finally:
        vae.to('cpu');del vae;gc.collect();torch.cuda.empty_cache()
