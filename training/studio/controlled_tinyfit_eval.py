"""Fixed four-case, z0-only evaluation of the two-window tinyfit endpoint.

Reuse matched-window input/sampler implementations without changing their
globals or accepting their old checkpoint schema as a tinyfit checkpoint.
"""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained,require,sha256,write_json
from training.demo.relocation import ArtifactHashes
from training.studio import controlled_matched_window_eval as matched
from training.studio import controlled_lora_pair_eval as paired

MODE = 'controlled_tinyfit_eval'
REVISION = 'controlled-tinyfit20-matched-original-jlswap-v1'
STAGE = 'controlled_action_tinyfit_native97_v1'
ROOT = matched.BASE/'teacher-tinyfit-v1'
BASELINE_ROOT = matched.ROOT
CASES = {(target,condition):f'ctf-{target.lower()}-{suffix}-v1'
    for target in ('WJ','WL') for condition,suffix in (('original','original'),('jl_swap','swap'))}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def baseline_job(target,condition):
    suffix = 'original' if condition == 'original' else 'swap'
    return BASELINE_ROOT/f'cmw-{target.lower()}-{suffix}-v1'


def validate_request(request):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,'tinyfit evaluation route required')
    case = (request.get('target'),request.get('condition'))
    require(case in CASES and request.get('job_id') == CASES[case],'only four fixed tinyfit cases permitted')
    require(request.get('baseline_job') == str(baseline_job(*case)),'fixed completed matched baseline required')
    require(request.get('arm') == 'frozen' and request.get('checkpoint_step') == 20
        and request.get('seed') == 42 and request.get('steps') == 40 and request.get('cfg_scale') == 5
        and request.get('future_frames') == 96 and request.get('solver') == 'flow_euler'
        and request.get('fastpath') is False,'fixed endpoint/sampling contract required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 1200,'finite <=1200s job required')
    require(isinstance(request.get('source_revision'),str) and bool(request['source_revision']),'source revision required')
    for name in ('checkpoint','training_config'):
        require(isinstance(request.get(name),str),'explicit tinyfit source required')
        contained(request[name],ROOT)
    require(Path(request['checkpoint']).name == 'step-0000020.pt','only tinyfit step20 accepted')
    for name in ('checkpoint_sha256','training_config_sha256','baseline_request_sha256','baseline_result_sha256'):
        require(isinstance(request.get(name),str) and re.fullmatch('[0-9a-f]{64}',request[name]),'exact source digest required')
    require(not any(name in request for name in ('action_scale','ratio_cap','gate_scale','lora_strength',
        'initial_latent','future_clean_latents','prompt','actions','window_index','episode_id')),'no input/scale overrides accepted')


def checkpoint_digest(request):
    protected = [request['training_config'],Path(request['baseline_job'])/'request.json',Path(request['baseline_job'])/'result.json']
    return ArtifactHashes(ROOT/'.eval-checkpoint-hashes.json',protected_paths=protected)(request['checkpoint'])


def make_request(*,job_id,source_revision,checkpoint,training_config,target,condition,max_seconds=1200):
    require((target,condition) in CASES,'fixed target/condition required')
    source = baseline_job(target,condition)
    request = dict(mode=MODE,experiment_revision=REVISION,job_id=job_id,source_revision=source_revision,
        checkpoint=str(checkpoint),training_config=str(training_config),checkpoint_step=20,
        training_config_sha256=sha256(training_config),baseline_job=str(source),
        baseline_request_sha256=sha256(source/'request.json'),baseline_result_sha256=sha256(source/'result.json'),
        target=target,condition=condition,arm='frozen',seed=42,steps=40,cfg_scale=5,future_frames=96,
        solver='flow_euler',fastpath=False,max_seconds=max_seconds)
    contained(checkpoint,ROOT)
    request['checkpoint_sha256'] = checkpoint_digest(request)
    validate_request(request)
    return request


def _load_checkpoint(request,parent_checkpoint):
    import torch
    import train_controlled_tinyfit as entry
    require(sha256(request['training_config']) == request['training_config_sha256']
        and checkpoint_digest(request) == request['checkpoint_sha256'],'tinyfit checkpoint/config changed')
    config = entry.load_config(request['training_config'])
    hashes = entry.artifact_hashes(config,request['training_config'])
    payload = torch.load(request['checkpoint'],map_location='cpu',weights_only=False)
    entry.validate_checkpoint(payload,config=config,hashes=hashes)
    entry.validate_base_identity(payload,config)
    require(payload['stage'] == STAGE and payload['step'] == 20
        and payload['micro_batches_consumed'] == payload['new_stage_micro_batches_consumed'] == 160
        and payload['historical_micro_batches_before_stage'] == 320
        and payload['lifetime_micro_batches_consumed'] == 480,'actual tinyfit stage/exposure differs')
    parent = payload['parent_lineage']
    require(parent['parent_checkpoint'] == parent_checkpoint
        and parent['parent_stage'] == paired.STAGE and parent['parent_arm'] == 'frozen'
        and parent['parent_step'] == 20,'tinyfit must inherit the exact matched baseline endpoint')
    lineage = dict(stage=STAGE,step=20,checkpoint=request['checkpoint'],checkpoint_sha256=request['checkpoint_sha256'],
        parent_lineage=parent,arm='frozen',new_stage_micro_batches_consumed=160,
        historical_micro_batches_before_stage=320,lifetime_micro_batches_consumed=480,
        checkpoint_schema=payload['checkpoint_schema'],method_contract=payload['method_contract'],
        precision_contract=payload['precision_contract'],sampling_contract=payload['sampling_contract'],
        parent_transfer='weights_only_new_optimizer_and_noise_namespace_not_strict_parent_resume',
        original_tinyfit_config=config.to_dict(),model_config_projection='data_factory_only_for_unchanged_model_constructor',
        trained_lora=False,reference_memory_implemented=False,complete_abot_reproduction=False)
    return payload['trainable_model'],entry.model_config(config),lineage


def load_inputs(request):
    validate_request(request)
    directory = Path(request['baseline_job'])
    require(sha256(directory/'request.json') == request['baseline_request_sha256']
        and sha256(directory/'result.json') == request['baseline_result_sha256'],'matched baseline records changed')
    before_request,before = read(directory/'request.json'),read(directory/'result.json')
    require(before['status'] == 'completed' and before_request['target'] == request['target']
        and before_request['condition'] == request['condition'],'matching completed baseline required')
    first,prompt,actions,initial,negative,_,lineage,_ = matched.load_inputs(before_request)
    for name in ('selection','checkpoint','checkpoint_sha256','first_latent_sha256_float32',
        'prompt_sha256_float32','negative_source','actual_action_sha256_float32','actual_actions'):
        require(lineage[name] == before['lineage'][name],'matched baseline input changed: '+name)
    state,model_config,checkpoint = _load_checkpoint(request,lineage['checkpoint'])
    lineage = copy.deepcopy(lineage)
    parent_exposure = lineage.pop('training_exposure')
    indices = list(range(0 if request['target'] == 'WJ' else 1,160,2))
    lineage.update(checkpoint=request['checkpoint'],checkpoint_sha256=request['checkpoint_sha256'],checkpoint_lineage=checkpoint,
        matched_baseline_job=str(directory),baseline_request_sha256=request['baseline_request_sha256'],
        baseline_result_sha256=request['baseline_result_sha256'],parent_training_exposure=parent_exposure,
        training_exposure=dict(new_stage_sample_indices=indices,additional_exposures=80,
            sampling='fixed_WJ_WL_alternation',new_noise_namespace=True),
        method_scope='two_window_tinyfit_learning_capacity_diagnostic_not_generalization',
        overfit_diagnostic=True,teacher_ready=False)
    return first,prompt,actions,initial,negative,state,model_config,lineage,before


def prepare(*,output,**options):
    output = contained(output,ROOT)
    require(not output.exists() and output.name == options['job_id'],'new matching tinyfit eval directory required')
    request = make_request(**options)
    values = load_inputs(request)
    lineage = values[-2]
    del values
    gc.collect()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',request=request,lineage=lineage,
        gpu_launched=False,weights_loaded_cpu_only=True,video_generated=False))
    return request


def run(request,directory,lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory,ROOT)
    require(directory.name == request['job_id'],'tinyfit eval job identity differs')
    target = directory/'matched-evaluation'
    target.mkdir(exist_ok=False)
    began = time.monotonic()
    report = dict(status='running',mode=MODE,revision=REVISION,request=request,training_window=True,
        future_clean_latents_supplied=False,independent_dev_evaluation=False,generalization_verified=False,
        quality_improvement_verified=False,action_correctness_verified=False,realtime=False,overfit_diagnostic=True)
    runtime = engine = None
    def check():
        require(time.monotonic()-began < request['max_seconds']-20,'finite tinyfit evaluation deadline reached')
        require(matched.directory_bytes(directory) < matched.JOB_BYTES_LIMIT-100_000,'tinyfit job reached20MB output cap')
        validate_active_authority(lease)
    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1','offline assets required')
        first,prompt,actions,initial,negative,state,config,lineage,before = load_inputs(request)
        report['lineage'] = lineage
        Image.fromarray(initial).save(target/'initial.png')
        np.save(target/'actions.npy',actions,allow_pickle=False)
        write_json(target/'inputs.json',dict(request=request,lineage=lineage,actual_actions=actions.tolist()))
        check()
        tick = time.monotonic()
        runtime = paired.ControlledPairRuntime(state,config,negative,arm='frozen')
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic()-tick
        engine = matched.CachedInitialEngine({},runtime.config,residency='phase',fastpath=False,_runtime=runtime,
            cached_first=first,_writer_factory=lambda *args:matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = report['load_seconds']
        generation,_ = engine.run_segment(initial=initial,prompt=prompt,actions=actions,seed=42,
            output_path=target/'raw.mp4',steps=40)
        stages = generation['stage_totals_seconds']
        require(generation['frames_written'] == 97 and runtime.forward_calls == 80
            and 'cached_z0_condition' in stages and 'encode_initial' not in stages and 'encode_endpoint' not in stages,
            'tinyfit must use cached z0 and97 frames/40CFG pairs without encoding')
        for name in ('first_latent_sha256_float32','noise_sha256_float32'):
            require(generation[name] == before['generation'][name],'before/after generation identity differs: '+name)
        generation.update(initial_condition='cached_training_z0;RGB0_display_only',
            continuation='independent_cached_z0_condition;no_future_ground_truth',vae_encoding_used=False)
        generation['sampler'].update(cfg='text_classifier_free',cfg_scale=5,model_forwards_per_step=2)
        report.update(status='completed',frames_written=97,generation=generation,raw_video=str(target/'raw.mp4'),
            actual_model_forward_calls=runtime.forward_calls,peak_allocated_bytes=generation['peak_vram_bytes'],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),quality_evaluation='awaiting_visual_review')
        check()
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if engine is not None: engine.close()
            elif runtime is not None: runtime.close()
        finally:
            report['elapsed_seconds'] = time.monotonic()-began
            write_json(target/'result.json',report)
    return report
