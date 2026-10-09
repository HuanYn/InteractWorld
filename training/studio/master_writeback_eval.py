"""Independent master-writeback endpoints on the exact saved B evaluation inputs."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import paired_turn_b_eval as b

MODE = 'master_writeback_eval'
REVISION = 'master-writeback-pair-fixed-b-inputs-v1'
SOURCE_REVISION = 'creator-master-writeback-pair-v1'
STAGE = 'controlled_action_master_writeback_native97_v1'
SCHEMA = 'controlled_action_master_writeback_checkpoint_v1'
ROOT = b.a.matched.BASE / 'teacher-master-writeback-v1'
ARMS = ('round', 'retain')
CASES = {(arm, target, condition):f'cmaster-{arm}-{target.lower()}-{suffix}-v1'
    for arm in ARMS for target in ('J', 'L') for condition, suffix in (('original','original'), ('jl_swap','swap'))}
read = b.read


def paths(arm):
    require(arm in ARMS, 'explicit round or retain arm required')
    profile = ROOT / f'cmaster-{arm}-profile2-v1'
    return dict(checkpoint=profile / 'training/checkpoints/step-0000020.pt',
                config=profile / 'training.yaml', endpoint=ROOT / f'cmaster-{arm}-to20-v1')


def stat_identity(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def completed_endpoint(arm):
    from training.studio import master_writeback_train as training
    p = paths(arm)
    request, result, settlement = (read(p['endpoint'] / (name+'.json')) for name in ('request','result','settlement'))
    training.validate_request(request, p['endpoint'])
    require(result.get('mode') == 'master_writeback_train' and result.get('status') == 'completed'
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'actual completed and settled new arm endpoint required')
    execution = result['execution']
    require(execution.get('stage') == STAGE and execution.get('checkpoint_schema') == SCHEMA
            and execution.get('arm') == arm and execution.get('step') == 20 and execution.get('optimizer_internal_step') == 40
            and execution.get('checkpoint_path') == str(p['checkpoint'])
            and execution.get('micro_batches_consumed') == execution.get('new_stage_micro_batches_consumed') == 160
            and execution.get('historical_micro_batches_before_stage') == 480
            and execution.get('lifetime_micro_batches_consumed') == 640 and execution.get('sampling_cursor') == 320,
            'actual master migration step20/new160/history480 endpoint required')
    require(request.get('arm') == arm and request.get('stop_after_step') == 20
            and request.get('parent_lineage') == execution.get('parent_lineage')
            and request.get('config') == str(p['config']) and request.get('config_sha256') == sha256(p['config'])
            and p['checkpoint'].is_file(), 'new arm config/checkpoint missing or mismatched')
    return request, result


def validate_request(request):
    case = (request.get('arm'), request.get('target'), request.get('condition'))
    require(case in CASES and request.get('job_id') == CASES[case], 'only eight fixed master endpoints/cases permitted')
    p = paths(case[0])
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        checkpoint=str(p['checkpoint']), checkpoint_stage=STAGE, checkpoint_schema=SCHEMA, checkpoint_step=20,
        training_config=str(p['config']), endpoint_job=str(p['endpoint']),
        baseline_job=str(b.ROOT / b.CASES[case[1:]]), model_arm='frozen',
        seed=42, steps=40, cfg_scale=5, future_frames=96, solver='flow_euler', fastpath=False,cleanup_seconds=60)
    require(all(request.get(k) == v for k,v in expected.items()), 'new schema/source/B-input sampling contract changed')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 1200, 'bounded <=1200-second eval required')
    require(set(request) == set(expected) | {'job_id','arm','target','condition','max_seconds','checkpoint_stat',
        'training_config_sha256','baseline_request_sha256','baseline_result_sha256','baseline_settlement_sha256',
        'endpoint_request_sha256','endpoint_result_sha256','endpoint_settlement_sha256'}, 'unapproved evaluation override')
    identity = request.get('checkpoint_stat', {})
    require(set(identity) == {'bytes','mtime_ns'} and all(type(v) is int and v > 0 for v in identity.values()), 'actual new checkpoint stat required')
    for key, value in request.items():
        if key.endswith('_sha256'):
            require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}', value), 'small source record binding required')


def make_request(*, arm, target, condition, source_revision, max_seconds=1200):
    require((arm,target,condition) in CASES, 'fixed original or JL-only swap required')
    p = paths(arm)
    endpoint_request, _ = completed_endpoint(arm)
    baseline = b.ROOT / b.CASES[target,condition]
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision,
        job_id=CASES[arm,target,condition], arm=arm, target=target, condition=condition,
        checkpoint=str(p['checkpoint']), checkpoint_stage=STAGE, checkpoint_schema=SCHEMA, checkpoint_step=20,
        checkpoint_stat=stat_identity(p['checkpoint']), training_config=str(p['config']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_job=str(p['endpoint']),
        baseline_job=str(baseline), model_arm='frozen', seed=42, steps=40, cfg_scale=5,
        future_frames=96, solver='flow_euler', fastpath=False, max_seconds=max_seconds,cleanup_seconds=60)
    for prefix, directory in (('baseline',baseline), ('endpoint',p['endpoint'])):
        for name in ('request','result','settlement'):
            request[prefix+'_'+name+'_sha256'] = sha256(directory / (name+'.json'))
    validate_request(request)
    return request


def load_inputs(request):
    validate_request(request)
    p = paths(request['arm'])
    _, endpoint = completed_endpoint(request['arm'])
    baseline = Path(request['baseline_job'])
    for prefix, directory in (('baseline',baseline), ('endpoint',p['endpoint'])):
        for name in ('request','result','settlement'):
            require(sha256(directory / (name+'.json')) == request[prefix+'_'+name+'_sha256'], 'completed source record changed')
    before_request, before, settlement = (read(baseline / (name+'.json')) for name in ('request','result','settlement'))
    b.validate_request(before_request)
    require(before_request['target'] == request['target'] and before_request['condition'] == request['condition']
            and before.get('request') == before_request and before.get('mode') == b.MODE
            and before.get('revision') == b.REVISION and before.get('status') == 'completed'
            and before.get('frames_written') == 97 and before.get('actual_model_forward_calls') == 80
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed', 'actual matching B case required')
    # Original loader is used as itself, with its own actual B request, never a fake new-schema request.
    first,prompt,actions,initial,negative,original,_,_ = b.load_inputs(before_request)
    for name in ('selection','cache_binding_sha256','first_latent_sha256_float32','prompt_sha256_float32',
        'prompt_text','negative_source','original_action_sha256_float32','actual_action_sha256_float32',
        'original_actions','actual_actions','actual_action_condition'):
        require(original[name] == before['lineage'][name], 'actual B input changed: '+name)
    require(original['future_clean_latents_read'] is False and original['future_clean_latents_supplied'] is False,
            'only cached initial latent can condition generation')
    lineage = copy.deepcopy(original)
    lineage.pop('checkpoint_sha256',None)  # Never relabel B's weight digest as the new endpoint's.
    lineage.pop('checkpoint_lineage',None)
    lineage.update(arm=request['arm'], checkpoint=request['checkpoint'], checkpoint_stat=request['checkpoint_stat'],
        B_checkpoint=before_request['checkpoint'], B_checkpoint_sha256=before_request['checkpoint_sha256'],
        B_baseline_job=str(baseline), B_noise_sha256_float32=before['generation']['noise_sha256_float32'],
        B_training_exposure=original['training_exposure'],
        training_exposure=dict(new_stage_sample_indices=list(range(0 if request['target']=='J' else 1,160,2)),
            source_sample_indices=list(range(160 if request['target']=='J' else 161,320,2)),
            additional_exposures=80, historical_micro_batches_before_stage=480,lifetime_micro_batches_consumed=640),
        method_scope='FP32_master_retention_vs_explicit_per_step_projection;same_BF16_forward_and_8bit_optimizer',
        precision_migration_not_old_B_strict_resume=True, automatic_further_training=False,
        teacher_ready=False, independent_dev_evaluation=False,generalization_verified=False)
    return first,prompt,actions,initial,negative,lineage,before,endpoint


def _load_checkpoint(request, endpoint):
    import torch
    import train_master_writeback as entry
    p = paths(request['arm'])
    require(sha256(p['config']) == request['training_config_sha256']
            and stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'new checkpoint/config changed')
    config = entry.load_config(p['config'])
    hashes = entry.artifact_hashes(config,p['config'])
    payload = torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    entry.validate_checkpoint(payload,config=config,hashes=hashes,arm=request['arm'])
    entry.validate_base_identity(payload,config)
    require(payload['checkpoint_schema'] == SCHEMA and payload['stage'] == STAGE
            and payload['arm'] == request['arm'] and payload['step'] == 20
            and payload['micro_batches_consumed'] == payload['new_stage_micro_batches_consumed'] == 160
            and payload['historical_micro_batches_before_stage'] == 480
            and payload['lifetime_micro_batches_consumed'] == 640 and payload['sampling_cursor'] == 320,
            'only independently validated master migration endpoint allowed')
    for key in ('parent_lineage','sampling_contract'):
        require(payload[key] == endpoint['execution'][key], 'new endpoint provenance differs: '+key)
    state = payload['trainable_model']
    require(all(value.dtype == (torch.float32 if name.endswith('controlled_action_gate') else torch.bfloat16)
                for name,value in state.items()), 'inference must use saved BF16 Adapter and FP32 gate export')
    lineage = dict(stage=STAGE,checkpoint_schema=SCHEMA,arm=request['arm'],step=20,
        checkpoint=request['checkpoint'],checkpoint_stat=request['checkpoint_stat'],
        parent_lineage=payload['parent_lineage'],sampling_contract=payload['sampling_contract'],
        method_contract=payload['method_contract'],precision_contract=payload['precision_contract'],
        master_mapping=payload['master_mapping'],optimizer_internal_step=40,new_stage_micro_batches_consumed=160,
        historical_micro_batches_before_stage=480,lifetime_micro_batches_consumed=640,sampling_cursor=320,
        inference_uses='validated_BF16_model_export_not_direct_FP32_master',
        original_training_config=config.to_dict(),strict_resume_only_within_new_arm=True,
        reference_memory_implemented=False,complete_abot_reproduction=False)
    require(stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'checkpoint changed during validation')
    return state,entry.model_config(config),lineage


def prepare(*,output,arm,target,condition,source_revision,max_seconds=1200):
    output = contained(output,ROOT)
    request = make_request(arm=arm,target=target,condition=condition,source_revision=source_revision,max_seconds=max_seconds)
    require(output == ROOT/request['job_id'] and not output.exists(), 'new fixed arm evaluation directory required')
    values = load_inputs(request)
    lineage = values[5]
    del values
    gc.collect()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',request=request,lineage=lineage,
        weights_loaded=False,gpu_launched=False,automatic_further_training=False))
    return request


def run(request,directory,lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory,ROOT)
    require(directory == ROOT/request['job_id'], 'fixed arm job directory required')
    target = directory/'matched-evaluation'
    target.mkdir(exist_ok=False)
    began,runtime,engine = time.monotonic(),None,None
    report = dict(status='running',mode=MODE,revision=REVISION,request=request,arm=request['arm'],
        future_clean_latents_supplied=False,independent_dev_evaluation=False,generalization_verified=False,
        quality_improvement_verified=False,action_correctness_verified=False,realtime=False,automatic_further_training=False)
    def check():
        require(time.monotonic()-began < request['max_seconds']-20, 'finite master evaluation deadline reached')
        require(b.a.matched.directory_bytes(directory) < b.a.matched.JOB_BYTES_LIMIT-100_000, 'job20MB cap reached')
        validate_active_authority(lease)
    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
        first,prompt,actions,initial,negative,lineage,before,endpoint = load_inputs(request)
        state,config,checkpoint_lineage = _load_checkpoint(request,endpoint)
        lineage['checkpoint_lineage'] = checkpoint_lineage
        report['lineage'] = lineage
        Image.fromarray(initial).save(target/'initial.png')
        np.save(target/'actions.npy',actions,allow_pickle=False)
        write_json(target/'inputs.json',dict(request=request,lineage=lineage,actual_actions=actions.tolist()))
        tick = time.monotonic()
        runtime = b.a.paired.ControlledPairRuntime(state,config,negative,arm='frozen')
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic()-tick
        engine = b.a.matched.CachedInitialEngine({},runtime.config,residency='phase',fastpath=False,_runtime=runtime,
            cached_first=first,_writer_factory=lambda *args:b.a.matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = report['load_seconds']
        generation,_ = engine.run_segment(initial=initial,prompt=prompt,actions=actions,seed=42,
            output_path=target/'raw.mp4',steps=40)
        b.validate_generation(generation,runtime.forward_calls,before)
        generation.update(initial_condition='same_B_cached_z0;RGB0_display_only',
            continuation='independent_cached_z0_condition;no_future_ground_truth',vae_encoding_used=False)
        generation['sampler'].update(cfg='text_classifier_free',cfg_scale=5,model_forwards_per_step=2)
        report.update(status='completed',frames_written=97,generation=generation,raw_video=str(target/'raw.mp4'),
            actual_model_forward_calls=runtime.forward_calls,peak_allocated_bytes=generation['peak_vram_bytes'],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),quality_evaluation='awaiting_actual_visual_review')
        check()
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
            elif runtime is not None:
                runtime.close()
        finally:
            report['elapsed_seconds'] = time.monotonic()-began
            write_json(target/'result.json',report)
    return report
