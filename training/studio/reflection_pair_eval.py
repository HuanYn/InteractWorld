"""Four fixed endpoint jobs, two original-view action conditions per model load."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import master_writeback_eval as master
from training.studio import reflection_pair_cache as cache

ROOT = cache.ROOT
MODE = 'reflection_pair_eval'
REVISION = 'reflection-pair-fixed-original-inputs-v1'
SOURCE_REVISION = cache.SOURCE_REVISION
STAGE = 'controlled_action_reflection_pair_native97_v1'
SCHEMA = 'controlled_action_reflection_pair_checkpoint_v1'
ARMS = ('original','reflect')
CONDITIONS = ('original','jl_swap')
JOBS = {(arm,target):f'crf-{arm}-{target.lower()}-v1' for arm in ARMS for target in ('J','L')}
read = master.read


def paths(arm):
    require(arm in ARMS, 'fixed reflection comparison arm required')
    profile = ROOT/f'crf-{arm}-profile2-v1'
    return dict(checkpoint=profile/'training/checkpoints/step-0000020.pt',config=profile/'training.yaml',
                endpoint=ROOT/f'crf-{arm}-to20-v1')


def completed_endpoint(arm):
    from training.studio import reflection_pair_train as training
    import train_reflection_pair as entry
    p = paths(arm)
    request,result,settlement = (read(p['endpoint']/(name+'.json')) for name in ('request','result','settlement'))
    training.validate_request(request,p['endpoint'])
    require(result.get('mode') == 'reflection_pair_train' and result.get('status') == 'completed'
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'actual completed settled new reflection endpoint required')
    ex = result['execution']
    lineage = ex['parent_lineage']
    entry.validate_lineage(lineage)
    history = lineage['historical_micro_batches_before_stage']
    require(ex.get('stage') == STAGE and ex.get('checkpoint_schema') == SCHEMA and ex.get('arm') == arm
            and ex.get('step') == ex.get('optimizer_internal_step') == 20
            and ex.get('micro_batches_consumed') == ex.get('new_stage_micro_batches_consumed') == ex.get('sampling_cursor') == 160
            and ex.get('historical_micro_batches_before_stage') == history
            and ex.get('lifetime_micro_batches_consumed') == history+160
            and ex.get('checkpoint_path') == str(p['checkpoint']) and p['checkpoint'].is_file(),
            'new arm20/new160 with real inherited exposure history required')
    require(request.get('arm') == arm and request.get('stop_after_step') == 20
            and request.get('parent_lineage') == lineage and request.get('config') == str(p['config'])
            and request.get('config_sha256') == sha256(p['config'])
            and lineage == entry.parent_binding(), 'unchanged completed round20 weights-only parent required')
    return request,result


def source_records(target,condition,binding=None):
    require(target in ('J','L') and condition in CONDITIONS, 'fixed source case only')
    directory = master.ROOT/master.CASES['round',target,condition]
    identity = dict(job=str(directory),**{name+'_sha256':sha256(directory/(name+'.json'))
                                         for name in ('request','result','settlement')})
    require(binding is None or binding == identity, 'actual round input source records changed')
    req,result,settlement = (read(directory/(name+'.json')) for name in ('request','result','settlement'))
    master.validate_request(req)
    require(req['arm'] == 'round' and req['target'] == target and req['condition'] == condition
            and result.get('mode') == master.MODE and result.get('revision') == master.REVISION
            and result.get('request') == req and result.get('status') == 'completed'
            and result.get('frames_written') == 97 and result.get('actual_model_forward_calls') == 80
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'fixed completed settled actual round video required')
    return req,result,identity


def load_source(target,condition,binding):
    req,result,_ = source_records(target,condition,binding)
    values = master.load_inputs(req)  # Actual old request, never a fabricated new-stage header.
    lineage = values[5]
    for key in ('selection','cache_binding_sha256','first_latent_sha256_float32','prompt_sha256_float32',
                'prompt_text','negative_source','original_action_sha256_float32','actual_action_sha256_float32',
                'original_actions','actual_actions','actual_action_condition'):
        require(lineage[key] == result['lineage'][key], 'actual unmirrored evaluation input changed: '+key)
    require(lineage['selection']['episode_id'] == cache.parent.FIXED_EPISODE
            and lineage['selection']['source_start'] == cache.parent.FIXED_STARTS[target]
            and lineage['future_clean_latents_read'] is False and lineage['future_clean_latents_supplied'] is False,
            'only same original cached z0 input allowed, never reflected or future latent')
    return req,result,values


def validate_pair_inputs(original,swapped):
    import numpy as np
    import torch
    # The augmented training pool swaps AD+JL. Evaluation deliberately swaps ONLY JL.
    for i in (0,1,4):
        require(torch.equal(original[i],swapped[i]), 'paired z0/text/negative must be exactly identical')
    require(np.array_equal(original[3],swapped[3]) and np.array_equal(
        original[2][:,[0,1,2,3,4,7,6,5]],swapped[2]), 'evaluation must retain original RGB and swap only J/L')


def fixed_fields(arm,target):
    require((arm,target) in JOBS, 'fixed original/reflect by J/L only')
    p = paths(arm)
    return dict(mode=MODE,experiment_revision=REVISION,source_revision=SOURCE_REVISION,
        job_id=JOBS[arm,target],arm=arm,target=target,conditions=list(CONDITIONS),
        checkpoint=str(p['checkpoint']),checkpoint_stage=STAGE,checkpoint_schema=SCHEMA,checkpoint_step=20,
        training_config=str(p['config']),endpoint_job=str(p['endpoint']),cleanup_seconds=60,
        seed=42,steps=40,cfg_scale=5,future_frames=96,solver='flow_euler',residency='phase',fastpath=False,
        evaluation_view='original_unmirrored',evaluation_action_swap='JL_only')


def validate_request(request):
    expected = fixed_fields(request.get('arm'),request.get('target'))
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k,v in expected.items()),
            'fixed reflection evaluation sampling contract changed')
    require(set(request) == set(expected)|{'max_seconds','checkpoint_stat','training_config_sha256',
        'endpoint_binding','source_bindings','cache_binding'}, 'unapproved evaluation override')
    require(type(request['max_seconds']) is int and 60 <= request['max_seconds'] <= 1200, 'two-case job<=1200 seconds')
    require(set(request['source_bindings']) == set(CONDITIONS), 'both existing source input records required')


def _cache_binding(result):
    return {key:result[key] for key in ('manifest','manifest_sha256','cache_index','cache_index_sha256',
                                       'receipt','receipt_sha256','visual_gate')}


def make_request(*,arm,target,source_revision,max_seconds=1200):
    request = fixed_fields(arm,target)
    require(source_revision == SOURCE_REVISION, 'fixed source revision required')
    endpoint_request,_ = completed_endpoint(arm)
    p = paths(arm)
    request.update(max_seconds=max_seconds,checkpoint_stat=master.stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'],
        endpoint_binding={name+'_sha256':sha256(p['endpoint']/(name+'.json')) for name in ('request','result','settlement')},
        source_bindings={condition:source_records(target,condition)[2] for condition in CONDITIONS},
        cache_binding=_cache_binding(cache.validate_completed_cache()))
    validate_request(request)
    return request


def prepare(*,output,arm,target,source_revision,max_seconds=1200):
    output = contained(output,ROOT)
    request = make_request(arm=arm,target=target,source_revision=source_revision,max_seconds=max_seconds)
    require(output == ROOT/request['job_id'] and not output.exists(), 'new fixed evaluation directory only')
    values = [load_source(target,condition,request['source_bindings'][condition])[2] for condition in CONDITIONS]
    validate_pair_inputs(*values)
    del values
    gc.collect()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',request=request,weights_loaded=False,
        future_clean_latents_read=False,gpu_launched=False,automatic_next_stage=False))
    return request


def _load_checkpoint(request,endpoint):
    import torch
    import train_reflection_pair as entry
    p = paths(request['arm'])
    require(master.stat_identity(p['checkpoint']) == request['checkpoint_stat']
            and sha256(p['config']) == request['training_config_sha256'], 'new checkpoint/config changed')
    config = entry.load_config(p['config'])
    hashes = entry.artifact_hashes(config,p['config'])
    payload = torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    entry.validate_checkpoint(payload,config=config,hashes=hashes,arm=request['arm'])
    entry.validate_base_identity(payload,config)
    require(payload['stage'] == STAGE and payload['checkpoint_schema'] == SCHEMA
            and payload['arm'] == request['arm'] and payload['step'] == payload['optimizer_internal_step'] == 20,
            'independently validated reflection endpoint20 required')
    ex = endpoint['execution']
    for key in ('parent_lineage','sampling_contract','method_contract','precision_contract',
                'new_stage_micro_batches_consumed','historical_micro_batches_before_stage',
                'lifetime_micro_batches_consumed','sampling_cursor'):
        require(payload[key] == ex[key], 'actual endpoint metadata differs: '+key)
    state = payload['trainable_model']
    require(all(value.dtype == (torch.float32 if name.endswith('controlled_action_gate') else torch.bfloat16)
                for name,value in state.items()), 'saved BF16 Adapter/FP32 gate export required')
    lineage = dict(stage=STAGE,checkpoint_schema=SCHEMA,arm=request['arm'],step=20,
        checkpoint=request['checkpoint'],checkpoint_stat=request['checkpoint_stat'],
        **{key:payload[key] for key in ('parent_lineage','sampling_contract','method_contract','precision_contract',
            'new_stage_micro_batches_consumed','historical_micro_batches_before_stage','lifetime_micro_batches_consumed')},
        optimizer_internal_step=20,strict_resume_only_within_new_arm=True,
        inference_uses='independently_validated_BF16_model_export',training_config=config.to_dict(),
        trained_lora=False,reference_memory_implemented=False,complete_abot_reproduction=False)
    require(master.stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'endpoint changed during validation')
    return state,entry.model_config(config),lineage


def run(request,directory,lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_lease,validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory,ROOT)
    require(directory == ROOT/request['job_id'] and read(directory/'request.json') == request
            and not any((directory/name).exists() for name in ('progress.json',*CONDITIONS)), 'new fixed prepared evaluation only')
    require(lease['gpu_index'] in (1,2), 'only original GPU1/2 authority')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],request_sha256=sha256(directory/'request.json'))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
    began,runtime,engine = time.monotonic(),None,None
    results = []
    def check(minimum_remaining=30):
        validate_active_authority(lease)
        require(request['max_seconds']-(time.monotonic()-began) >= minimum_remaining, 'finite two-case evaluation deadline')
        require(master.b.a.matched.directory_bytes(directory) < 40_000_000, 'two-case40MB limit')
    try:
        check()
        require(_cache_binding(cache.validate_completed_cache()) == request['cache_binding'], 'training data/gate changed')
        _,endpoint = completed_endpoint(request['arm'])
        p = paths(request['arm'])
        require(request['endpoint_binding'] == {name+'_sha256':sha256(p['endpoint']/(name+'.json'))
            for name in ('request','result','settlement')}, 'actual endpoint records changed')
        sources = {condition:load_source(request['target'],condition,request['source_bindings'][condition]) for condition in CONDITIONS}
        validate_pair_inputs(sources['original'][2],sources['jl_swap'][2])
        state,config,checkpoint_lineage = _load_checkpoint(request,endpoint)
        tick = time.monotonic()
        runtime = master.b.a.paired.ControlledPairRuntime(state,config,sources['original'][2][4],arm='frozen')
        state.clear(); gc.collect()
        load_seconds = time.monotonic()-tick
        engine = master.b.a.matched.CachedInitialEngine({},runtime.config,residency='phase',fastpath=False,
            _runtime=runtime,cached_first=sources['original'][2][0],
            _writer_factory=lambda *args:master.b.a.matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = load_seconds
        for condition in CONDITIONS:
            check(270)
            source_request,source_result,values = sources[condition]
            first,prompt,actions,initial,negative,source_lineage,_,_ = values
            target = directory/condition
            target.mkdir(exist_ok=False)
            inputs = dict(request=request,arm=request['arm'],target=request['target'],condition=condition,
                source_request=source_request,source_lineage=copy.deepcopy(source_lineage),
                checkpoint_lineage=checkpoint_lineage,actual_actions=actions.tolist(),seed=42,
                negative_source=source_lineage['negative_source'],evaluation_view='original_unmirrored',
                future_clean_latents_read=False,future_clean_latents_supplied=False)
            Image.fromarray(initial).save(target/'initial.png')
            np.save(target/'actions.npy',actions,allow_pickle=False)
            write_json(target/'inputs.json',inputs)
            before_calls = runtime.forward_calls
            generation,endpoint_rgb = engine.run_segment(initial=initial,prompt=prompt,actions=actions.copy(),
                seed=42,output_path=target/'raw.mp4',steps=40)
            del endpoint_rgb
            calls = runtime.forward_calls-before_calls
            master.b.validate_generation(generation,calls,source_result)
            require(runtime.forward_calls == before_calls+80, 'exact80 model forwards per action case')
            generation.update(initial_condition='same_original_round_cached_z0;RGB0_display_only',
                continuation='independent_cached_z0;no_future_ground_truth',vae_encoding_used=False)
            generation['sampler'].update(cfg='text_classifier_free',cfg_scale=5,model_forwards_per_step=2)
            row = dict(status='completed',mode=MODE,revision=REVISION,arm=request['arm'],target=request['target'],
                condition=condition,source_request=source_request,source_lineage=source_lineage,checkpoint_lineage=checkpoint_lineage,
                generation=generation,frames_written=97,raw_video=str(target/'raw.mp4'),
                actual_model_forward_calls=calls,cumulative_model_forward_calls=runtime.forward_calls,
                quality_evaluation='awaiting_actual_visual_review',quality_improvement_verified=False,
                action_correctness_verified=False,generalization_verified=False,automatic_next_stage=False)
            results.append(row)
            write_json(target/'result.json',row)
            write_json(directory/'progress.json',dict(status='running',cases=results,elapsed_seconds=time.monotonic()-began))
        require(len(results) == 2 and runtime.forward_calls == 160, 'two actual conditions/160 forwards required')
        require(torch.cuda.max_memory_reserved() < 23*1024**3, 'reserved memory below23GiB required')
        check()
        return dict(status='completed',mode=MODE,revision=REVISION,request=request,arm=request['arm'],target=request['target'],
            cases=results,actual_model_forward_calls=160,model_load_count=1,load_seconds=load_seconds,
            checkpoint_lineage=checkpoint_lineage,elapsed_seconds=time.monotonic()-began,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),future_clean_latents_supplied=False,
            quality_improvement_verified=False,action_correctness_verified=False,generalization_verified=False,
            independent_dev_evaluation=False,realtime=False,automatic_next_stage=False)
    except BaseException as error:
        write_json(directory/'failure.json',dict(status='failed',cases=results,error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-began,automatic_retry=False,
            partial_generation=copy.deepcopy(engine.last_run_report) if engine is not None else None))
        raise
    finally:
        if engine is not None: engine.close()
        elif runtime is not None: runtime.close()
