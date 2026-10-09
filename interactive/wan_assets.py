"""Relocate immutable legacy assets without rewriting checkpoints or receipts.

The old loader deliberately binds absolute training paths. This adapter verifies
the same original bytes/lineage and only then changes the runtime model location.
No dataset/video scan; only the assets required by this single inference contract.
"""
from dataclasses import replace
import gc
import json
import os
from pathlib import Path


def runtime_path(path, snapshot):
    original=Path(path)
    original_root=os.environ.get('IW_ASSET_ORIGINAL_ROOT')
    if not original_root or not Path(original_root).is_absolute():
        raise ValueError('IW_ASSET_ORIGINAL_ROOT must name the actual absolute source root recorded by your assets')
    old=Path(original_root)
    root=Path(snapshot).resolve()
    if original.is_relative_to(old):
        resolved=(root/original.relative_to(old)).resolve()
        if not resolved.is_relative_to(root): raise ValueError('Path escapes migration snapshot')
        return resolved
    resolved=original.resolve()
    if not resolved.is_relative_to(root): raise ValueError('Asset outside operator-selected snapshot')
    return resolved


def load_portable_action_inputs(rollout_path,snapshot):
    import torch
    import yaml
    from training.config import load_config
    from training.demo.action_backend import validate_action_config,compare_saved_config,tensor_sha,VAE_SHA256
    from training.demo.contracts import sha256,require,ACTION_STAGE
    raw=yaml.safe_load(Path(rollout_path).read_text())
    validate_action_config(raw); spec=raw['lineage']; scene=raw['scenes'][0]
    resolve=lambda value:runtime_path(value,snapshot)
    checkpoint=resolve(spec['checkpoint_path'])
    require(sha256(checkpoint)==spec['checkpoint_sha256'],'Checkpoint differs from frozen showcase')
    artifacts={k:resolve(v) for k,v in spec['artifact_paths'].items()}
    config=load_config(artifacts['training_config'])
    # Preserve configuration equality, including original absolute paths.
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(payload.get('format_version')==1 and payload.get('stage')==ACTION_STAGE,'Unexpected checkpoint stage')
    compare_saved_config(payload.get('config'),config.to_dict(),payload['step'])
    hashes={k:sha256(v) for k,v in artifacts.items()}
    require(hashes==payload.get('manifest_hashes'),'Migrated artifact bytes differ from checkpoint lineage')
    require(spec['expected_base_model_path']==config.model.base_model_path,'Original model lineage differs')
    require(resolve(config.data.manifest_path)==artifacts['dataset_manifest'],'Manifest relocation differs')
    require(resolve(config.data.prompt_cache_path)==artifacts['prompt_cache'],'Prompt-cache relocation differs')
    feature=json.loads(artifacts['feature_receipt'].read_text())
    require(feature.get('schema_version')==1,'Unsupported feature receipt')
    require(resolve(feature['index'])==artifacts['feature_index'] and resolve(feature['manifest'])==artifacts['dataset_manifest'],'Feature path binding differs')
    require(feature['index_sha256']==hashes['feature_index'] and feature['manifest_sha256']==hashes['dataset_manifest'],'Feature bytes binding differs')
    receipt=json.loads(artifacts['prompt_cache_receipt'].read_text())
    require(receipt.get('prompt_policy')=='scene_static_only_v1','Unexpected text policy')
    require(resolve(receipt['cache_path'])==artifacts['prompt_cache'],'Prompt receipt path binding differs')
    for receipt_key,artifact_key in [('cache_sha256','prompt_cache'),('manifest_sha256','dataset_manifest'),
        ('feature_index_sha256','feature_index'),('feature_receipt_sha256','feature_receipt')]:
        require(receipt[receipt_key]==hashes[artifact_key],'Prompt source binding differs')
    binding=receipt['episodes'].get(scene['source_episode_id'])
    require(binding and binding['split']=='dev' and binding['prompt']==scene['prompt'],'Scene text not bound to selected embedding')
    state=payload['trainable_model']
    require(any('.lora_a.' in k for k in state) and any('act_control_adapter.' in k for k in state),'Missing trained action parameters')
    lineage={'stage':payload['stage'],'step':payload['step'],'original_config':payload['config'],
        'checkpoint':str(checkpoint),'artifact_binding_verified':True,'runtime_path_relocation_only':True}
    del payload; gc.collect()
    cache=torch.load(artifacts['prompt_cache'],map_location='cpu',weights_only=True)
    require(cache.get('schema_version')==1 and cache.get('kind')=='scene_static_prompt_cache','Unexpected prompt tensor cache')
    prompt=cache['prompt_embeds'][scene['source_episode_id']]
    require(prompt.ndim==2 and 1<=prompt.shape[0]<=512 and prompt.shape[1]==4096 and bool(torch.isfinite(prompt).all()),'Invalid text embedding')
    prompt=prompt.to(dtype=torch.bfloat16).clone(); del cache; gc.collect()
    model_dir=resolve(config.model.base_model_path)
    vae=model_dir/'Wan2.2_VAE.pth'
    require(sha256(vae)==VAE_SHA256,'Migrated VAE differs from original inference VAE')
    runtime_model=replace(config.model,base_model_path=str(model_dir))
    lineage.update(runtime_model_path=str(model_dir),prompt_shape=list(prompt.shape),prompt_tensor_id=tensor_sha(prompt))
    return state,runtime_model,prompt,lineage
