"""Reuse a pinned real opening as media only, in a fresh story session.

The source may have later turns, but those files are never read or copied. This
is distinct from orphan recovery and from resuming committed story progress.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

from .models import WorldState


REQUIRED = {'video.mp4','input.png','last.png','actions.npy','render.json'}
ALLOWED = REQUIRED | {'inputs.mp4','contact.jpg'}
MAX_BYTES = 100 * 1024 * 1024


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _video_contract(path):
    import imageio.v2 as imageio
    with imageio.get_reader(path) as reader:
        meta=reader.get_meta_data()
        if abs(float(meta.get('fps',0))-16)>0.001 or reader.count_frames()!=97:
            raise ValueError('Opening must contain 97 actual frames at 16 fps')
        return meta


def import_initial_archive(engine,manifest_path):
    if (engine.history or engine.task is not None or engine.frame is not None or engine.video is not None
        or engine.completed_operations or engine.render_contexts or engine.pending_visual
        or engine.pending_visual_queue or engine.visual_reviews or engine.visual_warnings
        or engine.desk_inspection is not None or engine.initial_recovery is not None
        or engine.session_resume is not None or engine.state!=WorldState()
        or engine.memory.retrieve()!={'recent':[],'important':[]}):
        raise ValueError('Initial media import requires a fresh empty story session')
    if getattr(engine.renderer,'kind',None)!='wan_laboratory' or engine.renderer.seconds!=6:
        raise ValueError('Opening import requires the six-second R005 laboratory renderer')
    manifest_path=Path(manifest_path).resolve()
    raw=manifest_path.read_bytes()
    if not 0<len(raw)<=65536: raise ValueError('Invalid bounded opening manifest')
    manifest=json.loads(raw)
    session=manifest.get('source_session_id','')
    if manifest.get('schema')!='initial-import/v1' or not re.fullmatch(r'[a-f0-9]{32}',session):
        raise ValueError('A pinned initial-import/v1 source session is required')
    root=engine.root.resolve()
    source=Path(manifest['source_initial_dir']).resolve()
    if (source!=root/session/'initial' or source.parent.parent!=root
        or source.parent==engine.folder.resolve()):
        raise ValueError('Source initial media must belong to another session in this session root')
    hashes=manifest.get('files')
    if (not isinstance(hashes,dict) or not REQUIRED.issubset(hashes) or not set(hashes).issubset(ALLOWED)
        or any(not isinstance(value,str) or not re.fullmatch(r'[a-f0-9]{64}',value) for value in hashes.values())):
        raise ValueError('Manifest must pin only the allowed initial media files')
    if (manifest.get('checkpoint')!='R005-step1040' or manifest.get('seed')!=engine.renderer.seed
        or manifest.get('frames')!=97 or manifest.get('fps')!=16):
        raise ValueError('Pinned opening identity or seed does not match the fresh renderer')
    total=0
    for name,expected in hashes.items():
        path=(source/name).resolve()
        if path.parent!=source or not path.is_file(): raise ValueError('Opening media missing or outside source')
        total+=path.stat().st_size
        if total>MAX_BYTES: raise ValueError('Opening media exceeds the 100 MiB copy bound')
        if sha256(path)!=expected: raise ValueError('Opening media SHA256 mismatch: '+name)
    metadata=json.loads((source/'render.json').read_text(encoding='utf-8'))
    if (metadata.get('checkpoint')!='R005-step1040' or metadata.get('prompt_key')!='scene'
        or metadata.get('seed')!=manifest['seed'] or metadata.get('frames')!=97
        or metadata.get('fps')!=16 or metadata.get('width')!=832 or metadata.get('height')!=480):
        raise ValueError('Actual opening metadata does not match the pinned R005 opening')
    actions=np.load(source/'actions.npy',allow_pickle=False)
    if actions.shape!=(96,8) or not np.isfinite(actions).all() or np.any(actions):
        raise ValueError('Opening must have an actual all-zero 96 by 8 action schedule')
    for name in ('input.png','last.png'):
        with Image.open(source/name) as image:
            if image.mode!='RGB' or image.size!=(832,480): raise ValueError('Opening RGB dimensions differ')
            image.verify()
    video_meta=_video_contract(source/'video.mp4')
    if tuple(video_meta.get('size',()))!=(832,480): raise ValueError('Opening video dimensions differ')
    if 'inputs.mp4' in hashes: _video_contract(source/'inputs.mp4')
    destination=engine.folder/'initial'
    # The claim belongs only to the new session. Never write in the source.
    claim=engine.folder/'initial-import-claim.json'
    with claim.open('x',encoding='utf-8') as stream:
        json.dump({'manifest_sha256':hashlib.sha256(raw).hexdigest(),'source_session_id':session},stream)
    destination.mkdir(exist_ok=False)
    copied={}
    for name,expected in hashes.items():
        shutil.copyfile(source/name,destination/name)
        copied[name]=sha256(destination/name)
        if copied[name]!=expected: raise ValueError('Opening changed during bounded media copy: '+name)
    frame=str(destination/'last.png')
    observation=engine.observer.observe(frame)
    state=engine.reconciler.reconcile(engine.state,observation)
    # Reconciliation may refresh visible evidence, never imported story effects.
    logical=state.model_dump(mode='json'); logical.pop('environment')
    default=WorldState().model_dump(mode='json'); default.pop('environment')
    if logical!=default: raise ValueError('Opening observation changed logical story state')
    receipt={
        'type':'pinned_initial_media_import','source_session_id':session,
        'source_initial_dir':str(source),'source_hashes':hashes,'copied_hashes':copied,
        'manifest_path':str(manifest_path),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
        'new_session_id':engine.session_id,'preserved_new_session_id':True,
        'imported_turns':0,'applied_rule_effects':0,'new_initial_generation':False,
        'reused_historical_opening':True,'source_story_archive_read':False,'source_claim_written':False,
        'opening_checkpoint':metadata['checkpoint'],'opening_frames':97,'opening_fps':16,
        'opening_seed':metadata['seed'],'source_sampler':metadata.get('sampler',{}),
        'next_sampler':{'solver':'flow_euler','steps':40,'shift':5,'cfg':engine.renderer.cfg},
        'next_seed':engine.renderer.seed+1,'copied_bytes':total,
        'visual_success_verified':False,'later_turns':'new model requests, never inherited media or story actions'}
    engine.frame=frame
    engine.video=str(destination/('inputs.mp4' if 'inputs.mp4' in hashes else 'video.mp4'))
    engine.observation=observation; engine.state=state
    engine.renderer.counter=1
    engine.initial_recovery=receipt
    (engine.folder/'initial-import-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    engine.export_session()
    return receipt
