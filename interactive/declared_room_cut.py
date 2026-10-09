"""Explicit authored room cut, never navigation or a generated visual event.

Only the new advisory story-lineage entry may call this opt-in API. Asset
identity is pinned by an operator manifest and its explicit SHA, never by a browser/LLM.
"""
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from .live_continuation import rgb_sha256
from .models import Observation, RuleResult


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def story_entry_enabled(engine):
    return (getattr(engine.renderer, 'story_lineage_guard', False)
            and not getattr(engine.renderer, 'live_current_frame', False)
            and not getattr(engine.renderer, 'npc_framing', False)
            and engine.story_profile == 'spatial_v1'
            and engine.visual_review_policy == 'demo_advisory')


def load_room_manifest(path):
    source = Path(path).resolve(strict=True)
    if source.stat().st_size > 65536:
        raise ValueError('Declared room manifest exceeds the small operator manifest limit')
    expected=os.environ.get('IW_ROOM_MANIFEST_SHA256','')
    if len(expected)!=64 or file_sha256(source)!=expected:
        raise ValueError('Set IW_ROOM_MANIFEST_SHA256 to the actual operator manifest SHA256')
    manifest = json.loads(source.read_text(encoding='utf-8'))
    if (manifest.get('version') != 'declared_room_cut_v1'
            or manifest.get('enabled') is not True
            or manifest.get('story_profile') != 'spatial_v1'
            or not isinstance(manifest.get('assets'),dict)
            or set(manifest['assets']) != {'storage_room'}):
        raise ValueError('Operator room manifest must explicitly bind the supported authored storage room')
    asset=manifest['assets']['storage_room']
    if not isinstance(asset,dict) or not isinstance(asset.get('provenance'),str) or not asset['provenance'].strip():
        raise ValueError('Authored room provenance is required')
    for path_key,hash_key in (('path','sha256'),('source_path','source_sha256'),('source_receipt','source_receipt_sha256')):
        if (not isinstance(asset.get(path_key),str) or not Path(asset[path_key]).is_absolute()
                or not isinstance(asset.get(hash_key),str) or len(asset[hash_key])!=64
                or file_sha256(asset[path_key])!=asset[hash_key]):
            raise ValueError('Pinned authored room source changed: '+path_key)
    return source, manifest


def ending_observation_proposal(engine, plan):
    """Separate opt-in exception; old RuleEngine/strict-live stay unchanged."""
    if (not story_entry_enabled(engine) or not engine.state.story.terminal
            or plan.needs_clarification or plan.player_intent == 'safe_fallback'
            or plan.semantic_action != 'inspect' or plan.target not in ('left', 'right', 'room')):
        return None
    attempted = sum(bool(row.get('post_ending_observation')) for row in engine.history)
    if attempted >= 2:
        return RuleResult(valid=False, reason='结局后最多两次非剧情观察已用尽；结局和背包保持不变。')
    return RuleResult(valid=True,
        reason='结局后仅从当前画面观察；不改变剧情、背包、位置或已结束状态。')


def declared_storage_context(engine):
    context = getattr(engine.renderer, '_declared_room_context', None)
    cuts = getattr(engine, 'declared_scene_cuts', [])
    if (not story_entry_enabled(engine) or not isinstance(context, dict)
            or context.get('session_id') != engine.session_id
            or getattr(engine.renderer, '_story_session_id', None) != engine.session_id
            or context.get('room') != 'storage_room'
            or engine.state.player.location != 'storage_room'
            or not cuts or cuts[-1]['operation_id'] != context.get('cut_operation_id')
            or cuts[-1]['session_id'] != engine.session_id
            or cuts[-1]['room'] != 'storage_room'):
        return None
    current = Path(engine.frame or '').resolve()
    if not current.is_file() or not current.is_relative_to(engine.folder.resolve()):
        return None
    return context


def ending_observation_context(engine):
    if not engine.state.story.terminal or declared_storage_context(engine) is None:
        return None
    remaining = 2-sum(bool(row.get('post_ending_observation')) for row in engine.history)
    if remaining <= 0:
        return None
    return dict(version='declared_storage_observation_v1',
        permitted_calls=[['inspect', 'right', None]], remaining_attempts=remaining,
        state_effects_allowed=False, source='engine_opt_in_capability_not_visual_truth')


def declared_room_action_rejection(engine, plan):
    if (declared_storage_context(engine) is not None
            and plan.semantic_action in ('inspect', 'observe', 'wait')
            and plan.target in ('left', 'right', 'room', 'laboratory')
            and (plan.semantic_action != 'inspect' or plan.target != 'right')):
        return '门内当前仅声明右侧观察文本条件；左侧或整屋观察尚不支持，未用右转或旧实验室文本替代。'
    return None


def declare_room_scene_cut(engine, room, *, operation_id,
                           expected_current_rgb_sha256, manifest_path=None, enabled=False):
    if enabled is not True or not story_entry_enabled(engine):
        raise ValueError('Declared room cut requires an explicit advisory story-lineage opt-in')
    if not manifest_path:
        raise ValueError('An explicit operator room manifest is required')
    manifest_source, manifest = load_room_manifest(manifest_path)
    room_assets=manifest['assets']
    if not isinstance(operation_id, str) or not operation_id or len(operation_id) > 200:
        raise ValueError('A unique declared-cut operation ID is required')
    if not isinstance(expected_current_rgb_sha256, str) or len(expected_current_rgb_sha256) != 64:
        raise ValueError('The caller must bind the actual current RGB SHA')
    if not engine.lock.acquire(blocking=False):
        raise ValueError('A turn/task is in flight; declared cut was not applied')
    try:
        signature = {'room': room, 'expected_current_rgb_sha256': expected_current_rgb_sha256,
                     'manifest_sha256': file_sha256(manifest_source)}
        saved = engine.declared_scene_cut_operations.get(operation_id)
        if saved:
            if saved['signature'] != signature:
                raise ValueError('The same cut operation ID cannot identify another request')
            return dict(saved['record'], idempotent_replay=True)
        if operation_id in engine.completed_operations:
            raise ValueError('This operation ID already identifies an interaction turn')
        if (getattr(engine.renderer, '_story_request_active', False)
                or (engine.task is not None and engine.task.status in ('ready', 'running'))
                or (engine.desk_inspection is not None
                    and getattr(engine.desk_inspection, 'status', '') not in ('completed', 'cancelled', 'failed'))):
            raise ValueError('Finish or pause the current task/inspection before a declared cut')
        if engine.pending_visual or engine.pending_visual_queue:
            raise ValueError('An unresolved visual review must not be erased by a cut')
        if room not in room_assets:
            raise ValueError('No qualified exact RGB authored asset for this room; core must not reuse storage')
        if not engine.history:
            raise ValueError('A real committed move_to room turn is required before this cut')
        turn = engine.history[-1]
        if (not turn.get('rule_committed') or not turn.get('valid')
                or turn.get('renderer_status') != 'completed'
                or turn.get('plan', {}).get('semantic_action') != 'move_to'
                or turn.get('plan', {}).get('target') != room
                or engine.state.player.location != room
                or turn.get('frame') != engine.frame):
            raise ValueError('The latest completed committed turn must be move_to this room')
        if any(item['after_rule_turn_id'] == turn['turn_id'] for item in engine.declared_scene_cuts):
            raise ValueError('This room-entry turn already consumed its one declared cut')
        folder = engine.folder.resolve(strict=True)
        before = Path(engine.frame).resolve(strict=True)
        if not before.is_relative_to(folder) or rgb_sha256(before) != expected_current_rgb_sha256:
            raise ValueError('Declared cut does not match this session current frame')
        asset = room_assets[room]
        source = Path(asset['path']).resolve(strict=True)
        for path_key, hash_key in (('path', 'sha256'), ('source_path', 'source_sha256'),
                                   ('source_receipt', 'source_receipt_sha256')):
            if file_sha256(asset[path_key]) != asset[hash_key]:
                raise ValueError('Pinned authored room source changed: '+path_key)
        with Image.open(source) as image:
            image.load()
            if image.mode != 'RGB' or image.size != (832, 480):
                raise ValueError('Room art must already be exact832x480 RGB; no resize/crop allowed')
            copied = image.copy()
        before_state = engine.state.serialize()
        cut_folder = folder/'scene-cuts'/f'{len(engine.declared_scene_cuts)+1:04d}'
        cut_folder.mkdir(parents=True, exist_ok=False)
        owned = cut_folder/'current.png'
        copied.save(owned)
        if rgb_sha256(owned) != rgb_sha256(source):
            raise ValueError('Owned room RGB differs from the declared source')
        record = dict(
            kind='explicit_authored_room_scene_cut', operation_id=operation_id,
            session_id=engine.session_id, room=room, after_rule_turn_id=turn['turn_id'],
            after_rule_operation_id=turn.get('operation_id'), before_frame=str(before),
            before_frame_sha256=file_sha256(before), before_rgb_sha256=expected_current_rgb_sha256,
            before_video=engine.video, asset=dict(asset), owned_current_frame=str(owned),
            operator_manifest=str(manifest_source), operator_manifest_sha256=signature['manifest_sha256'],
            owned_current_file_sha256=file_sha256(owned), owned_current_rgb_sha256=rgb_sha256(owned),
            state_sha256=hashlib.sha256(before_state.encode()).hexdigest(),
            state_changed=False, memory_changed=False, seed_changed=False,
            world_effects={},
            new_video_generated=False, model_calls=0, gpu_calls=0,
            continuous_navigation_verified=False, threshold_crossing_verified=False,
            visual_success_verified=False, training_qualified=False,
            status='declared_authored_room_cut_not_navigation_or_visual_success')
        (cut_folder/'scene-cut.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        engine.frame = str(owned)
        engine.video = None
        engine.selected_output = None
        engine.observation = Observation(source='unavailable',
            evidence='Explicit authored room cut; no visual inference or navigation certification. Refresh from current RGB.')
        engine.renderer._story_binding = None
        engine.renderer._declared_room_context = dict(
            session_id=engine.session_id, room=room, cut_operation_id=operation_id,
            owned_initial_frame=str(owned), authored_asset_sha256=asset['sha256'],
            prompt_route='interior_right_for_inspect_right_only', visual_success_verified=False)
        engine.declared_scene_cuts.append(record)
        engine.declared_scene_cut_operations[operation_id] = {'signature': signature, 'record': record}
        return record
    finally:
        engine.lock.release()
