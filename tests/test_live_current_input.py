"""CPU current-input lifecycle only; fixture pixels are not visual evidence."""
import json
from pathlib import Path

from PIL import Image
import pytest

from interactive.engine import InteractionEngine
from interactive.live_continuation import request_evidence,rgb_sha256
from interactive.models import Observation,WorldState
from interactive.renderers import LaboratoryWanRenderer


class ObserveOnly:
    def __init__(self):self.paths=[]
    def observe(self,path):
        self.paths.append(path)
        return Observation(source='vlm',scene='storage room',visible_objects=['red_key'],
                           major_event='fixture claimed key found',confidence=0.5)


def setup(tmp_path):
    image=tmp_path/'declared-current.png'
    Image.new('RGB',(832,480),'#293e51').save(image)
    renderer=LaboratoryWanRenderer('http://never-called.invalid',image,seconds=3,
        live_current_frame=True,live_prompt_key='archive_neutral',live_full_window_pan=True)
    observer=ObserveOnly()
    engine=InteractionEngine(tmp_path/'sessions',renderer=renderer,observer=observer)
    return engine,renderer,observer,image


def test_current_input_never_calls_renderer_or_increments_seed(tmp_path,monkeypatch):
    engine,renderer,observer,image=setup(tmp_path)
    renderer.counter=7;old_session=engine.session_id
    def forbidden(*a,**kw):raise AssertionError('No video generation is allowed during initialization')
    monkeypatch.setattr(renderer,'generate',forbidden)
    monkeypatch.setattr('interactive.renderers.requests.post',forbidden)
    result=engine.start_from_current_frame(image)
    assert engine.session_id!=old_session and renderer.counter==7
    owned=engine.folder/'initial-input/current.png'
    assert Path(engine.frame)==owned and rgb_sha256(owned)==rgb_sha256(image)
    assert result['video'] is None and engine.video is None and observer.paths==[str(owned)]
    assert not list(engine.folder.rglob('*.mp4'))
    assert renderer._live_input_initialized and not renderer._live_has_generated
    assert result['renderer']['video_generation_calls']==0


def test_current_input_does_not_import_story_inventory_or_task(tmp_path):
    engine,_,_,image=setup(tmp_path)
    engine.state=engine.state.apply_delta({'inventory':{'items':['red_key']},
        'story':{'route':'B','flags':{'found_key':True}}})
    engine.start_from_current_frame(image)
    assert engine.state.story==WorldState().story
    assert not engine.state.inventory.items and not engine.history and engine.task is None
    assert engine.state.environment.observation_source=='vlm'
    assert not engine.state.objects['red_key'].visible
    exported=json.loads(Path(engine.export_session()).read_text())
    assert exported['initial_recovery'] is None and exported['session_resume'] is None
    metadata=exported['initial_current_frame']
    assert metadata['pixel_copy_verified'] and not metadata['new_video_generated']
    assert not metadata['historical_session_restored'] and not metadata['visual_success_verified']


def test_established_input_allows_first_current_frame_action_not_initial_fallback(tmp_path):
    engine,renderer,_,image=setup(tmp_path);engine.start_from_current_frame(image)
    binding=engine._bind_live_frame(engine.frame)
    assert not binding.initial and not renderer._live_has_generated
    evidence=request_evidence(renderer,engine.frame,engine.folder/'turn001',None,['pan_right'])
    assert evidence['input_initialized_without_video'] and evidence['expected_frames']==49
    with pytest.raises(ValueError,match='only before'):
        engine._bind_live_frame(None,initial=True)
    with pytest.raises(ValueError,match='established current session'):
        engine._bind_live_frame(None)


def test_current_input_establish_cannot_borrow_foreign_frame_or_reinitialize(tmp_path):
    engine,renderer,_,image=setup(tmp_path)
    with pytest.raises(ValueError,match='belong to the new engine session'):
        renderer.establish_live_input(engine.session_id,image,engine.folder)
    engine.start_from_current_frame(image)
    with pytest.raises(ValueError,match='fresh idle session'):
        renderer.establish_live_input(engine.session_id,engine.frame,engine.folder)
    engine.reset()
    assert not renderer._live_input_initialized and not renderer._live_has_generated
    assert engine.frame is None and engine.initial_current_frame is None


def test_invalid_rgb_input_rejects_before_reset_and_legacy_is_unchanged(tmp_path):
    engine,_,_,image=setup(tmp_path);before=engine.session_id
    for mode,size in [('RGB',(416,240)),('RGBA',(832,480))]:
        Image.new(mode,size).save(image)
        with pytest.raises(ValueError,match='exact832x480 RGB'):
            engine.start_from_current_frame(image)
        assert engine.session_id==before and engine.frame is None
    legacy=InteractionEngine(tmp_path/'legacy')
    with pytest.raises(ValueError,match='requires live'):
        legacy.start_from_current_frame(image)
