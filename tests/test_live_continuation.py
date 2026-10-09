"""CPU interface fixtures only: no playable generated media or visual success."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from interactive.agent import SkillSelector
from interactive.live_continuation import (
    bind_current_frame,compile_live_skills,request_evidence,rgb_sha256,verify_result)
from interactive.renderers import LaboratoryWanRenderer,RenderResult


SESSION='a'*32


@pytest.fixture
def setup(tmp_path):
    folder=tmp_path/SESSION;folder.mkdir()
    initial=tmp_path/'initial.png';Image.new('RGB',(832,480),'#314559').save(initial)
    current=folder/'previous-last.png';Image.new('RGB',(832,480),'#456577').save(current)
    renderer=LaboratoryWanRenderer('http://not-called.invalid',str(initial),seconds=6,
        live_current_frame=True,live_prompt_key='neutral_current')
    return renderer,folder,current,initial


def bind_ready(setup):
    renderer,folder,current,_=setup
    renderer._live_has_generated=True
    renderer.bind_current_frame(SESSION,current,session_dir=folder)
    return renderer,folder,current


def response_fixture(setup,*,target='right'):
    renderer,folder,current=bind_ready(setup)
    plan=SimpleNamespace(semantic_action='inspect',target=target)
    skills=compile_live_skills(plan,['static'])
    out=folder/'turn001';out.mkdir()
    # Intentionally not a playable MP4; finish_render/decode is separate.
    (out/'video.mp4').write_bytes(b'CPU contract fixture, NOT generated video')
    Image.open(current).save(out/'input.png');Image.open(current).save(out/'last.png')
    actions=np.zeros((96,8),np.float32);cursor=0
    for segment in SkillSelector().action_segments(skills,frames=96):
        end=cursor+segment['frames']
        for key in segment['keys']:
            actions[cursor:end,['W','A','S','D','I','J','K','L'].index(key)]=1
        cursor=end
    np.save(out/'actions.npy',actions)
    evidence=request_evidence(renderer,current,out,plan,skills)
    result={'video_path':str(out/'video.mp4'),'last_frame':str(out/'last.png'),
        'metadata':{'kind':'wan_laboratory','conditioning_image':str(out/'input.png'),
                    'checkpoint':'R005-step1040','frames':97,'fps':16,'prompt_key':'neutral_current'}}
    return renderer,current,out,plan,evidence,result


def test_legacy_optout_is_noop_and_live_forbids_npc_goal(setup):
    renderer,folder,current,initial=setup
    legacy=LaboratoryWanRenderer('http://not-called.invalid',initial,npc_framing=True,seconds=6)
    assert bind_current_frame(legacy,'not-a-session',None) is None
    with pytest.raises(ValueError,match='forbids NPC'):
        LaboratoryWanRenderer('http://not-called.invalid',initial,npc_framing=True,
                              seconds=6,live_current_frame=True)
    renderer.bind_current_frame(SESSION,None,initial=True,session_dir=folder)
    evidence=request_evidence(renderer,None,folder/'initial',None,['static'])
    assert evidence['initial'] and Path(evidence['actual_previous_last'])==initial


def test_continuation_cannot_fallback_to_initial_or_precede_initial(setup):
    renderer,folder,current,_=setup
    with pytest.raises(ValueError,match='established current session'):
        renderer.bind_current_frame(SESSION,current,session_dir=folder)
    renderer._live_has_generated=True
    with pytest.raises(ValueError,match='established current session'):
        renderer.bind_current_frame(SESSION,None,session_dir=folder)
    with pytest.raises(ValueError,match='only before'):
        renderer.bind_current_frame(SESSION,None,initial=True,session_dir=folder)


def test_foreign_session_and_outside_frame_are_rejected(setup):
    renderer,folder,current=bind_ready(setup)
    foreign=folder.parent/('b'*32);foreign.mkdir()
    Image.open(current).save(foreign/'last.png')
    with pytest.raises(ValueError,match='another session'):
        renderer.bind_current_frame('b'*32,foreign/'last.png',session_dir=foreign)
    with pytest.raises(ValueError,match='belong to this engine session'):
        renderer.bind_current_frame(SESSION,foreign/'last.png',session_dir=folder)


def test_request_must_match_engine_binding_and_owned_output(setup):
    renderer,folder,current=bind_ready(setup)
    other=folder/'other.png';Image.open(current).save(other)
    for invalid in (other,None):
        with pytest.raises(ValueError,match='differs'):
            request_evidence(renderer,invalid,folder/'turn001',None,['static'])
    with pytest.raises(ValueError,match='inside this engine session'):
        request_evidence(renderer,current,folder.parent/'foreign',None,['static'])


def test_pixel_tamper_and_continuation_resize_are_rejected(setup):
    renderer,folder,current=bind_ready(setup)
    Image.new('RGB',(832,480),'red').save(current)
    with pytest.raises(ValueError,match='pixels changed'):
        request_evidence(renderer,current,folder/'turn001',None,['static'])
    Image.new('RGB',(416,240),'red').save(current)
    with pytest.raises(ValueError,match='crop/resize substitute'):
        renderer.bind_current_frame(SESSION,current,session_dir=folder)


def test_direction_compiles_real_j_l_and_no_other_motion_override():
    for target,key in [('left','J'),('right','L')]:
        plan=SimpleNamespace(semantic_action='observe',target=target)
        skills=compile_live_skills(plan,['static'])
        assert SkillSelector().action_segments(skills,96)==[
            {'frames':48,'keys':[key]},{'frames':48,'keys':[]}]
    assert compile_live_skills(SimpleNamespace(semantic_action='inspect',target='desk'),['static'])==['static']


def test_return_contract_pass_is_not_visual_or_navigation_success(setup):
    _,_,_,plan,evidence,result=response_fixture(setup)
    actual=verify_result(result,evidence,plan)
    assert actual['conditioning_rgb_match'] and actual['lineage_check_passed']
    assert actual['actual_action_tensor_shape']==[96,8]
    assert actual['actual_nonzero_action_values']==48
    assert actual['lineage_is_not_visual_success']
    for key in ('visual_success_verified','camera_motion_verified','world_navigation_verified'):
        assert actual[key] is False


def test_prebuilt_goal_retained_and_recovered_results_reject(setup):
    _,_,out,plan,evidence,result=response_fixture(setup)
    for forbidden in ('pre_generated','explicit_goal_input','existing_frame_reused',
                      'recovered_existing_initial','framing_condition'):
        result['metadata'][forbidden]=True
        with pytest.raises(ValueError,match='not a live visual action'):
            verify_result(result,evidence,plan)
        del result['metadata'][forbidden]
    (out/'supplied-framing-goal.png').write_bytes(b'forbidden CPU fixture')
    with pytest.raises(ValueError,match='goal conditioning artifacts'):
        verify_result(result,evidence,plan)


def test_returned_input_and_direction_actions_must_match(setup):
    _,current,out,plan,evidence,result=response_fixture(setup)
    Image.new('RGB',(832,480),'red').save(out/'input.png')
    with pytest.raises(ValueError,match='Actual model input differs'):
        verify_result(result,evidence,plan)
    Image.open(current).save(out/'input.png')
    np.save(out/'actions.npy',np.zeros((96,8),np.float32))
    with pytest.raises(ValueError,match='Actual action schedule differs'):
        verify_result(result,evidence,plan)


def test_returned_artifact_foreign_path_rejects(setup):
    _,current,_,plan,evidence,result=response_fixture(setup)
    result['last_frame']=str(current)
    with pytest.raises(ValueError,match='do not belong'):
        verify_result(result,evidence,plan)


def test_explicit_reset_allows_new_story_but_never_inflight(setup):
    renderer,folder,current=bind_ready(setup)
    new_id='b'*32;new_folder=folder.parent/new_id;new_folder.mkdir()
    renderer._live_request_active=True
    with pytest.raises(ValueError,match='active render'):
        renderer.reset_live_session(new_id)
    assert renderer._live_session_id==SESSION
    renderer._live_request_active=False
    renderer.reset_live_session(new_id)
    assert renderer._live_binding is None and not renderer._live_has_generated
    renderer.bind_current_frame(new_id,None,initial=True,session_dir=new_folder)
    assert renderer._live_binding.session_id==new_id


def test_renderer_submission_is_once_bound_and_contains_no_goal(setup,monkeypatch):
    renderer,current,out,plan,_,result=response_fixture(setup)
    calls=[]
    health=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'busy':False})
    def post(url,**kwargs):
        calls.append(kwargs['json'])
        assert renderer._live_request_active
        return SimpleNamespace(ok=True,json=lambda:result)
    monkeypatch.setattr('interactive.renderers.requests.get',lambda *a,**kw:health)
    monkeypatch.setattr('interactive.renderers.requests.post',post)
    monkeypatch.setattr(renderer,'finish_render',lambda result,*a:RenderResult(
        result['video_path'],result['last_frame'],result['metadata']))
    generated=renderer.generate_plan(current,['static'],'',None,out,plan)
    assert len(calls)==1 and calls[0]['skills']==['pan_right','static']
    assert 'framing_condition' not in calls[0]
    assert generated.metadata['live_current_frame_evidence']['lineage_check_passed']
    assert renderer._live_binding is None and not renderer._live_request_active
    with pytest.raises(ValueError,match='before every live request'):
        renderer.generate_plan(current,['static'],'',None,out,plan)
    assert len(calls)==1


def test_live_requires_explicit_nonempty_scene_key_and_legacy_is_unchanged(setup):
    _,_,_,initial=setup
    for key in (None,'','   ',' neutral_current'):
        with pytest.raises(ValueError,match='explicit live_prompt_key'):
            LaboratoryWanRenderer('http://not-called.invalid',initial,live_current_frame=True,live_prompt_key=key)
    legacy=LaboratoryWanRenderer('http://not-called.invalid',initial)
    assert legacy.prompt_for(None)=='scene' and legacy.live_prompt_key is None


def test_returned_missing_or_wrong_prompt_key_is_not_accepted(setup):
    _,_,_,plan,evidence,result=response_fixture(setup)
    for key in (None,'scene','another_neutral'):
        result['metadata']['prompt_key']=key
        with pytest.raises(ValueError,match='explicitly configured scene cache key'):
            verify_result(result,evidence,plan)


def test_full_window_pan_is_explicit_three_second_only(setup):
    _,_,_,initial=setup
    for invalid in (dict(seconds=6,live_current_frame=True,live_full_window_pan=True),
                    dict(seconds=3,live_current_frame=False,live_full_window_pan=True),
                    dict(seconds=3,live_current_frame=True,live_full_window_pan=1)):
        with pytest.raises(ValueError,match='Full-window live pan'):
            LaboratoryWanRenderer('http://not-called.invalid',initial,live_prompt_key='neutral_current',**invalid)
    for target,key in [('right','L'),('left','J')]:
        plan=SimpleNamespace(semantic_action='observe',target=target)
        skills=compile_live_skills(plan,['static'],full_window_pan=True)
        assert SkillSelector().action_segments(skills,48)==[{'frames':48,'keys':[key]}]
        assert len(compile_live_skills(plan,['static']))==2


def test_explicit_prompt_is_used_for_initial_and_camera_requests(setup,monkeypatch):
    _,folder,current,initial=setup
    for initial_phase in (True,False):
        renderer=LaboratoryWanRenderer('http://not-called.invalid',initial,seconds=3,
            live_current_frame=True,live_prompt_key='neutral_archive',live_full_window_pan=True)
        renderer._live_has_generated=not initial_phase
        frame=None if initial_phase else current
        renderer.bind_current_frame(SESSION,frame,initial=initial_phase,session_dir=folder)
        plan=None if initial_phase else SimpleNamespace(semantic_action='inspect',target='right')
        payloads=[]
        def submit(payload,*unused):
            payloads.append(payload)
            return RenderResult(None,None,{'software_fixture_only':True})
        monkeypatch.setattr(renderer,'_submit',submit)
        renderer.generate_plan(frame,['static'],'',None,folder/('initial' if initial_phase else 'turn001'),plan)
        assert payloads[0]['prompt_key']=='neutral_archive'
        assert payloads[0]['skills']==(['static'] if initial_phase else ['pan_right'])
        assert payloads[0]['seconds']==3 and 'framing_condition' not in payloads[0]


def test_wrong_native_duration_rejects_even_when_frame_count_is_supported(setup):
    _,_,_,plan,evidence,result=response_fixture(setup)
    result['metadata']['frames']=49
    with pytest.raises(ValueError,match='bounded live frame count'):
        verify_result(result,evidence,plan)
