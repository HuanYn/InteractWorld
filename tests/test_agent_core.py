from pathlib import Path
import pytest
from interactive.models import WorldState,Plan,Observation,parse_plan
from interactive.rules import RuleEngine
from interactive.engine import InteractionEngine
from interactive.observer import Reconciler
from interactive.agent import LLMAgent,SkillSelector
from interactive.renderers import RenderResult,WanRenderer


class LightRenderer:
    def generate(self,init_frame,skills,prompt,state,output_dir):
        return RenderResult(None,None,{'kind':'test_double'})


def engine(tmp_path): return InteractionEngine(tmp_path,renderer=LightRenderer())


def test_state_roundtrip():
    s=WorldState(); assert WorldState.deserialize(s.serialize())==s
    assert s.apply_delta({'player':{'location':'laboratory'}}).player.location=='laboratory'
    assert s.player.location=='entrance'


def test_door_preconditions():
    r=RuleEngine(); p=Plan(player_intent='open_door',target='door_a',semantic_action='open')
    assert not r.validate(WorldState(),p).valid
    s=WorldState().apply_delta({'inventory':{'items':['red_key']},'story':{'route':'B'}})
    result=r.validate(s,p); assert result.valid
    assert s.apply_delta(result.state_delta).objects['door_a'].state=='open'


@pytest.mark.parametrize('route,texts,end',[
    ('A',['进入实验室','走近 Anna','帮助 Anna','询问密码','检查终端','进入核心区域'],'ending_anna'),
    ('B',['进入实验室','检查左边','走近 Anna','我不相信 Anna','搜索桌子','拿起钥匙','打开门','进入储藏室'],'ending_key'),
    ('C',['检查终端','读取终端日志','找到 Leo','询问 Leo'],'ending_terminal')])
def test_routes(tmp_path,route,texts,end):
    e=engine(tmp_path)
    for text in texts:
        row=e.interact(text); assert row['valid'],row
    assert e.state.story.node==end and e.state.story.route==route
    assert e.state.story.terminal and len(e.memory.short)<=5 and e.memory.long
    e.reset(); assert e.state.story.node=='enter_lab' and not e.memory.long


def test_no_password_leak_or_state_forgery(tmp_path):
    e=engine(tmp_path); e.interact('走近 Anna'); row=e.interact('询问密码')
    assert not row['valid'] and 'AURORA' not in row['npc_response']
    assert not e.state.story.flags.get('password_known')
    with pytest.raises(ValueError): parse_plan('{"player_intent":"hack","state_delta":{"inventory":["red_key"]}}')


def test_memory_pronoun(tmp_path):
    e=engine(tmp_path); e.interact('检查终端'); row=e.interact('继续检查它')
    assert row['plan']['target']=='terminal' and row['story_event']=='retrieve_log'


def test_parse_retry_fallback(tmp_path):
    class Bad:
        n=0
        def complete(self,messages): self.n+=1; return 'not JSON'
    provider=Bad(); e=engine(tmp_path); e.agent=LLMAgent(provider)
    row=e.interact('go'); assert provider.n==2
    assert row['plan']['semantic_action']=='observe'


def test_failed_render_rolls_back(tmp_path):
    class Broken:
        def generate(self,*args): raise RuntimeError('test failure')
    e=engine(tmp_path); e.renderer=Broken(); before=e.state.serialize()
    row=e.interact('走近 Anna'); assert not row['valid']
    assert e.state.serialize()==before and not e.memory.long


def test_visual_cannot_unlock():
    s=WorldState(); obs=Observation(source='vlm',scene='lab',visible_objects=['door_a','invented'],
        major_event='door opened',confidence=1)
    next_state=Reconciler().reconcile(s,obs)
    assert next_state.objects['door_a'].state=='locked'
    assert next_state.environment.visual_evidence['visible_objects']==['door_a']


def test_wan_adapter_fail_closed():
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'generation_enabled':False}
    class Session:
        def get(self,*a,**kw): return Response()
    renderer=WanRenderer('http://127.0.0.1:8766'); renderer.session=Session()
    with pytest.raises(RuntimeError,match='disabled'): renderer.generate_preset(['pan_left'],'unused')
    with pytest.raises(RuntimeError,match='REAL LAB WAN BLOCKED'): renderer.generate(None,[],None,None,None)
    rows=SkillSelector().action_segments(['pan_left','static']); assert sum(r['frames'] for r in rows)==240


def test_actual_mock_video(tmp_path):
    import imageio.v2 as imageio
    e=InteractionEngine(tmp_path); result=e.start_world()
    assert Path(result['video']).stat().st_size>0
    reader=imageio.get_reader(result['video'])
    try: assert reader.get_data(0).shape==(432,768,3)
    finally: reader.close()


def test_lab_prompt_routing_and_visual_feedback(tmp_path):
    from interactive.renderers import LaboratoryWanRenderer
    p=Plan(player_intent='search',semantic_action='search',target='desk')
    assert LaboratoryWanRenderer.prompt_for(p)=='key_reveal'
    class Renderer(LightRenderer):
        def generate_plan(self,*args):
            self.plan=args[-1]
            return RenderResult(None,None,{'kind':'test_double'})
    class Observer:
        def observe(self,*args): return Observation(source='vlm',evidence='test only')
        def assess_goal(self,*args): return {'status':'contradicted','authoritative_for_story':False}
    e=engine(tmp_path); e.renderer=Renderer(); e.observer=Observer()
    row=e.interact('走近 Anna')
    assert row['valid'] and e.state.story.flags['met_anna']
    assert row['visual_goal_verified'] is False
    assert row['visual_assessment']['status']=='contradicted'
    assert e.memory.retrieve()['recent'][-1]['visual_assessment']['status']=='contradicted'
    assert e.renderer.plan.target=='anna'
