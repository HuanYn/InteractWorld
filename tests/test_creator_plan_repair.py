"""Bounded planning-loop fixtures. No real GPU or model-capability claims."""
import copy
import json

import pytest

from training.creator import model_worker as worker
from training.creator.planner import plan_request, expand_segments
from training.creator.providers import validated_planning_trace


def patch(*triples, scope='all'):
    return dict(status='ready', explanation='控制提案。', edit_scope=scope, goals=['输入控制目标'],
                edits=[dict(op='replace_intervals', key=k, intervals=[dict(start_seconds=s,end_seconds=e)]) for k,s,e in triples])


def run(tmp_path, monkeypatch, proposals, text='前4秒向右移动，然后低头2秒', previous=None):
    (tmp_path/'config.json').write_text('{}')
    seen, instances = [], []
    iterator = iter(proposals)
    class Session:
        load_count = 0
        def __init__(self, path):
            instances.append(self)
        def generate(self, system, content):
            self.load_count = 1
            seen.append((system,json.loads(content)))
            value = next(iterator)  # A third call fails the test.
            if isinstance(value, str):
                return worker.decode_model_response(value)
            return copy.deepcopy(value)
    monkeypatch.setattr(worker, '_ModelSession', Session)
    request = dict(kind='plan',text=text,previous_plan=previous,capabilities={})
    frozen = copy.deepcopy(request)
    result = worker.execute_request(tmp_path,request,tmp_path)
    assert request == frozen
    validated_planning_trace(result['planning_trace'])
    return result,seen,instances


def test_missing_action_gets_one_feedback_repair_not_rule_answer(tmp_path,monkeypatch):
    bad,good=patch(('D',0,4)),patch(('D',0,4),('K',4,6))
    result,seen,instances=run(tmp_path,monkeypatch,[bad,good])
    trace=result.pop('planning_trace')
    assert result==good and trace['outcome']=='repaired'
    assert trace['revision_count']==1 and trace['model_load_count']==1 and len(instances)==1
    feedback=seen[1][1]['validator_feedback']
    assert feedback['code']=='missing_actions' and feedback['missing_keys']==['K']
    assert 'intervals' not in json.dumps(feedback) and 'action_segments' not in json.dumps(feedback)
    assert seen[1][1]['original_request']==seen[0][1]
    assert seen[1][1]['rejected_proposal']==bad
    assert json.loads((tmp_path/'model-proposal.json').read_text(encoding='utf-8'))==bad
    assert json.loads((tmp_path/'model-proposal-repair.json').read_text(encoding='utf-8'))==good
    assert plan_request('前4秒向右移动，然后低头2秒',proposal=result)['status']=='ready'


def test_second_failure_is_terminal_without_fallback_or_third_call(tmp_path,monkeypatch):
    result,seen,_=run(tmp_path,monkeypatch,[patch(('D',0,4)),patch(('K',4,6))])
    assert len(seen)==2 and result['status']=='clarify' and result['edits']==[]
    assert result['planning_trace']['outcome']=='failed'
    assert [x['feedback']['missing_keys'] for x in result['planning_trace']['attempts']]==[['K'],['D']]


def test_ready_first_proposal_never_gets_a_second_call(tmp_path,monkeypatch):
    result,seen,_=run(tmp_path,monkeypatch,[patch(('D',0,4),('K',4,6))])
    assert len(seen)==1 and result['planning_trace']['outcome']=='first_pass'


def test_malformed_json_requires_new_model_answer_not_local_json_repair(tmp_path,monkeypatch):
    raw='{"status":"ready",'
    result,seen,_=run(tmp_path,monkeypatch,[raw,patch(('D',0,4),('K',4,6))])
    assert result['planning_trace']['outcome']=='repaired'
    assert seen[1][1]['rejected_proposal']==raw
    assert seen[1][1]['validator_feedback']['code']=='invalid_model_json'
    assert (tmp_path/'model-response.txt').read_text(encoding='utf-8')==raw


@pytest.mark.parametrize('status',['clarify','unsupported'])
def test_model_decline_not_forced_to_ready(tmp_path,monkeypatch,status):
    proposal=dict(status=status,explanation='模型不能明确完成此要求。',edit_scope='all',goals=[],edits=[])
    result,seen,_=run(tmp_path,monkeypatch,[proposal])
    assert len(seen)==1 and result['status']==status and result['planning_trace']['revision_count']==0
    assert result['planning_trace']['attempts'][0]['feedback']['repairable'] is False


@pytest.mark.parametrize('text',['第0.1到1秒抬头','走到桥边','再晚一点'])
def test_user_ambiguity_invalid_time_navigation_no_model_load(tmp_path,monkeypatch,text):
    result,seen,instances=run(tmp_path,monkeypatch,[],text=text)
    assert result['status']!='ready' and seen==[] and instances==[]
    assert result['planning_trace']['model_load_count']==0 and result['planning_trace']['attempts']==[]


def test_locked_baseline_and_tracks_still_checked_after_repair(tmp_path,monkeypatch):
    previous=plan_request('一直前进，后半段抬头')
    bad=patch(('I',7,9),scope='camera')
    good=patch(('I',8,10),scope='camera')
    text='保持前进不变，只在第8到10秒抬头，其他不要动'
    before=copy.deepcopy(previous)
    result,seen,_=run(tmp_path,monkeypatch,[bad,good],text=text,previous=previous)
    assert previous==before and seen[1][1]['original_request']['previous_plan']==before
    trace=result.pop('planning_trace')
    assert trace['outcome']=='repaired' and trace['attempts'][0]['feedback']['code']=='timing_mismatch'
    rows=expand_segments(plan_request(text,previous,result)['action_segments'])
    assert all('W' in row for row in rows) and [i for i,r in enumerate(rows) if 'I' in r]==list(range(128,160))


def test_fake_model_trace_cannot_be_injected_as_worker_metadata(tmp_path,monkeypatch):
    bad=patch(('D',0,4))
    bad['planning_trace']={'outcome':'first_pass'}
    result,seen,_=run(tmp_path,monkeypatch,[bad,bad])
    assert result['status']=='clarify' and len(seen)==2
    assert result['planning_trace']['outcome']=='failed'
