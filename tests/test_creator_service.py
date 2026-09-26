"""CPU-only service contracts with explicit test doubles, never model evidence.

These tests do not load a model, launch a GPU worker, or manufacture demo videos.
The in-memory queue and canned visual assessment exist only inside unit tests.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.creator.planner import plan_request
from training.creator.service import CreatorService


class UnitTestQueue:
    """In-memory queue test double; submit never executes inference."""

    def __init__(self, root):
        self.deployment = SimpleNamespace(project_root=root, jobs_root=root / 'jobs')
        self.catalog = SimpleNamespace(scenes={'courtyard': object(), 'forest': object()})
        self.jobs = {}
        self.submitted = []

    def list_jobs(self):
        return deepcopy(list(self.jobs.values()))

    def submit(self, payload):
        self.submitted.append(deepcopy(payload))
        job = dict(job_id=f'unit-job-{len(self.submitted)}', status='queued')
        self.jobs[job['job_id']] = job
        return deepcopy(job)


class UnitTestProvider:
    """Canned responses for service tests only; not a production provider."""

    name = 'unit_test_only_not_a_model'

    def __init__(self):
        self.plans = []
        self.inspections = []
        self.references = []
        self.on_inspect = None
        self.assessment = dict(
            verdict='unsatisfied', decision='revise',
            evidence=[dict(time_seconds=10.0, observation='UNIT TEST ONLY: camera action unclear')],
            revision_text='缩短抬头',
        )

    def plan(self, text, previous):
        self.plans.append((text, deepcopy(previous)))
        # Model-like proposal with independent schema output; audit metadata in
        # the input must not alter the production planner's validation contract.
        prior = {'action_segments': previous['action_segments']} if previous else None
        return plan_request(text, previous=prior)

    def inspect(self, video, goals, *, reference_video_path=None):
        self.inspections.append((video, deepcopy(goals)))
        self.references.append(reference_video_path)
        if self.on_inspect:
            self.on_inspect()
        return deepcopy(self.assessment)


@pytest.fixture
def service(tmp_path):
    return CreatorService(UnitTestQueue(tmp_path), provider=UnitTestProvider(), visual_revision_enabled=True)


def make_plan(service, **changes):
    payload = dict(scene_id='courtyard', seed=42, text='先前进，然后抬头', request_id='unit-request-001')
    payload.update(changes)
    return service.plan(payload)


def test_visual_revision_defaults_off_and_preserves_raw_model_recommendation(tmp_path):
    service = CreatorService(UnitTestQueue(tmp_path), provider=UnitTestProvider())
    session = complete(service, make_plan(service))
    result = service.inspect(target(session))
    review = result['versions'][-1]['review']
    assert service.visual_revision_enabled is False
    assert review['decision'] == 'ask_user'
    assert review['model_decision'] == 'revise'
    assert review['model_revision_text'] == service.provider.assessment['revision_text']
    assert review['downgrade_reason'] and review['visual_revision_enabled'] is False
    stored = service.sessions[session['session_id']]['versions'][-1]['review']
    assert stored['decision'] == 'revise'
    assert 'downgrade_reason' not in stored
    with pytest.raises(ValueError, match='visual revision is disabled'):
        service.revise(target(session))
    assert len(service.provider.plans) == 1
    assert len(service.demo.submitted) == 1


def test_disabled_policy_overlays_saved_reviews_without_rewriting_evidence_or_human_review(service):
    session = complete(service, make_plan(service))
    service.inspect(target(session))
    stored_session = service.sessions[session['session_id']]
    # A previously saved review predates the separate original-decision fields.
    stored_session['versions'][-1]['review'].pop('model_decision')
    stored_session['versions'][-1]['review'].pop('model_revision_text')
    service._save(stored_session)
    path = service.root / (session['session_id'] + '.json')
    before = path.read_bytes()
    raw_review = deepcopy(service.sessions[session['session_id']]['versions'][-1]['review'])
    disabled = CreatorService(service.demo, provider=service.provider)
    result = disabled.list_sessions()[0]['versions'][-1]['review']
    assert result['decision'] == 'ask_user' and result['model_decision'] == 'revise'
    assert disabled.sessions[session['session_id']]['versions'][-1]['review'] == raw_review
    assert path.read_bytes() == before
    human = disabled.review({**target(session), 'verdict': 'uncertain', 'evidence': 'Human must check the actual video.'})
    stored_human = disabled.sessions[session['session_id']]['versions'][-1]['review']
    assert human['versions'][-1]['review'] == stored_human
    assert human['versions'][-1]['review']['source'] == 'human'
    assert 'downgrade_reason' not in stored_human
    assert stored_human['previous_assessment'] == raw_review


def test_visual_revision_does_not_replace_a_newer_user_plan(service):
    original = complete(service, make_plan(service))
    service.inspect(target(original))
    make_plan(service, session_id=original['session_id'], text='缩短抬头', request_id='new-user-edit-001')
    with pytest.raises(ValueError, match='newer user plan'):
        service.revise(target(original))


def test_explicit_rule_planning_never_calls_model_or_submits_gpu(service):
    session = make_plan(service, planner='rule_fallback')
    version = session['versions'][-1]
    assert version['planner_kind'] == 'rule_fallback'
    assert version['provider'] == 'rule_fallback'
    assert version['plan']['status'] == 'ready'
    assert service.provider.plans == [] and service.demo.submitted == []
    assert make_plan(service, planner='rule_fallback') == session
    with pytest.raises(ValueError, match='different planner mode'):
        make_plan(service, planner='local_model')
    modified = make_plan(service, session_id=session['session_id'], planner='rule_fallback',
        request_id='explicit-rule-edit-001', text='保留前进，只缩短抬头')
    assert modified['versions'][-1]['plan']['status'] == 'ready'
    assert service.provider.plans == [] and service.demo.submitted == []


def test_model_failure_does_not_silently_fall_back_to_rules(service):
    def fail(*args):
        raise RuntimeError('UNIT TEST: cold load failed')
    service.provider.plan = fail
    with pytest.raises(RuntimeError, match='cold load failed'):
        make_plan(service, planner='local_model')
    assert all(not s['versions'] for s in service.sessions.values())
    assert not service.operations and not service.demo.submitted


def test_rule_only_server_rejects_explicit_model_and_unknown_modes(service):
    service.provider = None
    with pytest.raises(ValueError, match='not configured'):
        make_plan(service, planner='local_model')
    with pytest.raises(ValueError, match='unknown planner'):
        make_plan(service, planner='silent_auto_fallback')
    assert not service.sessions
    assert make_plan(service)['versions'][-1]['provider'] == 'rule_fallback'


def test_human_criteria_and_prior_assessments_survive_edits_and_reload(service):
    session = complete(service, make_plan(service))
    inspected = service.inspect(target(session))
    assessment = deepcopy(inspected['versions'][-1]['review'])
    first = service.review({**target(session), 'verdict': 'uncertain',
        'evidence': 'UNIT TEST ONLY: compare movement and camera separately.',
        'criteria': {'movement_response': 'satisfied', 'camera_response': 'uncertain'}})
    human = deepcopy(first['versions'][-1]['review'])
    second = service.review({**target(session), 'verdict': 'unsatisfied',
        'evidence': 'UNIT TEST ONLY: revised human observation.',
        'criteria': {'temporal_stability': 'unsatisfied'}})
    current = second['versions'][-1]
    assert current['review']['previous_assessment'] == assessment
    assert current['review_history'] == [assessment, human]
    assert second['accepted_version'] is None
    reloaded = CreatorService(service.demo, provider=service.provider, visual_revision_enabled=True)
    persisted = reloaded.list_sessions()[0]['versions'][-1]
    assert persisted['review'] == current['review']
    assert persisted['review_history'] == current['review_history']
    with pytest.raises(ValueError, match='human review already exists'):
        reloaded.inspect(target(session))


@pytest.mark.parametrize('criteria', [None, [], {'unknown': 'satisfied'},
    {'camera_response': True}, {'camera_response': 'pass'}])
def test_invalid_human_criteria_do_not_mutate_session(service, criteria):
    session = complete(service, make_plan(service))
    path = service.root / (session['session_id'] + '.json')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='invalid human'):
        service.review({**target(session), 'verdict': 'satisfied',
            'evidence': 'UNIT TEST ONLY', 'criteria': criteria})
    assert path.read_bytes() == before
    assert service.list_sessions()[0] == session


def test_uncertain_insufficient_frames_can_degrade_without_fake_evidence(service):
    session = complete(service, make_plan(service))
    service.provider.assessment = dict(verdict='uncertain', decision='ask_user', evidence=[],
        revision_text='', confidence_note='insufficient actual frames')
    result = service.inspect(target(session))
    assert result['versions'][-1]['review']['verdict'] == 'uncertain'
    assert not result['versions'][-1]['review']['automatic_acceptance']


def test_explicit_retry_preserves_plan_and_does_not_repeat_llm(service):
    session = service.generate(target(make_plan(service)))
    old = session['versions'][-1]
    request = {**target(session), 'request_id': 'explicit-retry-001'}
    with pytest.raises(ValueError, match='only failed'):
        service.retry(request)
    service.demo.jobs[old['job_id']]['status'] = 'failed'
    result = service.retry(request)
    new = result['versions'][-1]
    assert new['plan'] == old['plan'] and new['retry_of'] == old['version_id']
    assert new['job_id'] != old['job_id'] and len(service.provider.plans) == 1
    assert service.retry(request)['versions'] == result['versions']
    assert service.retry({**request, 'request_id': 'explicit-retry-other-id'})['versions'] == result['versions']
    assert len(service.demo.submitted) == 2


def test_retry_chain_stays_bounded_after_reload_and_parent_replays(service):
    original = service.generate(target(make_plan(service)))
    service.demo.jobs[original['versions'][-1]['job_id']]['status'] = 'failed'
    first = service.retry({**target(original), 'request_id': 'retry-chain-first'})
    service.demo.jobs[first['versions'][-1]['job_id']]['status'] = 'failed'
    second = service.retry({**target(first), 'request_id': 'retry-chain-second'})
    service.demo.jobs[second['versions'][-1]['job_id']]['status'] = 'failed'
    reloaded = CreatorService(service.demo, provider=service.provider)
    for index, parent in enumerate((original, first)):
        result = reloaded.retry({**target(parent), 'request_id': f'retry-parent-replay-{index}'})
        assert len(result['versions']) == 3
        assert result['versions'][-1]['version_id'] == second['versions'][-1]['version_id']
    with pytest.raises(ValueError, match='two retries exhausted'):
        reloaded.retry({**target(second), 'request_id': 'retry-chain-third'})
    assert [v.get('retry_count', 0) for v in result['versions']] == [0, 1, 2]
    assert len(service.demo.submitted) == 3
    assert len(service.provider.plans) == 1


def target(session, version_index=-1):
    return dict(session_id=session['session_id'], version_id=session['versions'][version_index]['version_id'])


def complete(service, session):
    session = service.generate({**target(session), 'request_id': 'unit-generate-001'})
    service.demo.jobs[session['versions'][-1]['job_id']]['status'] = 'completed'
    return service.list_sessions()[0]


def test_plan_is_idempotent_before_and_after_session_reload(service):
    first = make_plan(service)
    assert first['versions'][0]['plan']['status'] == 'ready'
    assert make_plan(service) == first
    assert len(service.provider.plans) == 1
    assert service.demo.submitted == []  # Planning does not imply generation.

    reloaded = CreatorService(service.demo, provider=service.provider)
    assert make_plan(reloaded) == first
    assert len(service.provider.plans) == 1
    with pytest.raises(ValueError, match='different input'):
        make_plan(reloaded, text='前进')
    with pytest.raises(ValueError, match='condition mismatch'):
        make_plan(reloaded, seed=43)


@pytest.mark.parametrize('field,value', [('scene_id', 'forest'), ('seed', 43)])
def test_session_scene_and_seed_are_immutable(service, field, value):
    first = make_plan(service)
    with pytest.raises(ValueError, match='immutable'):
        make_plan(service, session_id=first['session_id'], request_id='unit-request-002', **{field: value})
    assert len(service.list_sessions()[0]['versions']) == 1


def test_local_edit_keeps_a_ready_plan_with_provider_provenance(service):
    first = make_plan(service)
    assert first['versions'][0]['provider'] == service.provider.name
    edited = make_plan(service, session_id=first['session_id'], request_id='unit-request-002',
                       text='保留前进，缩短抬头')
    plan = edited['versions'][-1]['plan']
    assert plan['status'] == 'ready', plan['explanation']
    assert plan['edit_scope'] == 'camera'
    assert plan['preserved'] is True
    assert edited['versions'][-1]['parent_version'] == first['versions'][0]['version_id']
    assert service.demo.submitted == []


def test_generate_is_idempotent_and_uses_frozen_scene_seed_actions(service):
    first = make_plan(service)
    generated = service.generate({**target(first), 'request_id': 'unit-generate-001'})
    assert service.generate({**target(first), 'request_id': 'unit-generate-002'}) == generated
    assert len(service.demo.submitted) == 1
    assert service.demo.submitted[0] == dict(
        scene_id='courtyard', seed=42, action_segments=first['versions'][0]['plan']['action_segments'])


def test_unexecutable_request_does_not_reach_queue(service):
    first = make_plan(service, text='打开门')
    assert first['versions'][0]['plan']['status'] == 'unsupported'
    with pytest.raises(ValueError, match='clarify'):
        service.generate(target(first))
    assert service.demo.submitted == []


@pytest.mark.parametrize('status', ['queued', 'running', 'failed'])
def test_noncompleted_jobs_cannot_be_accepted_or_inspected(service, status):
    session = service.generate(target(make_plan(service)))
    service.demo.jobs[session['versions'][0]['job_id']]['status'] = status
    for action in (service.accept, service.inspect):
        with pytest.raises(ValueError, match='completed real generation'):
            action(target(session))
    assert service.provider.inspections == []
    assert service.list_sessions()[0]['accepted_version'] is None
    assert service.list_sessions()[0]['versions'][0]['review'] is None


def test_inspection_reads_raw_video_and_cites_temporal_model_evidence(service):
    session = complete(service, make_plan(service))
    inspected = service.inspect(target(session))
    video, goals = service.provider.inspections[0]
    assert Path(video) == service.demo.deployment.jobs_root / session['versions'][0]['job_id'] / 'raw.mp4'
    assert goals[:-1] == session['versions'][0]['plan']['goals']
    assert goals[-1] == '用户当前请求：' + session['versions'][0]['text']
    review = inspected['versions'][0]['review']
    assert review['source'] == 'model_assessment'
    assert review['calibrated'] is False
    assert review['automatic_acceptance'] is False
    assert review['evidence'][0]['time_seconds'] == 10.0
    assert inspected['accepted_version'] is None


def test_visual_feedback_creates_at_most_one_revision_without_generating(service):
    original = complete(service, make_plan(service))
    reviewed = service.inspect(target(original))
    revised = service.revise({**target(reviewed), 'request_id': 'unit-revise-001'})
    new = revised['versions'][-1]
    assert len(revised['versions']) == 2
    assert new['plan']['status'] == 'ready', new['plan']['explanation']
    assert new['automatic_revisions'] == 1
    assert new['origin'] == 'visual_feedback'
    assert new['parent_version'] == original['versions'][0]['version_id']
    assert new['root_request'] == original['versions'][0]['root_request']
    assert new['job_id'] is None
    assert len(service.demo.submitted) == 1
    assert service.revise({**target(reviewed), 'request_id': 'unit-revise-002'}) == revised

    completed_revision = complete(service, revised)
    reviewed_revision = service.inspect(target(completed_revision))
    # A repeat may be rejected or return the existing revision idempotently;
    # either result must preserve the finite bound without another proposal.
    before = len(service.provider.plans)
    try:
        result = service.revise({**target(reviewed_revision), 'request_id': 'unit-revise-003'})
    except ValueError as error:
        assert 'maximum one' in str(error)
    else:
        assert len(result['versions']) == 2
    assert len(service.provider.plans) == before


def test_human_review_preserves_model_evidence_and_blocks_automatic_overwrite(service):
    session = complete(service, make_plan(service))
    inspected = service.inspect(target(session))
    human = service.review({**target(session), 'verdict': 'satisfied', 'evidence': 'Human saw the requested camera motion.'})
    review = human['versions'][0]['review']
    assert review['source'] == 'human'
    assert review['verdict'] == 'satisfied'
    assert review['previous_assessment'] == inspected['versions'][0]['review']
    with pytest.raises(ValueError, match='human review already exists'):
        service.inspect(target(session))
    assert len(service.provider.inspections) == 1
    assert service.list_sessions()[0]['versions'][0]['review'] == review
    assert human['accepted_version'] is None  # Satisfaction is not acceptance.
    accepted = service.accept(target(session))
    assert accepted['accepted_version'] == session['versions'][0]['version_id']


def test_human_review_arriving_during_inspection_wins(service):
    session = complete(service, make_plan(service))
    service.provider.on_inspect = lambda: service.review({
        **target(session), 'verdict': 'uncertain', 'evidence': 'Human needs another look.'})
    with pytest.raises(ValueError, match='human review arrived'):
        service.inspect(target(session))
    stored = service.list_sessions()[0]['versions'][0]['review']
    assert stored['source'] == 'human'
    assert stored['verdict'] == 'uncertain'
    assert service.operations == {}


def test_unconfigured_provider_has_explicit_rule_and_human_fallback(tmp_path):
    service = CreatorService(UnitTestQueue(tmp_path))
    session = complete(service, make_plan(service))
    assert session['versions'][0]['plan']['planner_kind'] == 'rule_fallback'
    with pytest.raises(ValueError, match='not configured'):
        service.inspect(target(session))
    reviewed = service.review({**target(session), 'verdict': 'unsatisfied', 'evidence': 'Human observes missing movement.'})
    assert reviewed['versions'][0]['review']['source'] == 'human'
    with pytest.raises(ValueError, match='needs a model provider'):
        service.revise(target(session))


def test_model_failure_does_not_fabricate_a_review_or_advance_state(service):
    session = complete(service, make_plan(service))

    def fail():
        raise RuntimeError('unit test provider failure')

    service.provider.on_inspect = fail
    with pytest.raises(RuntimeError, match='unit test provider failure'):
        service.inspect(target(session))
    stored = service.list_sessions()[0]
    assert stored['versions'][0]['review'] is None
    assert len(stored['versions']) == 1
    assert stored['accepted_version'] is None
    assert service.operations == {}


def test_relative_inspection_uses_completed_parent_retry_and_keeps_history(service):
    original = service.generate(target(make_plan(service)))
    service.demo.jobs[original['versions'][0]['job_id']]['status'] = 'failed'
    edited = make_plan(service, session_id=original['session_id'], request_id='edit-relative-001', text='保留前进，缩短抬头')
    retry = service.retry({**target(original), 'request_id': 'retry-reference-001'})
    reference = retry['versions'][-1]
    service.demo.jobs[reference['job_id']]['status'] = 'completed'
    generated = service.generate(target(edited))
    version = next(v for v in generated['versions'] if v['version_id'] == target(edited)['version_id'])
    service.demo.jobs[version['job_id']]['status'] = 'completed'
    inspected = service.inspect(target(edited))
    review = next(v for v in inspected['versions'] if v['version_id'] == version['version_id'])['review']
    assert review['reference_version_id'] == reference['version_id']
    assert service.provider.references[-1] == str(service.demo.deployment.jobs_root / reference['job_id'] / 'raw.mp4')
    service.inspect(target(edited))
    stored = service._version(service._session(edited['session_id']), version['version_id'])
    assert len(stored['review_history']) == 1 and stored['review_history'][0] == review


def test_visual_revision_cannot_undo_the_users_shortening_request(service):
    original = complete(service, make_plan(service))
    edited = complete(service, make_plan(service, session_id=original['session_id'],
        request_id='shorten-parent-001', text='保留前进，缩短抬头'))
    service.provider.assessment['revision_text'] = '延长抬头'
    service.inspect(target(edited))
    restored_parent = deepcopy(original['versions'][0]['plan'])
    restored_parent['edit_scope'] = 'camera'
    service.provider.plan = lambda text, previous: restored_parent
    with pytest.raises(ValueError, match='contradicts the original'):
        service.revise(target(edited))
    assert len(service.list_sessions()[0]['versions']) == 2
    assert service.operations == {}
