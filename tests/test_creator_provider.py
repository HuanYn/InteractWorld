"""CPU-only bridge contract; no model process or GPU is invoked."""
import pytest

from training.creator.providers import CommandProvider, validated_planning_trace


def test_inspection_bridge_forwards_distinct_raw_reference(tmp_path, monkeypatch):
    current, reference = tmp_path / 'current.mp4', tmp_path / 'reference.mp4'
    current.write_bytes(b'path-only fixture')
    reference.write_bytes(b'path-only fixture')
    provider = CommandProvider(['unused', '{request}', '{output}'], tmp_path / 'requests')
    monkeypatch.setattr(provider, '_call', lambda request: request)
    request = provider.inspect(str(current), ['镜头抬头'], reference_video_path=str(reference))
    assert request['reference_video_path'] == str(reference.resolve())
    assert request['video_path'] == str(current.resolve())
    assert 'reference_video_path' not in provider.inspect(str(current), ['镜头抬头'])
    with pytest.raises(ValueError, match='different existing'):
        provider.inspect(str(current), ['镜头抬头'], reference_video_path=str(current))


def test_planning_bridge_splits_trace_but_legacy_plan_keeps_only_proposal(tmp_path, monkeypatch):
    provider = CommandProvider(['unused', '{request}', '{output}'], tmp_path / 'requests')
    proposal = dict(status='ready', edit_scope='all', edits=[])
    trace = trace_fixture()
    requests = []
    def respond(request):
        requests.append(request)
        return {**proposal, 'planning_trace': trace}
    monkeypatch.setattr(provider, '_call', respond)
    assert provider.plan_with_trace('向前走') == dict(proposal=proposal, planning_trace=trace)
    assert provider.plan('向前走') == proposal
    assert [request['kind'] for request in requests] == ['plan', 'plan']
    monkeypatch.setattr(provider, '_call', lambda request: proposal)
    assert provider.plan_with_trace('向前走') == dict(proposal=proposal, planning_trace=None)


def trace_fixture():
    return dict(schema_version=1, max_revisions=1, model_load_count=1, outcome='repaired',
        revision_count=1, attempts=[
            dict(attempt=1, status='clarify', elapsed_seconds=1.2,
                 feedback=dict(code='missing_actions', repairable=True, message='Missing K', missing_keys=['K'])),
            dict(attempt=2, status='ready', elapsed_seconds=1.3,
                 feedback=dict(code='ok', repairable=False, message='Input check passed'))])


def test_trace_validation_returns_copy_and_allows_no_load_preflight():
    original = trace_fixture()
    result = validated_planning_trace(original)
    result['attempts'][0]['feedback']['missing_keys'].append('W')
    assert original['attempts'][0]['feedback']['missing_keys'] == ['K']
    assert validated_planning_trace(dict(schema_version=1, max_revisions=1,
        model_load_count=0, outcome='unsupported', attempts=[], revision_count=0))['attempts'] == []


@pytest.mark.parametrize('update', [dict(attempts=[{}] * 3), dict(revision_count=2),
    dict(model_load_count=True), dict(outcome='automatic_accept'), dict(arbitrary_html='<script>bad</script>')])
def test_trace_rejects_unbounded_or_unknown_fields(update):
    with pytest.raises(ValueError):
        validated_planning_trace({**trace_fixture(), **update})


@pytest.mark.parametrize('field,value', [('elapsed_seconds', float('nan')),
    ('elapsed_seconds', 3601), ('feedback', {'code':'ok','repairable':True,'message':'x'*2001}),
    ('feedback', {'code':'ok','repairable':True,'message':'bad key','protected_keys':['ATTACK']}),
    ('feedback', {'code':'ok','repairable':True,'message':'unexpected','html':'<b>bad</b>'})])
def test_trace_rejects_bad_nested_values(field, value):
    trace = trace_fixture()
    trace['attempts'][0][field] = value
    with pytest.raises(ValueError):
        validated_planning_trace(trace)
