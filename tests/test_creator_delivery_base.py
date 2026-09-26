"""CPU-only delivery CLI contracts. HTTP responses below are explicit test doubles."""
from copy import deepcopy
import json

import pytest

from scripts import creator_delivery as delivery


@pytest.mark.parametrize('arguments', [
    ['--case', 'edit', '--base-version-id', 'base-a'],
    ['--case', 'initial', '--base-version-id', 'base-a'],
    ['--case', 'inspect', '--session-id', 's', '--version-id', 'v', '--base-version-id', 'base-a'],
    ['--case', 'retry', '--session-id', 's', '--version-id', 'v', '--base-version-id', 'base-a'],
    ['--case', 'edit', '--session-id', 's', '--base-version-id', ' '],
])
def test_invalid_base_cli_never_reaches_http_or_creates_report(monkeypatch, tmp_path, arguments):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('invalid CLI must not make HTTP calls')

    monkeypatch.setattr(delivery.Delivery, 'api', forbidden)
    output = tmp_path / 'invalid.json'
    with pytest.raises(SystemExit) as error:
        delivery.main(['--url', 'http://127.0.0.1:8881', '--output', str(output), *arguments])
    assert error.value.code == 2
    assert not calls and not output.exists()


@pytest.mark.parametrize('explicit', [False, True])
def test_edit_forwards_base_and_generates_request_version_not_last(monkeypatch, tmp_path, explicit):
    base = dict(version_id='base-a', request_id='old-a', plan={'status': 'ready'})
    latest = dict(version_id='latest-b', request_id='old-b', plan={'status': 'ready'})
    before = dict(session_id='session-a', scene_id='mountain', seed=42, versions=[base, latest])
    calls = []

    def api(self, path, payload=None):
        calls.append((path, deepcopy(payload)))
        if path == '/api/config':
            return {'csrf_token': 'test-only', 'generation_enabled': True}
        if path == '/api/sessions':
            return {'sessions': [deepcopy(before)]}
        if path == '/api/plan':
            assert payload.get('base_version_id') == ('base-a' if explicit else None)
            actual_base = 'base-a' if explicit else 'latest-b'
            version = dict(version_id='this-request-result', request_id=payload['request_id'],
                           parent_version=actual_base, plan={'status': 'ready'})
            response = {**deepcopy(before), 'versions': [deepcopy(base), version, deepcopy(latest)]}
            if explicit:
                version['base_version_id'] = actual_base
                response['planned_version_id'] = version['version_id']
            # The implicit case deliberately emulates an older server without
            # the added fields; request_id and parent_version remain enough.
            return response
        if path == '/api/generate':
            assert payload['version_id'] == 'this-request-result'
            return {'session_id': 'session-a', 'versions': []}
        raise AssertionError(f'unexpected fake HTTP endpoint: {path}')

    def poll(self, generated, version_id):
        assert version_id == 'this-request-result'
        self.finish('test_double_generation_not_real')
        return 0

    monkeypatch.setattr(delivery.Delivery, 'api', api)
    monkeypatch.setattr(delivery.Delivery, 'poll_job', poll)
    output = tmp_path / 'fake-responses.json'
    argv = ['--url', 'http://127.0.0.1:8881', '--output', str(output), '--case', 'edit',
            '--session-id', 'session-a', '--text', '保持移动，第8到10秒抬头']
    if explicit:
        argv += ['--base-version-id', 'base-a']
    assert delivery.main(argv) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['requested_base_version_id'] == ('base-a' if explicit else None)
    assert report['base_version_id'] == ('base-a' if explicit else 'latest-b')
    assert report['version_id'] == 'this-request-result'
    assert [path for path, _ in calls] == ['/api/config', '/api/sessions', '/api/plan', '/api/generate']


def test_server_ignoring_explicit_base_cannot_submit_generation(monkeypatch, tmp_path):
    calls = []

    def api(self, path, payload=None):
        calls.append(path)
        if path == '/api/config':
            return {'csrf_token': 'test-only', 'generation_enabled': True}
        if path == '/api/sessions':
            return {'sessions': [dict(session_id='session-a', scene_id='mountain', seed=42,
                versions=[dict(version_id='base-a', plan={'status': 'ready'})])]}
        if path == '/api/plan':
            return dict(session_id='session-a', versions=[dict(version_id='new-version',
                request_id=payload['request_id'], parent_version='wrong-base', plan={'status': 'ready'})])
        raise AssertionError('a baseline mismatch must not submit video generation')

    monkeypatch.setattr(delivery.Delivery, 'api', api)
    output = tmp_path / 'mismatch.json'
    result = delivery.main(['--url', 'http://127.0.0.1:8881', '--output', str(output), '--case', 'edit',
        '--session-id', 'session-a', '--base-version-id', 'base-a'])
    assert result == 1 and '/api/generate' not in calls
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == 'error'
    assert 'explicitly selected base version' in report['error']
