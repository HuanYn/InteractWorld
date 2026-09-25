"""CPU recovery contracts; placeholder bytes never claim video/model quality."""
from contextlib import contextmanager, nullcontext
import json

import numpy as np
import pytest
import yaml

from training.demo import backend
from training.demo.contracts import action_array, sha256, write_json


@pytest.fixture
def job(tmp_path, monkeypatch):
    directory = tmp_path / 'jobs' / ('a' * 32)
    directory.mkdir(parents=True)
    segments = [dict(frames=240, keys=['W'])]
    np.save(directory / 'initial.npy', np.zeros((480, 832, 3), dtype=np.uint8))
    np.save(directory / 'actions.npy', action_array(segments))
    scene = dict(scene_id='scene', source_episode_id='episode', seed=7, prompt='A courtyard.',
                 action_segments=segments, initial_frame_path=str(directory / 'initial.npy'))
    raw = dict(run_id=directory.name, output_root=str(directory), method='cpu-test-method',
               geometry=dict(width=832, height=480, fps=16, total_rgb_frames=241),
               lineage=dict(checkpoint_sha256='b' * 64), scenes=[scene])
    (directory / 'rollout.yaml').write_text(yaml.safe_dump(raw))
    request = dict(job_id=directory.name, **{key: scene[key] for key in ('scene_id', 'source_episode_id', 'seed', 'prompt', 'action_segments')},
        ground_truth_future_used=False, fps=16, future_frames=240, method='cpu-test-method', checkpoint_sha256='b' * 64,
        config_sha256=sha256(directory / 'rollout.yaml'), initial_sha256=sha256(directory / 'initial.npy'),
        actions_sha256=sha256(directory / 'actions.npy'))
    write_json(directory / 'request.json', request)
    lease = dict(job_id=directory.name, request_sha256=sha256(directory / 'request.json'),
                 reservation_id='CPU-only-old-reservation', authorization_id='CPU-test-only')
    write_json(directory / 'lease.json', lease)
    write_json(directory / 'settlement-request.json', dict(job_id=directory.name, request_sha256=lease['request_sha256'],
        lease=lease, outcome='failed', worker_returncode=1))
    write_json(directory / 'settlement.json', dict(status='settled', reservation_id=lease['reservation_id']))
    write_json(directory / 'status.json', dict(job_id=directory.name, status='failed', error='original annotation error',
                                              started_unix=100, finished_unix=3700, failure_type='RuntimeError'))
    (directory / 'raw.mp4').write_bytes(b'CPU-TEST-PLACEHOLDER-NOT-A-VIDEO')
    monkeypatch.setattr(backend, '_decode_recovery_video', lambda path: dict(width=832, height=480, frames=241, fps=16,
                                                                          complete_software_decode=True, test_only=True))
    from training.eval import input_header
    def annotate(source, output, **kwargs):
        assert source == directory / 'raw.mp4' and kwargs['initial_frame'].shape == (480, 832, 3)
        np.testing.assert_array_equal(kwargs['actions'], action_array(segments))
        output.write_bytes(b'CPU-TEST-DISPLAY-PLACEHOLDER-NOT-A-VIDEO')
        return {'test_only': True}
    monkeypatch.setattr(input_header, 'annotate_video', annotate)
    import training.gpu_gate as gpu_gate
    monkeypatch.setattr(gpu_gate, '_run_nvidia_smi', lambda _args: pytest.fail('CPU recovery must not inspect or execute a GPU'))
    return directory, tmp_path, request


def test_legacy_prepare_preserves_failure_and_does_not_invent_generation_metrics(job):
    directory, root, request = job
    before = (directory / 'status.json').read_bytes()
    raw_sha = sha256(directory / 'raw.mp4')
    report = backend.prepare_postprocess_recovery(directory, root)
    assert report['state'] == 'prepared' and report['new_gpu_seconds'] == 0
    assert report['original_job_elapsed_seconds'] == 3600
    assert 'not model/GPU time' in report['original_job_elapsed_scope']
    assert report['cpu_recovery_elapsed_seconds'] >= 0
    assert (directory / 'status.json').read_bytes() == before
    assert (directory / 'recovery.failed-status.json').read_bytes() == before
    assert sha256(directory / 'raw.mp4') == raw_sha
    receipt = backend.validate_result(directory, request)
    assert receipt['source'] == 'cpu_postprocess_recovery'
    assert receipt['generation'] is receipt['peak_vram_bytes'] is receipt['elapsed_seconds'] is None
    assert receipt['lineage'] is receipt['gpu_snapshot'] is receipt['generated_at_utc'] is None
    assert receipt['missing_generation_metadata_reason']
    assert not (directory / 'generation.json').exists()
    assert backend.prepare_postprocess_recovery(directory, root) == report


def test_recovery_rejects_changed_frozen_input_and_other_job_binding(job):
    directory, root, _ = job
    request_path = directory / 'request.json'
    request_bytes = request_path.read_bytes()
    changed = json.loads(request_bytes)
    changed['job_id'] = 'b' * 32
    write_json(request_path, changed)
    with pytest.raises(ValueError, match='identity/path'):
        backend.prepare_postprocess_recovery(directory, root)
    request_path.write_bytes(request_bytes)
    with (directory / 'actions.npy').open('ab') as stream:
        stream.write(b'CPU test input tampering')
    with pytest.raises(ValueError, match='job input changed'):
        backend.prepare_postprocess_recovery(directory, root)
    assert not (directory / 'inputs.mp4').exists() and not (directory / 'receipt.json').exists()


def test_finalize_requires_service_lock_and_empty_queue_and_keeps_failed_history(job, monkeypatch):
    directory, root, _ = job
    backend.prepare_postprocess_recovery(directory, root)
    @contextmanager
    def occupied(_path):
        raise RuntimeError('demo service is running')
        yield
    monkeypatch.setattr(backend, '_recovery_service_lock', occupied)
    with pytest.raises(RuntimeError, match='service is running'):
        backend.finalize_postprocess_recovery(directory, root)
    assert json.loads((directory / 'status.json').read_text())['status'] == 'failed'
    monkeypatch.setattr(backend, '_recovery_service_lock', lambda _path: nullcontext())
    other = directory.parent / ('b' * 32)
    other.mkdir()
    write_json(other / 'status.json', dict(job_id=other.name, status='running'))
    with pytest.raises(ValueError, match='queued/running'):
        backend.finalize_postprocess_recovery(directory, root)
    write_json(other / 'status.json', dict(job_id=other.name, status='failed'))
    report = backend.finalize_postprocess_recovery(directory, root)
    status = json.loads((directory / 'status.json').read_text())
    history = json.loads((directory / 'recovery.failed-status.json').read_text())
    assert report['state'] == 'finalized' and status['status'] == 'completed'
    assert status['recovery']['new_gpu_seconds'] == 0
    assert history['status'] == 'failed' and history['error'] == 'original annotation error'
    assert json.loads((other / 'status.json').read_text())['status'] == 'failed'
    assert backend.finalize_postprocess_recovery(directory, root) == report


def test_raw_validation_demands_complete_cpu_decode(monkeypatch, tmp_path):
    from training.eval import input_header
    calls = []
    monkeypatch.setattr(input_header, '_probe_video', lambda *args, **kwargs: dict(width=832, height=480, frames=241, fps=16))
    monkeypatch.setattr(backend.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    result = backend._decode_recovery_video(tmp_path / 'raw.mp4')
    command, options = calls[0]
    assert result['complete_software_decode'] is True
    assert command[command.index('-hwaccel') + 1] == 'none' and '-xerror' in command
    assert options['check'] is True and options['timeout'] == 120
    monkeypatch.setattr(input_header, '_probe_video', lambda *args, **kwargs: dict(width=832, height=480, frames=240, fps=16))
    with pytest.raises(ValueError, match='incomplete or invalid'):
        backend._decode_recovery_video(tmp_path / 'raw.mp4')


def test_source_revision_prefers_strict_deployment_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(backend.subprocess, 'check_output', lambda *args, **kwargs: pytest.fail('marker must bypass git'))
    marker = tmp_path / '.source-revision'
    marker.write_text('ABCDEF01' * 5 + '\n', encoding='ascii')
    assert backend._source_revision(tmp_path) == 'abcdef01' * 5
    marker.write_text('not-a-commit', encoding='ascii')
    assert backend._source_revision(tmp_path) == 'unknown'
    marker.write_text('a' * 40 + ' ' * 128 + 'invalid-tail', encoding='ascii')
    assert backend._source_revision(tmp_path) == 'unknown'


def test_source_revision_rejects_parent_repository_head(monkeypatch, tmp_path):
    code_root = tmp_path / 'deployed-code'
    code_root.mkdir()
    repository_root = tmp_path
    calls = []
    def git(command, **kwargs):
        assert kwargs['cwd'] == code_root.resolve() and kwargs['timeout'] == 5
        calls.append(command[-1])
        return str(repository_root) if command[-1] == '--show-toplevel' else 'a' * 40
    monkeypatch.setattr(backend.subprocess, 'check_output', git)
    assert backend._source_revision(code_root) == 'unknown'
    assert calls == ['--show-toplevel']
    repository_root = code_root
    calls.clear()
    assert backend._source_revision(code_root) == 'a' * 40
    assert calls == ['--show-toplevel', 'HEAD']
