"""CPU-only accounting/lease tests. Synthetic UUIDs never reach nvidia-smi."""
from contextlib import nullcontext
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys

import pytest

from training.creator import guard


UUID = 'GPU-00000000-0000-0000-0000-000000000002'
UUID_ONE = 'GPU-00000000-0000-0000-0000-000000000001'


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    project = data / 'project'
    model = data / 'model'
    project.mkdir(parents=True)
    model.mkdir()
    (model / 'config.json').write_text('{}', encoding='utf-8')
    authority = data / 'cpu-test-authority.json'
    authority.write_text(json.dumps(dict(status='ACTIVE', project='InterActWorld', until_user_stop=True,
        authorization_id='CPU-SYNTHETIC-NOT-A-USER-GRANT', total_gpu_hours_limit=160,
        total_storage_bytes_limit=400_000_000_000,
        hosts={guard.gpu_gate.SHARED_PROFILE: dict(hostname=socket.gethostname(), gpu_index=2,
                   gpu_uuid=UUID, desktop_memory_max_exclusive=500)})), encoding='utf-8')
    config = guard.GuardConfig(authority=authority, ledger=data / 'state' / 'ledger.json',
        lock_root=data / 'locks', project_root=project, python=Path(sys.executable), model=model,
        prior_gpu_hours_upper=130, max_round_gpu_hours=6, prior_storage_bytes=200_000_000_000,
        storage_root=data, storage_limit=400_000_000_000)
    instance = guard.CreatorGuard(config)
    queries, scans = [], []
    monkeypatch.setattr(guard, '_require_linux', lambda: None)
    monkeypatch.setattr(guard, '_file_lock', lambda _path: nullcontext())
    monkeypatch.setattr(guard, '_deadline', lambda: nullcontext())
    monkeypatch.setattr(instance, '_query', lambda lease: queries.append(lease['gpu_uuid']))
    monkeypatch.setattr(guard.gpu_gate, '_run_nvidia_smi', lambda _args: pytest.fail('no actual GPU query in CPU tests'))
    def directory_bytes(path):
        scans.append(Path(path))
        return sum(item.stat().st_size for item in Path(path).rglob('*') if item.is_file())
    monkeypatch.setattr(guard, '_directory_bytes', directory_bytes)
    return instance, queries, scans


def context(instance, name='job-one', max_seconds=60, request=None):
    directory = instance.config.project_root / 'jobs' / name
    directory.mkdir(parents=True)
    path = directory / 'request.json'
    path.write_text(json.dumps(request or {'job_id': name}), encoding='utf-8')
    return dict(job_id=name, request_sha256=guard._sha(path), job_directory=str(directory),
                project_root=str(instance.config.project_root), max_seconds=max_seconds)


def reserve(instance, ctx):
    return instance.dispatch(dict(operation='reserve', **ctx))


def settle(instance, ctx, lease, **changes):
    return instance.dispatch(dict(operation='settle', **ctx, lease=lease, **{
        'elapsed_seconds': 2, 'outcome': 'completed', 'worker_returncode': 0, **changes}))


def test_reservation_matches_video_backend_and_never_guesses_card(fixture):
    from training.demo.backend import validate_lease
    instance, queries, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    validate_lease(lease, max_seconds=60, job_id=ctx['job_id'], request_sha256=ctx['request_sha256'])
    assert lease['gpu_index'] == 2 and lease['gpu_uuid'] == UUID
    assert lease['profile'] == guard.gpu_gate.SHARED_PROFILE and queries == [UUID]
    assert lease['gpu_hours_reserved'] == 120 / 3600
    assert instance._marker(UUID).is_file()


def test_same_card_model_and_video_cannot_overlap_or_relaunch_same_job(fixture):
    instance, queries, _ = fixture
    first, second = context(instance), context(instance, 'plan-other')
    lease = reserve(instance, first)
    with pytest.raises(guard.GuardError, match='persistent lease'):
        reserve(instance, second)
    with pytest.raises(guard.GuardError, match='never launch it twice'):
        reserve(instance, first)
    assert queries == [UUID]
    settle(instance, first, lease)
    reserve(instance, second)


def test_expired_lease_is_not_automatically_released(fixture, monkeypatch):
    instance, _, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    monkeypatch.setattr(guard.time, 'time', lambda: lease['deadline_unix'] + 1)
    with pytest.raises(guard.GuardError, match='deadline'):
        instance.dispatch(dict(operation='check', **ctx, lease=lease))
    with pytest.raises(guard.GuardError, match='persistent lease'):
        reserve(instance, context(instance, 'another'))
    assert instance._marker(UUID).exists()
    assert instance._ledger()['reservations'][lease['reservation_id']]['status'] == 'reserved'


def test_settlement_checks_actual_idle_is_idempotent_and_accounts_elapsed(fixture):
    instance, queries, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    result = settle(instance, ctx, lease, elapsed_seconds=75)
    assert result['status'] == 'settled' and result['charged_gpu_seconds'] == 105
    assert queries == [UUID, UUID] and not instance._marker(UUID).exists()
    assert settle(instance, ctx, lease) == result
    assert queries == [UUID, UUID]


def test_failed_settlement_retains_reservation_and_marker(fixture, monkeypatch):
    instance, _, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    def busy(_lease):
        raise guard.GuardError('selected GPU still has compute processes')
    monkeypatch.setattr(instance, '_query', busy)
    with pytest.raises(guard.GuardError, match='compute processes'):
        settle(instance, ctx, lease)
    assert instance._marker(UUID).exists()
    assert instance._ledger()['reservations'][lease['reservation_id']]['status'] == 'reserved'


def test_settlement_write_failure_restores_marker_and_retains_open_budget(fixture, monkeypatch):
    instance, _, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    actual_write = guard._write
    def failed_ledger_write(path, value, **kwargs):
        if Path(path) == instance.config.ledger:
            raise OSError('CPU synthetic disk write failure')
        return actual_write(path, value, **kwargs)
    monkeypatch.setattr(guard, '_write', failed_ledger_write)
    with pytest.raises(OSError, match='disk write failure'):
        settle(instance, ctx, lease)
    assert instance._marker(UUID).exists()
    assert instance._ledger()['reservations'][lease['reservation_id']]['status'] == 'reserved'


def test_global_and_round_budgets_include_open_reservations(fixture):
    instance, queries, _ = fixture
    ctx = context(instance)
    for changes, message in (({'prior_gpu_hours_upper': 159.99}, 'global GPU-hour'),
                             ({'max_round_gpu_hours': 0.01}, 'round GPU-hour')):
        alternate = guard.CreatorGuard(replace(instance.config, **changes))
        with pytest.raises(guard.GuardError, match=message):
            reserve(alternate, ctx)
    assert not queries and not instance.config.ledger.exists()


def test_storage_counts_retained_assets_and_caps_added_bytes(fixture, monkeypatch):
    instance, _, _ = fixture
    ctx = context(instance)
    monkeypatch.setattr(guard, '_directory_bytes', lambda _path: 100_000_000_001)
    with pytest.raises(guard.GuardError, match='100GB'):
        reserve(instance, ctx)
    instance = guard.CreatorGuard(replace(instance.config, prior_storage_bytes=350_000_000_000))
    monkeypatch.setattr(guard, '_directory_bytes', lambda _path: 60_000_000_000)
    with pytest.raises(guard.GuardError, match='retained prior'):
        reserve(instance, ctx)


def test_check_scans_only_job_directory_and_rejects_storage_growth(fixture, monkeypatch):
    instance, _, scans = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    scans.clear()
    result = instance.dispatch(dict(operation='check', **ctx, lease=lease))
    assert result['status'] == 'allowed' and scans == [Path(ctx['job_directory'])]
    monkeypatch.setattr(guard, '_directory_bytes', lambda _path: 100_000_000_001)
    with pytest.raises(guard.GuardError, match='100GB'):
        instance.dispatch(dict(operation='check', **ctx, lease=lease))
    assert instance._marker(UUID).exists()


def test_changed_authority_blocks_check_but_allows_verified_cleanup(fixture):
    instance, _, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    authority = guard._read(instance.config.authority)
    authority['status'] = 'PAUSED'
    guard._write(instance.config.authority, authority)
    with pytest.raises(guard.GuardError, match='ACTIVE'):
        instance.dispatch(dict(operation='check', **ctx, lease=lease))
    assert settle(instance, ctx, lease)['status'] == 'settled'


def test_no_authority_means_no_lease_artifacts_or_gpu_check(fixture):
    instance, queries, _ = fixture
    ctx = context(instance)
    authority = guard._read(instance.config.authority)
    authority['hosts'][guard.gpu_gate.SHARED_PROFILE]['gpu_index'] = None
    guard._write(instance.config.authority, authority)
    with pytest.raises(guard.GuardError, match='index is missing'):
        reserve(instance, ctx)
    assert not instance.config.ledger.exists() and not instance.config.lock_root.exists() and not queries


def test_lease_tampering_foreign_marker_and_changed_request_are_rejected(fixture):
    instance, _, _ = fixture
    ctx = context(instance)
    lease = reserve(instance, ctx)
    with pytest.raises(guard.GuardError, match='persistent reservation'):
        instance.dispatch(dict(operation='check', **ctx, lease={**lease, 'gpu_index': 0}))
    marker = instance._marker(UUID)
    guard._write(marker, {'reservation_id': 'someone-else'})
    with pytest.raises(guard.GuardError, match='marker changed'):
        settle(instance, ctx, lease)
    (Path(ctx['job_directory']) / 'request.json').write_text('{"changed":true}')
    with pytest.raises(guard.GuardError, match='request changed'):
        instance.dispatch(dict(operation='check', **ctx, lease=lease))


def test_configuration_change_cannot_reset_the_ledger(fixture):
    instance, _, _ = fixture
    reserve(instance, context(instance))
    changed = guard.CreatorGuard(replace(instance.config, prior_gpu_hours_upper=1))
    with pytest.raises(guard.GuardError, match='reconciliation'):
        changed._ledger()
    with pytest.raises(guard.GuardError, match='six GPU-hours'):
        guard.CreatorGuard(replace(instance.config, max_round_gpu_hours=7))


def test_run_model_uses_shared_reservation_watchdog_offline_data_environment(fixture, monkeypatch):
    instance, queries, _ = fixture
    ctx = context(instance, 'plan-test', request={'kind': 'plan', 'text': 'look left', 'capabilities': {}})
    request = Path(ctx['job_directory']) / 'request.json'
    output = request.with_name('output.json')
    calls = []
    class Process:
        pid = 999999
        returncode = 0
        def poll(self):
            return 0
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        output.write_text('{"status":"clarify","explanation":"CPU fake"}', encoding='utf-8')
        return Process()
    monkeypatch.setattr(guard.shutil, 'which', lambda name: '/usr/bin/timeout')
    monkeypatch.setattr(guard.subprocess, 'Popen', launch)
    monkeypatch.setattr(guard, '_terminate', lambda process: calls.append(('terminated', process.pid)))
    result = guard.run_model(instance, request, output)
    assert result['status'] == 'clarify' and queries == [UUID, UUID]
    command, options = calls[0]
    assert command[:4] == ['/usr/bin/timeout', '--signal=TERM', '--kill-after=2s', '480s']
    assert 'training.creator.model_worker' in command and options['start_new_session'] is True
    environment = options['env']
    assert environment['CUDA_VISIBLE_DEVICES'] == UUID and environment['HF_HUB_OFFLINE'] == '1'
    assert environment['WANDB_MODE'] == 'disabled'
    assert Path(environment['TMPDIR']) == request.parent
    assert Path(environment['HF_HOME']).is_relative_to(request.parent)
    assert calls[-1] == ('terminated', Process.pid) and not instance._marker(UUID).exists()
    record = next(iter(instance._ledger()['reservations'].values()))
    assert record['context']['max_seconds'] == 480 and record['status'] == 'settled'


def test_parent_sigterm_cleans_owned_model_group_before_settlement(fixture, monkeypatch):
    instance, _, _ = fixture
    ctx = context(instance, 'plan-interrupted', request={'kind': 'plan', 'text': 'look left', 'capabilities': {}})
    request = Path(ctx['job_directory']) / 'request.json'
    cleaned = []
    class Process:
        pid = 999998
        returncode = -15
        def poll(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(guard.shutil, 'which', lambda name: '/usr/bin/timeout')
    monkeypatch.setattr(guard.subprocess, 'Popen', lambda *args, **kwargs: Process())
    monkeypatch.setattr(guard, '_terminate', lambda process: cleaned.append(process.pid))
    actual_query = instance._query
    def query_after_cleanup(lease):
        if instance.config.ledger.exists():
            assert cleaned == [Process.pid]
        return actual_query(lease)
    monkeypatch.setattr(instance, '_query', query_after_cleanup)
    with pytest.raises(guard.GuardError, match='interrupted'):
        guard.run_model(instance, request, request.with_name('output.json'))
    record = next(iter(instance._ledger()['reservations'].values()))
    assert cleaned == [Process.pid] and record['settlement']['outcome'] == 'interrupted'
    assert not instance._marker(UUID).exists()


def test_cli_denies_non_json_and_does_not_return_fake_success(fixture, monkeypatch, capsys):
    instance, _, _ = fixture
    path = instance.config.storage_root / 'private-config.json'
    guard._write(path, instance.config.binding())
    monkeypatch.setattr(guard.sys, 'stdin', io.StringIO('not json'))
    assert guard.main(['--config', str(path)]) == 1
    result = capsys.readouterr()
    assert not result.out and json.loads(result.err)['status'] == 'denied'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='regression covers Linux venv interpreter symlinks')
def test_config_load_preserves_venv_python_symlink_path(fixture):
    instance, _, _ = fixture
    venv_python = instance.config.storage_root / 'venv' / 'bin' / 'python'
    venv_python.parent.mkdir(parents=True)
    interpreter_target = Path(sys.executable).resolve()
    venv_python.symlink_to(interpreter_target)
    model_alias = instance.config.storage_root / 'model-link'
    model_alias.symlink_to(instance.config.model, target_is_directory=True)
    path = instance.config.storage_root / 'venv-guard-config.json'
    guard._write(path, {**instance.config.binding(), 'python': str(venv_python), 'model': str(model_alias)})

    loaded = guard.GuardConfig.load(path)

    assert loaded.python == venv_python.absolute()
    assert loaded.python != interpreter_target
    assert loaded.python.is_symlink() and loaded.python.resolve() == interpreter_target
    assert loaded.binding()['python'] == str(venv_python.absolute())
    # Other storage paths retain normal canonical resolution.
    assert loaded.model == instance.config.model.resolve() and loaded.model != model_alias


def multi_card_guards(instance, monkeypatch, **changes):
    authority = guard._read(instance.config.authority)
    authority['hosts'][guard.gpu_gate.SHARED_PROFILE] = dict(hostname=socket.gethostname(), gpus=[
        dict(gpu_index=1, gpu_uuid=UUID_ONE, desktop_memory_max_exclusive=500),
        dict(gpu_index=2, gpu_uuid=UUID, desktop_memory_max_exclusive=500)])
    guard._write(instance.config.authority, authority)
    result = [guard.CreatorGuard(replace(instance.config, gpu_index=index, **changes)) for index in (1, 2)]
    for item in result:
        monkeypatch.setattr(item, '_query', lambda lease: None)
    return result


def test_two_authorized_cards_share_ledger_and_have_independent_markers(fixture, monkeypatch):
    from training.demo.backend import validate_lease
    instance, _, _ = fixture
    video, model = multi_card_guards(instance, monkeypatch)
    first, second = context(video, 'video-card-one'), context(model, 'model-card-two')
    lease_one, lease_two = reserve(video, first), reserve(model, second)
    assert video.config.binding() == model.config.binding()
    assert video._marker(UUID_ONE).exists() and model._marker(UUID).exists()
    assert len(video._ledger()['reservations']) == 2
    assert lease_two['global_gpu_hours_upper'] == pytest.approx(130 + 240 / 3600)
    for lease, ctx in ((lease_one, first), (lease_two, second)):
        validate_lease(lease, max_seconds=60, job_id=ctx['job_id'], request_sha256=ctx['request_sha256'])
    with pytest.raises(ValueError, match='not authorized'):
        validate_lease({**lease_one, 'gpu_uuid': UUID}, max_seconds=60)
    with pytest.raises(guard.GuardError, match='another configured GPU'):
        model.dispatch(dict(operation='check', **first, lease=lease_one))
    settle(video, first, lease_one)
    assert not video._marker(UUID_ONE).exists() and model._marker(UUID).exists()


def test_multicard_requires_explicit_selection_and_allows_no_budget_reset(fixture, monkeypatch):
    instance, _, _ = fixture
    video, model = multi_card_guards(instance, monkeypatch, max_round_gpu_hours=.06)
    with pytest.raises(guard.GuardError, match='explicit configured gpu_index'):
        instance._authority()
    with pytest.raises(guard.GuardError, match='not authorized'):
        guard.CreatorGuard(replace(video.config, gpu_index=0))._authority()
    reserve(video, context(video, 'first-card'))
    with pytest.raises(guard.GuardError, match='round GPU-hour budget'):
        reserve(model, context(model, 'second-card'))
    assert not model._marker(UUID).exists()


def test_concurrent_storage_growth_is_summed_once_across_cards(fixture, monkeypatch):
    instance, _, _ = fixture
    video, model = multi_card_guards(instance, monkeypatch)
    first, second = context(video, 'growing-video'), context(model, 'growing-model')
    sizes = {instance.config.storage_root: 100, Path(first['job_directory']): 10, Path(second['job_directory']): 20}
    monkeypatch.setattr(guard, '_directory_bytes', lambda path: sizes[Path(path)])
    one, two = reserve(video, first), reserve(model, second)
    baseline = video._ledger()['storage_bytes_upper']
    sizes[Path(first['job_directory'])] += 30
    sizes[Path(second['job_directory'])] += 50
    result = video.dispatch(dict(operation='check', **first, lease=one))
    assert result['global_storage_bytes_upper'] == instance.config.prior_storage_bytes + baseline + 80
    model.dispatch(dict(operation='check', **second, lease=two))
    assert model._ledger()['storage_bytes_upper'] == baseline + 80
    # Each separate 20-byte delta fits, but their 40-byte sum exceeds the cap.
    ledger = video._ledger()
    ledger['storage_bytes_upper'] = guard.MAX_NEW_STORAGE_BYTES - 30
    guard._write(video.config.ledger, ledger)
    sizes[Path(first['job_directory'])] += 20
    sizes[Path(second['job_directory'])] += 20
    with pytest.raises(guard.GuardError, match='100GB'):
        model.dispatch(dict(operation='check', **second, lease=two))
    assert video._marker(UUID_ONE).exists() and model._marker(UUID).exists()


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='real flock requires Linux; no GPU is used')
def test_real_budget_flock_refuses_concurrent_holder(tmp_path):
    import fcntl
    path = tmp_path / 'budget.lock'
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(guard.GuardError, match='locked'):
            with guard._file_lock(path):
                pytest.fail('a second open file description acquired the same lock')
