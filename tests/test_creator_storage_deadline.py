"""Bounded cold-disk inventory waits, not relaxed storage/GPU authorization."""
from types import SimpleNamespace

from training.creator import guard
from training.demo.backend import CommandGuard


def test_storage_scan_keeps_a_finite_checked_timeout(monkeypatch, tmp_path):
    calls=[]
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout='12345\t/data/project\n')
    monkeypatch.setattr(guard.subprocess, 'run', run)
    assert guard._directory_bytes(tmp_path)==12345
    assert calls[0][1]['timeout']==90
    assert calls[0][1]['check'] is True
    assert guard.STORAGE_SCAN_SECONDS < guard.RESERVE_RPC_SECONDS < 135
    assert guard.MAX_NEW_STORAGE_BYTES==100_000_000_000


def test_video_reserve_wait_is_longer_but_checks_remain_short(monkeypatch):
    calls=[]
    def run(argv, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout='{"status":"TEST"}')
    monkeypatch.setattr('training.demo.backend.subprocess.run', run)
    client=CommandGuard(SimpleNamespace(guard_command=('test-guard',)))
    for op in ('reserve','check','settle'):
        assert client.call(op)['status']=='TEST'
    assert [call['timeout'] for call in calls]==[135,45,45]
