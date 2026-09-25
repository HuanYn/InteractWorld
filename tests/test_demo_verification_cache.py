"""CPU-only persistent SHA-cache integrity and invalidation contracts."""
import json
from pathlib import Path

import pytest

from training.demo import relocation


def test_persistent_hash_reuse_and_stat_identity_invalidation(tmp_path, monkeypatch):
    artifact, cache = tmp_path / 'weights.bin', tmp_path / 'runtime' / 'verified.json'
    artifact.write_bytes(b'original immutable bytes')
    actual_hash, calls = relocation.sha256, []
    def counted(path):
        calls.append(Path(path))
        return actual_hash(path)
    monkeypatch.setattr(relocation, 'sha256', counted)
    first = relocation.ArtifactHashes(cache)(artifact)
    assert relocation.ArtifactHashes(cache)(artifact) == first and calls == [artifact]
    entry = json.loads(cache.read_text())['artifacts'][str(artifact.resolve())]
    stat = artifact.stat()
    assert entry['identity'] == [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    artifact.write_bytes(b'changed file bytes with a different size')
    assert relocation.ArtifactHashes(cache)(artifact) == actual_hash(artifact) != first
    assert calls == [artifact, artifact]


@pytest.mark.parametrize('broken', ['{truncated', '{"version":1,"artifacts":{"/invalid":{"identity":[],"sha256":"fake"}}}'])
def test_bad_cache_is_replaced_only_after_real_hashing(tmp_path, monkeypatch, broken):
    artifact, cache = tmp_path / 'weights.bin', tmp_path / 'verified.json'
    artifact.write_bytes(b'actual artifact bytes')
    cache.write_text(broken)
    expected, calls = relocation.sha256(artifact), []
    def hashed(path):
        calls.append(path)
        return expected
    monkeypatch.setattr(relocation, 'sha256', hashed)
    assert relocation.ArtifactHashes(cache)(artifact) == expected and calls == [artifact]
    assert json.loads(cache.read_text())['artifacts'][str(artifact)]['sha256'] == expected


def test_changed_during_hash_is_rejected_and_never_cached(tmp_path, monkeypatch):
    artifact, cache = tmp_path / 'weights.bin', tmp_path / 'verified.json'
    artifact.write_bytes(b'before')
    actual_hash = relocation.sha256
    def changed(path):
        digest = actual_hash(path)
        path.write_bytes(b'changed during the read')
        return digest
    monkeypatch.setattr(relocation, 'sha256', changed)
    with pytest.raises(ValueError, match='changed during hashing'):
        relocation.ArtifactHashes(cache)(artifact)
    assert not cache.exists()


def test_cache_cannot_overwrite_protected_original_artifacts(tmp_path):
    artifact = tmp_path / 'original-receipt.json'
    artifact.write_text('{"original": true}')
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match='must not overwrite'):
        relocation.ArtifactHashes(artifact, protected_paths=[artifact])
    assert artifact.read_bytes() == before
