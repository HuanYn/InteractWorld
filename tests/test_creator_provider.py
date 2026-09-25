"""CPU-only bridge contract; no model process or GPU is invoked."""
import pytest

from training.creator.providers import CommandProvider


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
