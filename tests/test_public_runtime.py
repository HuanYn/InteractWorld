"""CPU-only public entry boundaries; synthetic files are not model evidence."""
import hashlib
import json
from pathlib import Path
import pytest


def test_public_runtime_never_launches(monkeypatch):
    from interactive.runtime_control import start_models,service_status
    monkeypatch.setenv('IW_MODEL_START_ENABLED','1')
    monkeypatch.delenv('IW_CLOSED_DEMO_HANDOFF',raising=False)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**kw:pytest.fail('unexpected process launch'))
    assert 'HTTP' in start_models()
    assert service_status('mock',get=lambda *a,**kw:pytest.fail('unexpected HTTP request'))[0]


def test_explicit_read_only_does_not_consult_private_files(monkeypatch):
    from interactive.closed_handoff import closed_reason,verify_closed_handoff
    monkeypatch.setenv('IW_CLOSED_DEMO_HANDOFF','read_only')
    monkeypatch.setattr(Path,'read_text',lambda *a,**kw:pytest.fail('unexpected file read'))
    assert verify_closed_handoff()['new_gpu_authority'] is False
    assert closed_reason()
    monkeypatch.setenv('IW_CLOSED_DEMO_HANDOFF','unknown')
    assert closed_reason()
    with pytest.raises(ValueError):verify_closed_handoff()


def test_private_scope_validator_rejects_without_ui_or_model_import(monkeypatch):
    from interactive.runtime_control import validate_public_environment
    monkeypatch.setenv('IW_DAY1_CONTRACT_CONTINUATION_SCOPE','not-a-public-scope')
    with pytest.raises(ValueError,match='Private deployment/resume'):
        validate_public_environment()


def test_asset_mapping_requires_actual_explicit_original_root(monkeypatch,tmp_path):
    from interactive.wan_assets import runtime_path
    monkeypatch.delenv('IW_ASSET_ORIGINAL_ROOT',raising=False)
    with pytest.raises(ValueError,match='IW_ASSET_ORIGINAL_ROOT'):
        runtime_path(tmp_path/'model.bin',tmp_path)
    origin=tmp_path/'source';snapshot=tmp_path/'snapshot'
    monkeypatch.setenv('IW_ASSET_ORIGINAL_ROOT',str(origin))
    assert runtime_path(origin/'model.bin',snapshot)==(snapshot/'model.bin').resolve()
    with pytest.raises(ValueError,match='escapes'):
        runtime_path(origin/'..'/'outside.bin',snapshot)


def test_lab_assets_require_explicit_rollout_before_model_import(monkeypatch,tmp_path):
    from interactive.lab_video import LaboratoryVideoModel
    monkeypatch.delenv('IW_ACTION_ROLLOUT',raising=False)
    with pytest.raises(ValueError,match='IW_ACTION_ROLLOUT'):
        LaboratoryVideoModel(tmp_path,tmp_path/'absent-cache.pt')


def test_authored_manifest_pins_are_real_and_not_inferred(monkeypatch,tmp_path):
    from interactive.declared_room_cut import load_room_manifest
    asset={'provenance':'synthetic CPU test, not a visual result'}
    for field,digest in [('path','sha256'),('source_path','source_sha256'),('source_receipt','source_receipt_sha256')]:
        path=tmp_path/field;path.write_bytes(field.encode())
        asset[field]=str(path);asset[digest]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'version':'declared_room_cut_v1','enabled':True,
        'story_profile':'spatial_v1','assets':{'storage_room':asset}}),encoding='utf-8')
    monkeypatch.delenv('IW_ROOM_MANIFEST_SHA256',raising=False)
    with pytest.raises(ValueError,match='IW_ROOM_MANIFEST_SHA256'):load_room_manifest(manifest)
    monkeypatch.setenv('IW_ROOM_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert load_room_manifest(manifest)[1]['assets']['storage_room']==asset
    Path(asset['source_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):load_room_manifest(manifest)
