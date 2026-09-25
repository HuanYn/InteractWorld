"""CPU-only demo/API contracts. Test executors are never production backends."""
from dataclasses import replace
import copy
import http.client
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from training.demo.backend import CommandGuard, validate_lease, validate_result
from training.demo.contracts import ADAPTER, Catalog, Deployment, action_array, sha256, write_json
from training.demo.service import DemoService, QueueFullError, make_server
from training.gpu_gate import DEDICATED_PROFILE, SHARED_PROFILE


def fixture(tmp_path, *, enabled=False, scene_count=1):
    initial = tmp_path / 'source.npy'
    np.save(initial, np.arange(12, dtype=np.uint8).reshape(2, 2, 3), allow_pickle=False)
    prompt_receipt = tmp_path / 'prompt.pt.receipt.json'
    write_json(prompt_receipt, dict(prompt_policy='scene_static_only_v1', episodes={
        'episode-source': {'split': 'dev', 'prompt': 'A stable stone courtyard.'}}))
    config = dict(version=1, run_id='test-config', adapter_factory=ADAPTER,
                  geometry=dict(width=2, height=2, fps=16, num_chunks=20, latent_frames_per_chunk=3,
                                rgb_frames_per_latent=4, total_rgb_frames=241),
                  late_anchors=[{'frame_index': 240, 'weight': 1}],
                  lineage=dict(expected_stage='causal_teacher_forcing_v1', checkpoint_path=str(tmp_path / 'step-0001075.pt'),
                               checkpoint_sha256='a' * 64, artifact_paths={'prompt_cache': str(tmp_path / 'prompt.pt'),
                                                                        'prompt_cache_receipt': str(prompt_receipt)}),
                  scenes=[dict(scene_id='courtyard', source_episode_id='episode-source',
                               initial_frame_path=str(initial), reference_frames_path=str(tmp_path / 'never-read-ground-truth.npz'),
                               prompt='A stable stone courtyard.', seed=42,
                               action_segments=[{'frames': 240, 'keys': ['W']}])])
    path = tmp_path / 'rollout.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    deployment = Deployment(path, tmp_path, tmp_path / 'jobs', guard_command=(sys.executable, 'operator-only-fixture') if enabled else (), scene_count=scene_count)
    return deployment, config


def request():
    return {'scene_id': 'courtyard', 'seed': 43,
            'action_segments': [{'frames': 120, 'keys': ['W', 'L']}, {'frames': 120, 'keys': []}]}


def test_freezes_actual_initial_actions_seed_and_readonly_prompt(tmp_path):
    deployment, _ = fixture(tmp_path)
    catalog = Catalog(deployment)
    original = request()
    output = deployment.jobs_root / ('a' * 32)
    receipt = catalog.freeze(original, output, 'a' * 32)
    original['action_segments'][0]['keys'] = ['A']
    actual = np.load(output / 'actions.npy', allow_pickle=False)
    assert actual.shape == (240, 8) and actual.dtype == np.float32
    assert actual[0].tolist() == [1, 0, 0, 0, 0, 0, 0, 1]
    assert np.array_equal(np.load(output / 'initial.npy'), np.load(tmp_path / 'source.npy'))
    frozen = yaml.safe_load((output / 'rollout.yaml').read_text(encoding='utf-8'))
    assert frozen['scenes'][0]['prompt'] == 'A stable stone courtyard.'
    assert frozen['scenes'][0]['seed'] == 43
    assert frozen['scenes'][0]['action_segments'][0]['keys'] == ['W', 'L']
    assert receipt['ground_truth_future_used'] is False and receipt['prompt_editing'] is False
    assert receipt['actions_sha256'] == sha256(output / 'actions.npy')
    assert not (tmp_path / 'never-read-ground-truth.npz').exists()


@pytest.mark.parametrize('segments', [[], [{'frames': 239, 'keys': []}], [{'frames': True, 'keys': []}],
    [{'frames': 240, 'keys': ['W', 'S']}], [{'frames': 240, 'keys': ['ArrowUp']}],
    [{'frames': 240, 'keys': ['W', 'W']}], [{'frames': 241, 'keys': []}]])
def test_rejects_malformed_or_opposing_actions(segments):
    with pytest.raises(ValueError):
        action_array(segments)


@pytest.mark.parametrize('extra', ['prompt', 'gpu_uuid', 'command', 'checkpoint_path', 'initial_frame_path',
                                  'method', 'inference_mode', 'adapter_factory'])
def test_browser_cannot_change_operator_fields(tmp_path, extra):
    deployment, _ = fixture(tmp_path)
    body = request()
    body[extra] = 'forbidden'
    with pytest.raises(ValueError, match='operator-only'):
        Catalog(deployment).freeze(body, deployment.jobs_root / 'unused', 'unused')
    assert not (deployment.jobs_root / 'unused').exists()


def test_real_stage_and_prompt_binding_are_required(tmp_path):
    deployment, raw = fixture(tmp_path)
    for key, value, message in [('expected_stage', 'action_teacher_lora_v1', 'concrete self-trained adapter'),
                                 ('checkpoint_sha256', None, 'exact self-trained'),
                                 ('checkpoint_path', str(tmp_path / 'best.pt'), 'immutable')]:
        bad = copy.deepcopy(raw)
        bad['lineage'][key] = value
        deployment.rollout_config.write_text(yaml.safe_dump(bad), encoding='utf-8')
        with pytest.raises(ValueError, match=message):
            Catalog(deployment)
    raw['scenes'][0]['prompt'] = 'Changed only in UI, not really encoded.'
    deployment.rollout_config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='actual static embedding'):
        Catalog(deployment)


def test_default_disabled_and_queue_bounded(tmp_path):
    deployment, _ = fixture(tmp_path)
    service = DemoService(deployment, start_worker=False)
    with pytest.raises(ValueError, match='门禁'):
        service.submit(request())
    with pytest.raises(ValueError, match='disabled'):
        CommandGuard(deployment)
    service.close()
    enabled = replace(deployment, guard_command=(sys.executable,), max_pending=1)
    service = DemoService(enabled, start_worker=False)
    first = service.submit(request())
    assert first['status'] == 'queued'
    with pytest.raises(QueueFullError):
        service.submit(request())
    assert len(service.jobs) == 1
    service.close()


def test_worker_serial_failure_and_restart_no_replay(tmp_path):
    deployment, _ = fixture(tmp_path, enabled=True)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def explicit_cpu_test_executor(directory, stop_event):
        calls.append(directory.name)
        entered.set()
        release.wait(2)
        raise RuntimeError('intentional CPU fixture failure, no model/GPU')

    service = DemoService(deployment, executor=explicit_cpu_test_executor)
    first = service.submit(request())
    assert entered.wait(2)
    second = service.submit(request())
    assert service.jobs[first['job_id']]['status'] == 'running'
    assert service.jobs[second['job_id']]['status'] == 'queued'
    assert calls == [first['job_id']]
    release.set()
    service.pending.join()
    assert all(job['status'] == 'failed' for job in service.list_jobs())
    service.close()
    # A persisted unfinished task is failed, not replayed on restart.
    path = deployment.jobs_root / second['job_id'] / 'status.json'
    saved = json.loads(path.read_text())
    saved['status'] = 'running'
    write_json(path, saved)
    restarted = DemoService(deployment, start_worker=False)
    assert restarted.jobs[second['job_id']]['status'] == 'failed'
    assert restarted.pending.empty()
    restarted.close()


def test_local_deployment_and_path_guards(tmp_path):
    deployment, _ = fixture(tmp_path)
    for invalid in [replace(deployment, host='0.0.0.0'), replace(deployment, jobs_root=tmp_path.parent / 'escape'),
                    replace(deployment, scene_count=4), replace(deployment, guard_command=('sh', '-c', 'anything'))]:
        with pytest.raises(ValueError):
            invalid.validate()


def test_http_same_origin_csrf_preview_and_path_safety(tmp_path):
    deployment, _ = fixture(tmp_path, enabled=True)
    service = DemoService(deployment, start_worker=False)
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def fetch(path, method='GET', body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        result = response.status, data, dict(response.getheaders())
        conn.close()
        return result

    try:
        status, body, headers = fetch('/api/config')
        assert status == 200 and json.loads(body)['scenes'][0]['prompt'] == 'A stable stone courtyard.'
        assert 'Access-Control-Allow-Origin' not in headers and headers['X-Frame-Options'] == 'DENY'
        token = json.loads(body)['csrf_token']
        assert fetch('/api/config', headers={'Host': f'evil.invalid:{port}'})[0] == 400
        assert fetch('/api/scenes/courtyard/initial.png')[1].startswith(b'\x89PNG')
        assert fetch('/../../source.npy')[0] == 404
        auth_headers = {'Content-Type': 'application/json', 'Origin': f'http://127.0.0.1:{port}',
                        'X-InterActWorld-CSRF': token}
        for altered in [{}, {**auth_headers, 'Origin': 'https://attacker.invalid'},
                        {**auth_headers, 'X-InterActWorld-CSRF': 'wrong'}]:
            assert fetch('/api/jobs', 'POST', json.dumps(request()), altered)[0] == 400
        status, body, _ = fetch('/api/jobs', 'POST', json.dumps(request()), auth_headers)
        assert status == 202
        job_id = json.loads(body)['job_id']
        assert fetch(f'/api/jobs/{job_id}/files/lease.json')[0] == 404
        assert fetch(f'/api/jobs/{job_id}/files/raw.mp4')[0] == 400
        assert fetch('/api/jobs')[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def lease_fixture(tmp_path, *, profile=DEDICATED_PROFILE, storage_limit=300_000_000_000):
    host = dict(hostname=socket.gethostname(), gpu_uuid='GPU-00000000-0000-0000-0000-000000000001',
                gpu_index=2 if profile == SHARED_PROFILE else 0, desktop_memory_max_exclusive=500)
    path = tmp_path / 'authority.json'
    write_json(path, dict(status='ACTIVE', project='InterActWorld', authorization_id='test-standing',
                         until_user_stop=True, total_gpu_hours_limit=160, total_storage_bytes_limit=storage_limit,
                         hosts={profile: host}))
    lease = dict(status='reserved', reservation_id='test-reservation', authorization_id='test-standing',
                 authorization_record=str(path), authorization_sha256=sha256(path), profile=profile,
                 **{key: host[key] for key in ('gpu_uuid', 'gpu_index', 'desktop_memory_max_exclusive')},
                 gpu_hours_reserved=1, global_gpu_hours_upper=120, global_storage_bytes_upper=200_000_000_000,
                 start_before_unix=1600)
    return lease, path


def test_authorization_budget_lease_fail_closed(tmp_path):
    lease, path = lease_fixture(tmp_path)
    validate_lease(lease, max_seconds=900, now=1000)
    with pytest.raises(ValueError, match='another job'):
        validate_lease(lease, max_seconds=900, now=1000, job_id='new-job', request_sha256='a' * 64)
    for key, value in [('status', 'denied'), ('gpu_hours_reserved', .01), ('global_gpu_hours_upper', 161),
                       ('global_storage_bytes_upper', 300_000_000_001), ('start_before_unix', 999),
                       ('gpu_uuid', 'wrong-host-gpu')]:
        with pytest.raises(ValueError):
            validate_lease({**lease, key: value}, max_seconds=900, now=1000)
    authority = json.loads(path.read_text())
    authority['status'] = 'PAUSED'
    write_json(path, authority)
    with pytest.raises(ValueError, match='revoked'):
        validate_lease(lease, max_seconds=900, now=1000)


def test_shared_authority_binds_profile_card_threshold_and_explicit_storage_cap(tmp_path):
    lease, path = lease_fixture(tmp_path, profile=SHARED_PROFILE, storage_limit=400_000_000_000)
    lease['global_storage_bytes_upper'] = 400_000_000_000
    validate_lease(lease, max_seconds=900, now=1000)
    for key, value in [('profile', DEDICATED_PROFILE), ('profile', 'choose_any_free_gpu'),
                       ('gpu_index', 0), ('gpu_index', True), ('gpu_index', -1),
                       ('gpu_uuid', 'GPU-11111111-1111-1111-1111-111111111111'),
                       ('desktop_memory_max_exclusive', 501), ('global_storage_bytes_upper', 400_000_000_001),
                       ('global_gpu_hours_upper', 160.01)]:
        with pytest.raises(ValueError):
            validate_lease({**lease, key: value}, max_seconds=900, now=1000)
    authority = json.loads(path.read_text())
    authority['total_storage_bytes_limit'] = 300_000_000_000
    write_json(path, authority)
    lease['authorization_sha256'] = sha256(path)
    with pytest.raises(ValueError, match='storage'):
        validate_lease(lease, max_seconds=900, now=1000)
    authority['total_storage_bytes_limit'] = 500_000_000_000
    write_json(path, authority)
    lease['authorization_sha256'] = sha256(path)
    with pytest.raises(ValueError, match='budget scope'):
        validate_lease(lease, max_seconds=900, now=1000)


def test_dedicated_authority_still_rejects_nonzero_card_even_if_authority_lists_it(tmp_path):
    lease, path = lease_fixture(tmp_path)
    authority = json.loads(path.read_text())
    authority['hosts'][DEDICATED_PROFILE]['gpu_index'] = 2
    write_json(path, authority)
    lease.update(gpu_index=2, authorization_sha256=sha256(path))
    with pytest.raises(ValueError, match='GPU0 only'):
        validate_lease(lease, max_seconds=900, now=1000)


def test_600gb_lease_requires_explicit_active_600gb_authority(tmp_path):
    lease, path = lease_fixture(tmp_path, profile=SHARED_PROFILE, storage_limit=600_000_000_000)
    lease['global_storage_bytes_upper'] = 600_000_000_000
    validate_lease(lease, max_seconds=900, now=1000)
    with pytest.raises(ValueError, match='storage'):
        validate_lease({**lease, 'global_storage_bytes_upper': 600_000_000_001}, max_seconds=900, now=1000)
    with pytest.raises(ValueError, match='160 GPU-hours'):
        validate_lease({**lease, 'global_gpu_hours_upper': 160.01}, max_seconds=900, now=1000)
    authority = json.loads(path.read_text())
    authority['total_storage_bytes_limit'] = 400_000_000_000
    write_json(path, authority)
    lease['authorization_sha256'] = sha256(path)
    with pytest.raises(ValueError, match='storage'):
        validate_lease(lease, max_seconds=900, now=1000)


def test_new_output_must_match_this_job_and_inputs(tmp_path):
    deployment, _ = fixture(tmp_path)
    directory = deployment.jobs_root / ('f' * 32)
    frozen = Catalog(deployment).freeze(request(), directory, directory.name)
    write_json(directory / 'receipt.json', {'outcome': 'completed', 'job_id': 'an-old-job'})
    with pytest.raises(ValueError, match='another'):
        validate_result(directory, frozen)


@pytest.mark.parametrize('profile', [DEDICATED_PROFILE, SHARED_PROFILE])
def test_real_worker_routes_frozen_tensor_to_existing_rollout_and_header(tmp_path, monkeypatch, profile):
    """CPU wiring test only; dummy writer bytes are never exposed by production."""
    import torch
    import training.demo.backend as backend
    import training.eval.rollout15s as rollout
    import training.eval.wan_causal_adapter as wan
    import training.eval.input_header as header
    import training.gpu_gate as gate

    deployment, _ = fixture(tmp_path)
    directory = deployment.jobs_root / ('b' * 32)
    frozen = Catalog(deployment).freeze(request(), directory, directory.name)
    lease, _ = lease_fixture(tmp_path, profile=profile)
    write_json(directory / 'lease.json', lease)
    raw = yaml.safe_load((directory / 'rollout.yaml').read_text(encoding='utf-8'))
    config = SimpleNamespace(adapter_factory=ADAPTER, scenes=[SimpleNamespace(**raw['scenes'][0])],
                             lineage=SimpleNamespace(**raw['lineage'], expected_base_model_path='unit-test-base'))
    seen = []
    monkeypatch.setattr(backend, 'validate_lease', lambda *a, **k: seen.append('lease'))
    monkeypatch.setattr(rollout, 'load_rollout_config', lambda path: config)
    monkeypatch.setattr(rollout, 'verify_checkpoint_lineage', lambda value: {'stage': 'cpu-wiring-fixture'})
    def physical_gate(**kwargs):
        assert kwargs == dict(confirmed_index=lease['gpu_index'], confirmed_uuid=lease['gpu_uuid'],
                              profile=profile, memory_max_exclusive=500)
        seen.append('physical-gate')
        return SimpleNamespace(as_dict=lambda: {})

    selected_gate = 'query_shared_gpu' if profile == SHARED_PROFILE else 'query_dedicated_gpu'
    other_gate = 'query_dedicated_gpu' if profile == SHARED_PROFILE else 'query_shared_gpu'
    monkeypatch.setattr(gate, selected_gate, physical_gate)
    monkeypatch.setattr(gate, other_gate, lambda **kwargs: pytest.fail('worker crossed allocation profiles'))
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(wan, 'create_wan_causal_adapter', lambda **kwargs: (seen.append('real-factory-interface') or object()))

    def cpu_rollout(**kwargs):
        assert kwargs['scene'].prompt == 'A stable stone courtyard.'
        assert kwargs['scene'].seed == 43
        np.testing.assert_array_equal(kwargs['actions'], np.load(directory / 'actions.npy'))
        assert kwargs['actions'].dtype == np.float32
        np.testing.assert_array_equal(kwargs['initial'], np.load(directory / 'initial.npy'))
        kwargs['output_path'].write_bytes(b'CPU-WIRING-TEST-NOT-A-VIDEO')
        seen.append('rollout-actual-actions')

    def cpu_header(source, output, **kwargs):
        assert kwargs['prompt'] == 'A stable stone courtyard.'
        np.testing.assert_array_equal(kwargs['actions'], np.load(directory / 'actions.npy'))
        saved = json.loads((directory / 'generation.json').read_text())
        assert saved['outcome'] == 'raw_generated' and saved['job_id'] == directory.name
        assert saved['generation']['method'] == 'causal_rollout15s'
        assert saved['videos']['raw.mp4'] == sha256(source) and saved['peak_vram_bytes'] == 0
        output.write_bytes(b'CPU-WIRING-TEST-INPUTS-NOT-A-VIDEO')
        return {'test_only': True}

    monkeypatch.setattr(rollout, '_run_variant', cpu_rollout)
    monkeypatch.setattr(header, 'annotate_video', cpu_header)
    result = backend.run_model_job(directory, deployment.project_root, 900)
    assert seen.index('physical-gate') < seen.index('real-factory-interface') < seen.index('rollout-actual-actions')
    assert result['job_id'] == directory.name and result['ground_truth_future_used'] is False
    assert result['actions_sha256'] == frozen['actions_sha256']
    assert validate_result(directory, frozen)['total_rgb_frames'] == 241


def action_fixture(tmp_path, monkeypatch):
    """Tiny actual checkpoint/receipt files for CPU lineage and UI wiring tests."""
    import hashlib
    import torch
    from training.config import ActionTeacherConfig
    from training.paths import PINNED_MODEL_DIR
    from train_action_teacher import _manifest_hashes
    from training.demo import action_backend
    from training.demo.contracts import ACTION_ADAPTER, ACTION_METHOD, ACTION_STAGE
    root = tmp_path
    manifests, features = root / 'data/manifests', root / 'data/features'
    manifests.mkdir(parents=True)
    features.mkdir()
    manifest, index = manifests / 'train.jsonl', features / 'train.features.jsonl'
    row = {'episode_id': 'episode-source', 'split': 'dev'}
    manifest.write_text(json.dumps(row) + '\n')
    index.write_text(json.dumps(row) + '\n')
    index_receipt = index.with_suffix('.jsonl.receipt.json')
    write_json(index_receipt, dict(schema_version=1, index=str(index), manifest=str(manifest),
                                   index_sha256=sha256(index), manifest_sha256=sha256(manifest)))
    cache = features / 'static.pt'
    torch.save(dict(schema_version=1, kind='scene_static_prompt_cache',
                    prompt_embeds={'episode-source': torch.ones(2, 4096, dtype=torch.bfloat16)}), cache)
    prompt = 'A stable stone courtyard.'
    write_json(cache.with_suffix('.pt.receipt.json'), dict(schema_version=1, kind='scene_static_prompt_cache',
        prompt_policy='scene_static_only_v1', cache_path=str(cache), cache_sha256=sha256(cache),
        manifest_sha256=sha256(manifest), feature_index_sha256=sha256(index), feature_receipt_sha256=sha256(index_receipt),
        encoder={'kind': 'CPU-test-only'}, episodes={'episode-source': dict(split='dev', prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())}))
    base = root / 'models' / PINNED_MODEL_DIR
    base.mkdir(parents=True)
    vae = base / 'Wan2.2_VAE.pth'
    vae.write_bytes(b'CPU-TEST-NOT-A-VAE')
    monkeypatch.setattr(action_backend, 'VAE_SHA256', sha256(vae))
    config = ActionTeacherConfig()
    config.model.base_model_path = str(base)
    config.model.action_scale = .03
    config.data.manifest_path = str(manifest)
    config.data.prompt_cache_path = str(cache)
    config.training.max_steps = 780
    config.training.output_dir = str(root / 'run')
    training = root / 'training.yaml'
    training.write_text(yaml.safe_dump(config.to_dict()))
    saved = config.to_dict()
    saved['training']['max_steps'] = 1040
    checkpoint = root / 'step-0001040.pt'
    torch.save(dict(format_version=1, stage=ACTION_STAGE, step=1040, config=saved,
                    manifest_hashes=_manifest_hashes(config, training), initialization={'mode': 'CPU-fixture'},
                    trainable_model={'model.test.lora_a.weight': torch.ones(1),
                                     'model.act_control_adapter.weight': torch.ones(1)}), checkpoint)
    initial = root / 'source.npy'
    np.save(initial, np.zeros((480, 832, 3), dtype=np.uint8))
    actions = root / 'preset.npy'
    np.save(actions, np.ones((240, 8), dtype=np.float32) * np.array([1, 0, 0, 0, 0, 0, 0, 0], np.float32))
    return dict(training_config=training, checkpoint=checkpoint, checkpoint_sha256=sha256(checkpoint),
                initial_frame=initial, episode_id='episode-source', action_input=actions,
                output=root / 'action-rollout.yaml', seed=42, initial_origin='source_rgb')


@pytest.mark.parametrize('inference_mode', ['chunked', 'joint61', 'window6'])
def test_action_preset_actual_checkpoint_maxstep_and_static_embedding(tmp_path, monkeypatch, inference_mode):
    from training.demo.prepare_action import prepare
    from training.demo.action_backend import load_action_inputs
    args = action_fixture(tmp_path, monkeypatch)
    if inference_mode != 'chunked':
        args['inference_mode'] = inference_mode
    before = sha256(args['training_config'])
    report = prepare(**args)
    assert report['checkpoint_step'] == 1040 and not report['gpu_launched']
    assert not report['fixed_noise_regression'] and not report['t5_loaded']
    assert sha256(args['training_config']) == before
    deployment = Deployment(args['output'], tmp_path, tmp_path / 'jobs')
    catalog = Catalog(deployment)
    expected_method = {'chunked': 'action_teacher_chunked_ar15s_ui_v1',
                       'joint61': 'action_teacher_joint61_15s_ui_v1',
                       'window6': 'action_teacher_window6_15s_ui_v1'}[inference_mode]
    assert catalog.public()[0]['method'] == expected_method
    assert report['method'] == expected_method
    raw = yaml.safe_load(args['output'].read_text())
    _, model, prompt, lineage = load_action_inputs(raw, raw['scenes'][0])
    assert model.action_scale == .03 and tuple(prompt.shape) == (2, 4096)
    assert lineage['config']['training']['max_steps'] == 1040
    assert lineage['manifest_hashes']['training_config'] == before
    assert lineage['prompt_condition']['text'] == 'A stable stone courtyard.'
    changed = copy.deepcopy(raw['scenes'][0])
    changed['prompt'] = 'Only the display was changed.'
    with pytest.raises(ValueError, match='actual static embedding'):
        load_action_inputs(raw, changed)


@pytest.mark.parametrize('inference_mode', ['chunked', 'joint61', 'window6'])
def test_action_worker_routes_actual_static_embedding_and_timeline_after_gate(tmp_path, monkeypatch, inference_mode):
    """CPU interface test, not evidence that a GPU video was generated."""
    import torch
    import training.demo.backend as backend
    import training.demo.action_backend as action
    import training.eval.input_header as header
    import training.gpu_gate as gate
    from training.demo.prepare_action import prepare
    args = action_fixture(tmp_path, monkeypatch)
    if inference_mode != 'chunked':
        args['inference_mode'] = inference_mode
    expected_method = {'chunked': 'action_teacher_chunked_ar15s_ui_v1',
                       'joint61': 'action_teacher_joint61_15s_ui_v1',
                       'window6': 'action_teacher_window6_15s_ui_v1'}[inference_mode]
    prepare(**args)
    deployment = Deployment(args['output'], tmp_path, tmp_path / 'jobs')
    directory = deployment.jobs_root / ('c' * 32)
    submitted = dict(scene_id='action-dev1', seed=51,
                     action_segments=[dict(frames=120, keys=['W']), dict(frames=120, keys=['L'])])
    frozen = Catalog(deployment).freeze(submitted, directory, directory.name)
    lease, _ = lease_fixture(tmp_path)
    write_json(directory / 'lease.json', lease)
    seen = []
    monkeypatch.setattr(backend, 'validate_lease', lambda *a, **k: seen.append('lease'))
    monkeypatch.setattr(gate, 'query_dedicated_gpu', lambda **kwargs: (seen.append('physical-gate') or SimpleNamespace(as_dict=lambda: {})))
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)

    def cpu_generate(**kwargs):
        assert 'physical-gate' in seen
        assert kwargs['seed'] == 51 and kwargs['model_config'].action_scale == .03
        assert tuple(kwargs['prompt'].shape) == (2, 4096) and torch.all(kwargs['prompt'] == 1)
        assert 'model.act_control_adapter.weight' in kwargs['state']
        np.testing.assert_array_equal(kwargs['initial'], np.load(directory / 'initial.npy'))
        np.testing.assert_array_equal(kwargs['actions'], np.load(directory / 'actions.npy'))
        assert kwargs['actions'].dtype == np.float32
        assert np.all(kwargs['actions'][:120, 0] == 1) and np.all(kwargs['actions'][120:, 7] == 1)
        kwargs['output_path'].write_bytes(b'CPU-ACTION-WIRING-TEST-NOT-A-VIDEO')
        seen.append('actual-action-interface')
        return {'method': expected_method, 'test_only': True}

    def cpu_header(source, output, **kwargs):
        assert kwargs['prompt'] == 'A stable stone courtyard.' and kwargs['seed'] == 51
        np.testing.assert_array_equal(kwargs['actions'], np.load(directory / 'actions.npy'))
        output.write_bytes(b'CPU-ACTION-HUD-WIRING-TEST-NOT-A-VIDEO')
        return {'test_only': True}

    factories = {'chunked': 'generate_action_video', 'joint61': 'generate_action_joint_video',
                 'window6': 'generate_action_window6_video'}
    selected = factories[inference_mode]
    monkeypatch.setattr(action, selected, cpu_generate)
    for unselected in set(factories.values()) - {selected}:
        monkeypatch.setattr(action, unselected, lambda **kwargs: pytest.fail('operator method was not respected'))
    monkeypatch.setattr(header, 'annotate_video', cpu_header)
    result = backend.run_model_job(directory, deployment.project_root, 900)
    assert seen.index('physical-gate') < seen.index('actual-action-interface')
    assert result['lineage']['step'] == 1040 and result['lineage']['prompt_condition']['t5_loaded'] is False
    assert result['method'] == expected_method
    assert result['job_id'] == directory.name and result['actions_sha256'] == frozen['actions_sha256']
    assert validate_result(directory, frozen)['ground_truth_future_used'] is False
    wrong = copy.deepcopy(result)
    wrong['method'] = 'action_teacher_chunked_ar15s_ui_v1' if inference_mode == 'joint61' else 'action_teacher_joint61_15s_ui_v1'
    write_json(directory / 'receipt.json', wrong)
    with pytest.raises(ValueError, match='inference method'):
        validate_result(directory, frozen)


@pytest.mark.parametrize('latent_frames', [3, 25, 61])
def test_action_euler_preserves_initial_and_real_per_frame_timesteps(latent_frames):
    import torch
    from training.demo.action_backend import euler_rollout
    seen = []
    first = torch.full((1, 1, 1, 1, 1), 7.0)
    noise = torch.zeros((1, latent_frames, 1, 1, 1))

    def model(current, *, conditional_dict, timestep, replace_first_timestep_and_noise_latents):
        assert replace_first_timestep_and_noise_latents is True
        assert torch.equal(current[:, :1], first)
        assert timestep[0, 0] == 0 and bool((timestep[0, 1:] > 0).all())
        seen.append(timestep[0, 1].item())
        return torch.ones_like(current), None

    result = euler_rollout(model, first=first, noise=noise, conditions={'actual_action_tensor': 'test'},
                           sigmas=torch.linspace(1, 0, 41))
    assert len(seen) == 40
    assert torch.equal(result[:, :1], first)
    assert torch.allclose(result[:, 1:], torch.full_like(result[:, 1:], -1.0))
    assert torch.equal(noise, torch.zeros_like(noise))


def test_action_chunks_link_float_endpoints_and_do_not_reseed(tmp_path, monkeypatch):
    import torch
    import training.demo.action_backend as action
    monkeypatch.setattr(action, 'HEIGHT', 2)
    monkeypatch.setattr(action, 'WIDTH', 2)
    initial = np.full((2, 2, 3), 17, dtype=np.uint8)
    initial_latent = torch.zeros(1, 1, 48, 30, 52, dtype=torch.bfloat16)
    actions = np.zeros((240, 8), dtype=np.float32)
    actions[:48, 0] = 1
    actions[48:96, 7] = 1
    emitted, encoded, noise_hashes = [], [], []

    def generate(first, prompt, keys, noise, index):
        np.testing.assert_array_equal(keys.numpy()[0], actions[index * 48:(index + 1) * 48])
        if index:
            assert torch.all(first == float(index))
        noise_hashes.append(action.tensor_sha(noise))
        result = noise.clone()
        result[:, :1] = first.float()
        return result

    def decode(latent, index):
        result = torch.zeros(1, 49, 3, 2, 2)
        result[:, -1] = .123 + index / 10
        return result

    def encode(endpoint, index):
        encoded.append(endpoint.clone())
        assert endpoint.shape == (1, 3, 1, 2, 2)
        assert torch.allclose(endpoint, torch.full_like(endpoint, .123 + index / 10))
        return torch.full_like(initial_latent, float(index + 1))

    result = action.rollout_chunks(initial_latent=initial_latent, initial_rgb=initial,
        prompt=torch.ones(2, 4096), actions=actions, seed=42, generate=generate, decode=decode, encode=encode,
        emit=lambda frames, offset: emitted.append((offset, frames.copy())))
    assert [offset for offset, _ in emitted] == [0, 1, 49, 97, 145, 193]
    assert sum(len(frames) for _, frames in emitted) == 241
    np.testing.assert_array_equal(emitted[0][1][0], initial)
    assert len(encoded) == 4 and len(set(noise_hashes)) == 5
    assert result['frozen_noise_regression'] is False and result['native_long_context'] is False
    rng = torch.Generator(device='cpu').manual_seed(42)
    expected = [action.tensor_sha(torch.randn((1, 13, 48, 30, 52), generator=rng)) for _ in range(5)]
    assert noise_hashes == expected


def relocated_action_fixture(tmp_path, monkeypatch):
    """Move only synthetic CPU fixtures; stored YAML/receipts retain old paths."""
    from training.demo.prepare_action import prepare
    original, migrated = tmp_path / 'original', tmp_path / 'migrated'
    args = action_fixture(original, monkeypatch)
    prepare(**args)
    raw = yaml.safe_load(args['output'].read_text())
    spec = raw['lineage']
    hashes = {key: sha256(path) for key, path in spec['artifact_paths'].items()}
    # These exact temporary fixture directories are both within pytest's root.
    assert original.resolve().is_relative_to(tmp_path.resolve())
    assert migrated.resolve().is_relative_to(tmp_path.resolve())
    original.rename(migrated)
    spec['path_relocations'] = {str(original): str(migrated)}
    for key, value in spec['artifact_paths'].items():
        spec['artifact_paths'][key] = str(migrated / Path(value).relative_to(original))
    for key in ('checkpoint_path', 'expected_base_model_path'):
        spec[key] = str(migrated / Path(spec[key]).relative_to(original))
    return raw, hashes, original, migrated


def test_action_relocation_preserves_original_bytes_hashes_and_saved_config(tmp_path, monkeypatch):
    from training.demo.action_backend import load_action_inputs
    raw, hashes, original, migrated = relocated_action_fixture(tmp_path, monkeypatch)
    state, model, prompt, lineage = load_action_inputs(raw, raw['scenes'][0])
    assert state and tuple(prompt.shape) == (2, 4096)
    assert model.base_model_path == raw['lineage']['expected_base_model_path']
    assert Path(lineage['config']['model']['base_model_path']).is_relative_to(original)
    assert lineage['manifest_hashes'] == hashes
    assert lineage['path_relocations'] == {str(original): str(migrated)}
    assert not original.exists()
    assert {key: sha256(path) for key, path in raw['lineage']['artifact_paths'].items()} == hashes


@pytest.mark.parametrize('field', ['path_relocations', 'expected_base_model_path', 'artifact_paths'])
def test_action_relocation_rejects_missing_map_and_wrong_operator_paths(tmp_path, monkeypatch, field):
    from training.demo.action_backend import load_action_inputs
    raw, _, _, migrated = relocated_action_fixture(tmp_path, monkeypatch)
    if field == 'path_relocations':
        raw['lineage'].pop(field)
    elif field == 'expected_base_model_path':
        raw['lineage'][field] = str(migrated / 'different-model')
    else:
        raw['lineage'][field]['feature_index'] = str(migrated / 'unbound-index.jsonl')
    with pytest.raises(ValueError, match='paths differ|base-model pin'):
        load_action_inputs(raw, raw['scenes'][0])


def test_action_relocation_does_not_normalize_changed_checkpoint_config(tmp_path, monkeypatch):
    import torch
    from training.demo.action_backend import load_action_inputs
    raw, _, _, _ = relocated_action_fixture(tmp_path, monkeypatch)
    spec = raw['lineage']
    payload = torch.load(spec['checkpoint_path'], map_location='cpu', weights_only=False)
    # Relocating the saved checkpoint fields themselves is forbidden: compare
    # against the original YAML before considering the operator's prefix map.
    payload['config']['model']['base_model_path'] = spec['expected_base_model_path']
    torch.save(payload, spec['checkpoint_path'])
    spec['checkpoint_sha256'] = sha256(spec['checkpoint_path'])
    with pytest.raises(ValueError, match='changed fields beyond max_steps'):
        load_action_inputs(raw, raw['scenes'][0])


@pytest.mark.parametrize('tamper', ['feature_path', 'feature_schema', 'index_hash', 'prompt_schema',
                                  'prompt_path', 'prompt_policy', 'cache_hash', 'receipt_hash',
                                  'encoder', 'split', 'text_hash', 'unknown_episode'])
def test_action_relocation_retains_receipt_bindings(tmp_path, monkeypatch, tamper):
    from training.config import load_config
    from training.demo.relocation import relocated_manifest_hashes
    raw, _, _, migrated = relocated_action_fixture(tmp_path, monkeypatch)
    spec = raw['lineage']
    paths = spec['artifact_paths']
    key = 'feature_receipt' if tamper.startswith('feature_') or tamper == 'index_hash' else 'prompt_cache_receipt'
    path = Path(paths[key])
    receipt = json.loads(path.read_text())
    if tamper == 'feature_path':
        receipt['index'] = str(migrated / 'other-index.jsonl')
    elif tamper in ('feature_schema', 'prompt_schema'):
        receipt['schema_version'] = 2
    elif tamper == 'index_hash':
        receipt['index_sha256'] = '0' * 64
    elif tamper == 'prompt_path':
        receipt['cache_path'] = str(migrated / 'other-cache.pt')
    elif tamper == 'prompt_policy':
        receipt['prompt_policy'] = 'original_feature_cache_prompt'
    elif tamper == 'cache_hash':
        receipt['cache_sha256'] = '0' * 64
    elif tamper == 'receipt_hash':
        receipt['feature_receipt_sha256'] = '0' * 64
    elif tamper == 'encoder':
        receipt['encoder'] = {}
    elif tamper == 'split':
        receipt['episodes']['episode-source']['split'] = 'train'
    elif tamper == 'text_hash':
        receipt['episodes']['episode-source']['prompt_sha256'] = '0' * 64
    else:
        receipt['episodes']['unknown'] = receipt['episodes'].pop('episode-source')
    write_json(path, receipt)
    config = load_config(paths['training_config'])
    with pytest.raises(ValueError, match='schema|mismatch|provenance|unbound'):
        relocated_manifest_hashes(config, paths['training_config'], spec['path_relocations'])


def test_action_relocation_still_hashes_payload_bytes(tmp_path, monkeypatch):
    from training.config import load_config
    from training.demo.relocation import relocated_manifest_hashes
    raw, _, _, _ = relocated_action_fixture(tmp_path, monkeypatch)
    spec = raw['lineage']
    paths = spec['artifact_paths']
    with Path(paths['prompt_cache']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='cache_sha256 mismatch'):
        relocated_manifest_hashes(load_config(paths['training_config']), paths['training_config'], spec['path_relocations'])


def test_action_relocation_requires_component_prefix_and_valid_roots():
    from training.demo.relocation import relocate_path, validate_relocations
    mapping = validate_relocations({'/home/hmc/YH': '/data/migrated/YH'})
    assert relocate_path('/home/hmc/YH/data/train.jsonl', mapping).as_posix() == '/data/migrated/YH/data/train.jsonl'
    assert relocate_path('/home/hmc/YH-other/data.jsonl', mapping).as_posix() == '/home/hmc/YH-other/data.jsonl'
    for mapping in ({'/': '/data/migrated'}, {'relative': '/data/migrated'}, {'/home/hmc/YH': '/data/../other'}):
        with pytest.raises(ValueError, match='root|absolute|traversal'):
            validate_relocations(mapping)


@pytest.mark.parametrize('relocated', [False, True])
def test_action_verification_cache_reuses_all_hashes_but_enforces_original_pins(tmp_path, monkeypatch, relocated):
    import torch
    from training.demo import relocation
    from training.demo.action_backend import load_action_inputs
    from training.demo.prepare_action import prepare
    if relocated:
        raw, _, _, _ = relocated_action_fixture(tmp_path, monkeypatch)
    else:
        args = action_fixture(tmp_path, monkeypatch)
        prepare(**args)
        raw = yaml.safe_load(args['output'].read_text())
    spec = raw['lineage']
    cache = tmp_path / 'creator-runtime' / 'verified-artifacts.json'
    spec['verification_cache'] = str(cache)
    first = load_action_inputs(raw, raw['scenes'][0])
    saved_cache = json.loads(cache.read_text())
    assert len(saved_cache['artifacts']) == 8  # checkpoint, six bindings, VAE
    actual_hash = relocation.sha256
    monkeypatch.setattr(relocation, 'sha256', lambda _path: pytest.fail('unchanged artifact must reuse its verified SHA'))
    second = load_action_inputs(raw, raw['scenes'][0])
    assert second[3]['manifest_hashes'] == first[3]['manifest_hashes']
    assert torch.equal(second[2], first[2])
    changed_pin = copy.deepcopy(raw)
    changed_pin['lineage']['checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checkpoint SHA mismatch'):
        load_action_inputs(changed_pin, changed_pin['scenes'][0])
    vae = Path(spec['expected_base_model_path']) / 'Wan2.2_VAE.pth'
    vae.write_bytes(vae.read_bytes() + b'CPU test mutation')
    reads = []
    def rehash(path):
        reads.append(Path(path))
        return actual_hash(path)
    monkeypatch.setattr(relocation, 'sha256', rehash)
    with pytest.raises(ValueError, match='Wan VAE differs'):
        load_action_inputs(raw, raw['scenes'][0])
    assert reads == [vae.resolve()]
