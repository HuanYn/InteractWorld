"""Real Wan rollout worker and a fail-closed, operator-owned budget guard bridge.

The guard executable is supplied only by deployment JSON, never by a browser.
It must atomically reserve the shared multi-host budget and settle exactly once.
No guard is bundled with permissive defaults. See the protocol in this module.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from training.demo.contracts import action_array, contained, require, sha256, write_json


def validate_lease(lease, *, max_seconds, now=None, job_id=None, request_sha256=None):
    require(isinstance(lease, dict) and lease.get('status') == 'reserved', 'GPU budget reservation denied')
    require(isinstance(lease.get('reservation_id'), str) and lease['reservation_id'], 'reservation ID missing')
    require(isinstance(lease.get('authorization_id'), str), 'authorization ID missing')
    if job_id is not None:
        require(lease.get('job_id') == job_id and lease.get('request_sha256') == request_sha256,
                'reservation belongs to another job/request')
    now = time.time() if now is None else now
    require(type(lease.get('start_before_unix')) in (int, float)
            and now <= lease['start_before_unix'] <= now + 900, 'reservation start lease expired/invalid')
    reserved, upper = lease.get('gpu_hours_reserved'), lease.get('global_gpu_hours_upper')
    require(type(reserved) in (int, float) and math.isfinite(reserved) and reserved >= (max_seconds + 30) / 3600,
            'reservation must include full worker timeout and30s guard/termination allowance')
    require(type(upper) in (int, float) and math.isfinite(upper) and reserved <= upper <= 160,
            'global completed+open reservations exceed160 GPU-hours')
    storage = lease.get('global_storage_bytes_upper')
    require(type(storage) is int and 0 <= storage <= 300_000_000_000, 'global storage budget exceeds300GB')
    require(type(lease.get('gpu_index')) is int and lease['gpu_index'] == 0, 'this service supports dedicated GPU0 only')
    threshold = lease.get('desktop_memory_max_exclusive')
    require(type(threshold) is int and 1 <= threshold <= 1537, 'invalid desktop-only memory ceiling')
    validate_active_authority(lease)


def validate_active_authority(lease):
    path = Path(lease['authorization_record'])
    require(sha256(path) == lease.get('authorization_sha256'), 'authorization changed or revoked')
    authority = json.loads(path.read_text(encoding='utf-8'))
    require(authority.get('status') == 'ACTIVE' and authority.get('project') == 'InterActWorld'
            and authority.get('authorization_id') == lease['authorization_id']
            and authority.get('until_user_stop') is True, 'active standing authorization is required')
    require(authority.get('total_gpu_hours_limit') == 160
            and authority.get('total_storage_bytes_limit') == 300_000_000_000, 'authorization budget scope changed')
    host = authority.get('hosts', {}).get(lease.get('profile'))
    require(isinstance(host, dict) and str(host.get('hostname')).lower() == socket.gethostname().lower(),
            'authorization host mismatch')
    for key in ('gpu_uuid', 'gpu_index', 'desktop_memory_max_exclusive'):
        require(host.get(key) == lease.get(key), f'authorization {key} mismatch')


class CommandGuard:
    """stdin/stdout JSON protocol: reserve/check/settle, no shell expansion.

    reserve input includes job_id, request_sha256, max_seconds and project_root.
    Output must satisfy validate_lease. The private provider owns a locked global
    budget ledger, the160h/300GB checks and any cross-host open reservations.
    check returns {status: allowed}; settle returns {status: settled}. All calls
    include the immutable job ID and lease; settlement must be idempotent.
    """
    def __init__(self, deployment):
        require(bool(deployment.guard_command), 'GPU generation is disabled until an operator configures the budget/authorization guard')
        self.deployment = deployment

    def call(self, operation, **values):
        result = subprocess.run(list(self.deployment.guard_command),
                                input=json.dumps(dict(operation=operation, **values)),
                                text=True, capture_output=True, check=True, timeout=5)
        require(len(result.stdout) <= 65536, 'guard response is too large')
        return json.loads(result.stdout)


def terminate_owned_process(process):
    if os.name == 'posix':
        # The leader may already have exited while a decoder descendant remains.
        # Every worker was created in its own new session: this PGID is ours.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        if process.poll() is not None:
            return
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class SubprocessBackend:
    """One worker process per request; never serves an old video as a new job."""
    def __init__(self, deployment):
        self.deployment = deployment

    def __call__(self, directory, stop_event):
        require(os.name == 'posix', 'GPU worker supervision currently requires Linux; CPU UI works elsewhere')
        guard = CommandGuard(self.deployment)
        request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
        context = dict(job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'),
                       max_seconds=self.deployment.max_job_seconds, project_root=str(self.deployment.project_root),
                       job_directory=str(directory))
        lease = guard.call('reserve', **context)
        process = None
        started = time.monotonic()
        outcome = 'failed'
        try:
            validate_lease(lease, max_seconds=self.deployment.max_job_seconds,
                           job_id=context['job_id'], request_sha256=context['request_sha256'])
            write_json(directory / 'lease.json', lease)
            command = [self.deployment.python_executable, '-m', 'training.demo.worker',
                       '--job-directory', str(directory), '--project-root', str(self.deployment.project_root),
                       '--max-seconds', str(self.deployment.max_job_seconds)]
            with (directory / 'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                last_check = 0.0
                while process.poll() is None:
                    if stop_event.wait(1):
                        raise RuntimeError('service stopped; this owned job was interrupted')
                    require(time.monotonic() - started < self.deployment.max_job_seconds, 'generation exceeded its reserved time limit')
                    validate_active_authority(lease)
                    if time.monotonic() - last_check >= 30:
                        allowed = guard.call('check', lease=lease, **context)
                        require(allowed.get('status') == 'allowed', 'operator budget/storage guard revoked this job')
                        last_check = time.monotonic()
                require(process.returncode == 0, 'model worker failed; operator details are in worker.log')
            result = validate_result(directory, request)
            outcome = 'completed'
            return result
        finally:
            if process is not None:
                terminate_owned_process(process)
            # Preserve unresolved reservations rather than pretending a failed
            # settlement succeeded; the operator must reconcile this receipt.
            settlement = dict(**context, lease=lease, outcome=outcome,
                              elapsed_seconds=time.monotonic() - started,
                              worker_returncode=None if process is None else process.returncode)
            write_json(directory / 'settlement-request.json', settlement)
            result = guard.call('settle', **settlement)
            require(result.get('status') == 'settled', 'budget settlement failed; operator reconciliation required')
            write_json(directory / 'settlement.json', result)


def validate_result(directory, request):
    result = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
    require(result.get('outcome') == 'completed' and result.get('job_id') == request['job_id'], 'result belongs to another/incomplete job')
    require(result.get('request_sha256') == sha256(directory / 'request.json'), 'result request hash mismatch')
    for key in ('actions_sha256', 'initial_sha256', 'config_sha256', 'checkpoint_sha256'):
        require(result.get(key) == request[key], f'result {key} mismatch')
    require(result.get('total_rgb_frames') == 241 and result.get('fps') == 16, 'result is not a real15s rollout')
    require(result.get('ground_truth_future_used') is False, 'invalid generation provenance')
    require(result.get('method', 'causal_rollout15s') == request.get('method', 'causal_rollout15s'),
            'result inference method differs from the frozen request')
    for name in ('raw.mp4', 'inputs.mp4'):
        path = contained(directory / name, directory)
        require(path.is_file() and path.stat().st_size > 0 and result['videos'][name] == sha256(path), 'generated video missing or changed')
    return result


def run_model_job(directory, project_root, max_seconds):
    """The production inference path: exact trained lineage -> actual action tensors."""
    directory = contained(directory, project_root)
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    lease = json.loads((directory / 'lease.json').read_text(encoding='utf-8'))
    validate_lease(lease, max_seconds=max_seconds, job_id=request['job_id'],
                   request_sha256=sha256(directory / 'request.json'))
    require(directory.name == request['job_id'], 'job identity/path mismatch')
    for name, key in [('rollout.yaml', 'config_sha256'), ('initial.npy', 'initial_sha256'), ('actions.npy', 'actions_sha256')]:
        require(sha256(directory / name) == request[key], f'job input changed: {name}')
    require(not (directory / 'raw.mp4').exists() and not (directory / 'inputs.mp4').exists(), 'refusing to reuse previous output')
    from training.eval.rollout15s import load_rollout_config, verify_checkpoint_lineage, _run_variant, default_image_loader, ffmpeg_writer_factory
    from training.demo.contracts import ADAPTER, ACTION_JOINT_METHOD, ACTION_WINDOW6_METHOD, ACTION_STAGE, action_contract
    import yaml
    raw = yaml.safe_load((directory / 'rollout.yaml').read_text(encoding='utf-8'))
    is_action = raw.get('lineage', {}).get('expected_stage') == ACTION_STAGE
    action_inputs = None
    if is_action:
        from types import SimpleNamespace
        from training.demo.action_backend import load_action_inputs
        require(len(raw['scenes']) == 1, 'Action worker needs exactly one selected scene')
        config = SimpleNamespace(adapter_factory=raw['adapter_factory'],
                                 scenes=[SimpleNamespace(**raw['scenes'][0])],
                                 lineage=SimpleNamespace(**raw['lineage']))
        action_inputs = load_action_inputs(raw, raw['scenes'][0])
    else:
        config = load_rollout_config(directory / 'rollout.yaml')
    require(config.adapter_factory == (action_contract(raw['method'])[0] if is_action else ADAPTER) and len(config.scenes) == 1,
            'unexpected concrete backend/scene count')
    require(config.lineage.checkpoint_sha256 == request['checkpoint_sha256'], 'checkpoint pin changed')
    require(raw.get('method', 'causal_rollout15s') == request.get('method', 'causal_rollout15s'),
            'operator inference method differs from the frozen request')
    scene = config.scenes[0]
    require(scene.prompt == request['prompt'] and scene.seed == request['seed']
            and scene.scene_id == request['scene_id'] and scene.source_episode_id == request['source_episode_id'],
            'scene input binding changed')
    require(Path(scene.initial_frame_path).resolve() == directory / 'initial.npy', 'initial frame is not the frozen job condition')
    import numpy as np
    actions = np.load(directory / 'actions.npy', allow_pickle=False)
    require(actions.dtype == np.float32 and np.array_equal(actions, action_array(request['action_segments'])), 'actual action tensor differs from submitted timeline')
    lineage = action_inputs[3] if is_action else verify_checkpoint_lineage(config)
    validate_lease(lease, max_seconds=max_seconds, job_id=request['job_id'],
                   request_sha256=sha256(directory / 'request.json'))
    import training.gpu_gate as gate
    gate.MAX_IDLE_DISPLAY_MEMORY_MIB = lease['desktop_memory_max_exclusive']
    snapshot = gate.query_dedicated_gpu(confirmed_index=lease['gpu_index'], confirmed_uuid=lease['gpu_uuid'],
                                       profile='dedicated_local_single_gpu')
    # This is an explicit standing grant, not a refreshed900s confirmation.
    # All model construction occurs after actual UUID/no-compute/desktop checks.
    import torch
    from training.eval.input_header import annotate_video
    from training.runtime import git_revision
    torch.manual_seed(scene.seed)
    torch.cuda.manual_seed_all(scene.seed)
    started = time.monotonic()
    initial = default_image_loader(directory / 'initial.npy')
    if is_action:
        from training.demo.action_backend import generate_action_video, generate_action_joint_video, generate_action_window6_video
        state, model_config, prompt, _ = action_inputs
        generate = (generate_action_joint_video if raw['method'] == ACTION_JOINT_METHOD else
                    generate_action_window6_video if raw['method'] == ACTION_WINDOW6_METHOD else generate_action_video)
        generation = generate(state=state, model_config=model_config, prompt=prompt,
                              initial=initial, actions=actions, seed=scene.seed,
                              output_path=directory / 'raw.mp4')
    else:
        from training.eval.wan_causal_adapter import create_wan_causal_adapter
        adapter = create_wan_causal_adapter(checkpoint_path=config.lineage.checkpoint_path,
            checkpoint_sha256=config.lineage.checkpoint_sha256, base_model_path=config.lineage.expected_base_model_path,
            device='cuda')
        _run_variant(config=config, scene=scene, variant='user_submitted_actions', actions=actions,
                     initial=initial, adapter=adapter, output_path=directory / 'raw.mp4', writer_factory=ffmpeg_writer_factory)
        generation = dict(method='causal_rollout15s', context_mode=getattr(adapter, 'context_mode', 'unspecified'))
    require(generation['method'] == raw.get('method', 'causal_rollout15s'), 'generated with an unexpected inference method')
    display = annotate_video(directory / 'raw.mp4', directory / 'inputs.mp4', initial_frame=initial,
                             prompt=scene.prompt, actions=actions, seed=scene.seed, fps=16)
    result = dict(outcome='completed', job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'),
                  **{key: request[key] for key in ('actions_sha256', 'initial_sha256', 'config_sha256', 'checkpoint_sha256')},
                  lineage=lineage, gpu_snapshot=snapshot.as_dict(), authorization_id=lease['authorization_id'],
                  reservation_id=lease['reservation_id'], git_revision=git_revision(Path(__file__).resolve().parents[2]),
                  generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), total_rgb_frames=241, fps=16,
                  future_duration_seconds=15, ground_truth_future_used=False, quality_evaluation='not_run',
                  custom_action_ground_truth='not_available; no reference MSE is claimed',
                  generation=generation, method=request.get('method', generation['method']),
                  elapsed_seconds=time.monotonic() - started, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  display=display, videos={name: sha256(directory / name) for name in ('raw.mp4', 'inputs.mp4')})
    write_json(directory / 'receipt.json', result)
    return result
