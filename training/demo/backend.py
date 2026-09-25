"""Real Wan rollout worker and a fail-closed, operator-owned budget guard bridge.

The guard executable is supplied only by deployment JSON, never by a browser.
It must atomically reserve the shared multi-host budget and settle exactly once.
No guard is bundled with permissive defaults. See the protocol in this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import time

from training.demo.contracts import action_array, contained, require, sha256, write_json
from training.gpu_gate import DEDICATED_PROFILE, SHARED_PROFILE


def _source_revision(code_root):
    """Use the deployed revision marker, never an enclosing repository's HEAD."""
    code_root = Path(code_root).resolve()
    try:
        with (code_root / '.source-revision').open(encoding='ascii') as stream:
            revision = stream.read(128).strip()
            if stream.read(1):
                return 'unknown'
        return revision.lower() if re.fullmatch(r'[0-9a-fA-F]{40}', revision) else 'unknown'
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError):
        return 'unknown'
    try:
        def git_value(argument):
            return subprocess.check_output(['git', 'rev-parse', argument], cwd=code_root,
                                           text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        if Path(git_value('--show-toplevel')).resolve() != code_root:
            return 'unknown'
        revision = git_value('HEAD')
        return revision.lower() if re.fullmatch(r'[0-9a-fA-F]{40}', revision) else 'unknown'
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 'unknown'


def _validate_allocation(lease):
    profile, index = lease.get('profile'), lease.get('gpu_index')
    require(profile in (DEDICATED_PROFILE, SHARED_PROFILE), 'explicit dedicated/shared allocation profile required')
    require(type(index) is int and index >= 0, 'GPU index must be an explicit nonnegative integer')
    if profile == DEDICATED_PROFILE:
        require(index == 0, 'dedicated profile supports GPU0 only')
    uuid = lease.get('gpu_uuid')
    require(isinstance(uuid, str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid),
            'a fixed canonical GPU UUID is required')
    threshold = lease.get('desktop_memory_max_exclusive')
    ceiling = 500 if profile == SHARED_PROFILE else 1537
    require(type(threshold) is int and 1 <= threshold <= ceiling, f'invalid {profile} memory ceiling')


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
    validate_active_authority(lease)


def validate_active_authority(lease):
    _validate_allocation(lease)
    path = Path(lease['authorization_record'])
    require(sha256(path) == lease.get('authorization_sha256'), 'authorization changed or revoked')
    authority = json.loads(path.read_text(encoding='utf-8'))
    require(authority.get('status') == 'ACTIVE' and authority.get('project') == 'InterActWorld'
            and authority.get('authorization_id') == lease['authorization_id']
            and authority.get('until_user_stop') is True, 'active standing authorization is required')
    storage_limit = authority.get('total_storage_bytes_limit', 300_000_000_000)
    require(authority.get('total_gpu_hours_limit') == 160 and type(storage_limit) is int
            and storage_limit in (300_000_000_000, 400_000_000_000, 600_000_000_000), 'authorization budget scope changed')
    storage = lease.get('global_storage_bytes_upper')
    require(type(storage) is int and 0 <= storage <= storage_limit,
            'global storage budget exceeds the active authorization limit')
    host = authority.get('hosts', {}).get(lease.get('profile'))
    require(isinstance(host, dict) and str(host.get('hostname')).lower() == socket.gethostname().lower(),
            'authorization host mismatch')
    if 'gpus' in host:
        cards = host['gpus']
        require(lease['profile'] == SHARED_PROFILE and isinstance(cards, list) and cards
                and all(isinstance(card, dict) and type(card.get('gpu_index')) is int
                        and card['gpu_index'] >= 0 and isinstance(card.get('gpu_uuid'), str) for card in cards),
                'invalid shared authorization GPU list')
        require(len({card['gpu_index'] for card in cards}) == len(cards)
                and len({card['gpu_uuid'].lower() for card in cards}) == len(cards), 'duplicate authorization GPU identity')
        selected = [card for card in cards if card['gpu_index'] == lease['gpu_index']
                    and card['gpu_uuid'] == lease['gpu_uuid']]
        require(len(selected) == 1, 'lease GPU index/UUID is not authorized')
        host = selected[0]
    for key in ('gpu_uuid', 'gpu_index', 'desktop_memory_max_exclusive'):
        require(host.get(key) == lease.get(key), f'authorization {key} mismatch')


class CommandGuard:
    """stdin/stdout JSON protocol: reserve/check/settle, no shell expansion.

    reserve input includes job_id, request_sha256, max_seconds and project_root.
    Output must satisfy validate_lease. The private provider owns a locked global
    budget ledger, the160h checks, the explicitly authorized300GB/400GB/600GB storage ceiling,
    and any cross-host open reservations.
    check returns {status: allowed}; settle returns {status: settled}. All calls
    include the immutable job ID and lease; settlement must be idempotent.
    """
    def __init__(self, deployment):
        require(bool(deployment.guard_command), 'GPU generation is disabled until an operator configures the budget/authorization guard')
        self.deployment = deployment

    def call(self, operation, **values):
        result = subprocess.run(list(self.deployment.guard_command),
                                input=json.dumps(dict(operation=operation, **values)),
                                text=True, capture_output=True, check=True, timeout=45)
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
            timeout_program = shutil.which('timeout')
            require(timeout_program is not None, 'GNU timeout is required for an independent worker watchdog')
            command = [timeout_program, '--signal=TERM', '--kill-after=10s',
                       f'{self.deployment.max_job_seconds}s', self.deployment.python_executable, '-m', 'training.demo.worker',
                       '--job-directory', str(directory), '--project-root', str(self.deployment.project_root),
                       '--max-seconds', str(self.deployment.max_job_seconds)]
            environment = os.environ.copy()
            cache = directory / 'cache'
            cache.mkdir()
            environment.update(TMPDIR=str(directory), TMP=str(directory), TEMP=str(directory),
                CUDA_VISIBLE_DEVICES=lease['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTHONUNBUFFERED='1',
                XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache / 'torch'), HF_HOME=str(cache / 'huggingface'),
                CUDA_CACHE_PATH=str(cache / 'cuda'), TRITON_CACHE_DIR=str(cache / 'triton'),
                TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), PYTORCH_KERNEL_CACHE_PATH=str(cache / 'kernels'),
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                WANDB_DISABLED='true', WANDB_MODE='disabled')
            with (directory / 'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
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
    query_gpu = {DEDICATED_PROFILE: gate.query_dedicated_gpu, SHARED_PROFILE: gate.query_shared_gpu}[lease['profile']]
    snapshot = query_gpu(confirmed_index=lease['gpu_index'], confirmed_uuid=lease['gpu_uuid'],
                         profile=lease['profile'], memory_max_exclusive=lease['desktop_memory_max_exclusive'])
    # This is an explicit standing grant, not a refreshed900s confirmation.
    # All model construction occurs after actual UUID/no-compute/desktop checks.
    import torch
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
    metadata = dict(outcome='raw_generated', job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'),
                  **{key: request[key] for key in ('actions_sha256', 'initial_sha256', 'config_sha256', 'checkpoint_sha256')},
                  lineage=lineage, gpu_snapshot=snapshot.as_dict(), authorization_id=lease['authorization_id'],
                  reservation_id=lease['reservation_id'], git_revision=_source_revision(Path(__file__).resolve().parents[2]),
                  generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), total_rgb_frames=241, fps=16,
                  future_duration_seconds=15, ground_truth_future_used=False, quality_evaluation='not_run',
                  custom_action_ground_truth='not_available; no reference MSE is claimed',
                  generation=generation, method=request.get('method', generation['method']),
                  elapsed_seconds=time.monotonic() - started, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  elapsed_seconds_scope='raw_generation_before_annotation',
                  videos={'raw.mp4': sha256(directory / 'raw.mp4')})
    require(not (directory / 'generation.json').exists(), 'refusing to overwrite generation metadata')
    write_json(directory / 'generation.json', metadata)
    return package_video_result(directory, request, initial, actions, metadata)


def package_video_result(directory, request, initial, actions, metadata, *, recovery=None):
    """CPU display packaging, using actual saved metadata or explicit nulls."""
    from training.eval.input_header import annotate_video
    require(metadata.get('job_id') == request['job_id']
            and metadata.get('request_sha256') == sha256(directory / 'request.json'), 'generation metadata belongs to another job/request')
    for key in ('actions_sha256', 'initial_sha256', 'config_sha256', 'checkpoint_sha256'):
        require(metadata.get(key) == request[key], f'generation metadata {key} mismatch')
    require(metadata.get('method') == request.get('method', 'causal_rollout15s'), 'generation method mismatch')
    require(metadata.get('videos', {}).get('raw.mp4') == sha256(directory / 'raw.mp4'), 'raw generation bytes changed')
    require(not (directory / 'receipt.json').exists(), 'refusing to overwrite an existing receipt')
    began = time.monotonic()
    display = annotate_video(directory / 'raw.mp4', directory / 'inputs.mp4', initial_frame=initial,
                             prompt=request['prompt'], actions=actions, seed=request['seed'], fps=16)
    result = {**metadata, 'outcome': 'completed', 'display': display,
              'postprocessing_elapsed_seconds': time.monotonic() - began,
              'videos': {name: sha256(directory / name) for name in ('raw.mp4', 'inputs.mp4')}}
    require(result['videos']['raw.mp4'] == metadata['videos']['raw.mp4'], 'raw generation changed during packaging')
    if recovery is None:
        result['elapsed_seconds'] += result['postprocessing_elapsed_seconds']
        result['elapsed_seconds_scope'] = 'generation_and_postprocessing'
    else:
        result['source'] = 'cpu_postprocess_recovery'
        result['recovery'] = recovery
    write_json(directory / 'receipt.json', result)
    return result


def _read_job_json(directory, name):
    path = contained(directory / name, directory)
    require(path.is_file() and path.stat().st_size <= 2_000_000, f'missing/oversized {name}')
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'invalid {name}')
    return value


def _recovery_inputs(directory, project_root):
    """Validate frozen CPU inputs without loading weights or touching a GPU."""
    import numpy as np
    import yaml
    directory = contained(directory, project_root)
    request = _read_job_json(directory, 'request.json')
    require(re.fullmatch('[0-9a-f]{32}', directory.name) and request.get('job_id') == directory.name,
            'recovery job identity/path mismatch')
    require(request.get('ground_truth_future_used') is False and request.get('fps') == 16
            and request.get('future_frames') == 240, 'unsupported frozen request contract')
    for name, key in [('rollout.yaml', 'config_sha256'), ('initial.npy', 'initial_sha256'), ('actions.npy', 'actions_sha256')]:
        require(sha256(contained(directory / name, directory)) == request[key], f'job input changed: {name}')
    lease = _read_job_json(directory, 'lease.json')
    require(lease.get('job_id') == directory.name and lease.get('request_sha256') == sha256(directory / 'request.json'),
            'original lease does not bind this frozen request')
    settlement_request = _read_job_json(directory, 'settlement-request.json')
    settlement = _read_job_json(directory, 'settlement.json')
    require(settlement_request.get('job_id') == directory.name
            and settlement_request.get('request_sha256') == lease['request_sha256']
            and settlement_request.get('lease') == lease and settlement_request.get('outcome') == 'failed',
            'recovery requires the original failed worker settlement request')
    require(settlement.get('status') == 'settled' and settlement.get('reservation_id') == lease.get('reservation_id'),
            'original GPU reservation must already be settled')
    raw = yaml.safe_load((directory / 'rollout.yaml').read_text(encoding='utf-8'))
    require(isinstance(raw, dict) and raw.get('run_id') == directory.name
            and Path(raw.get('output_root', '')).resolve() == directory, 'frozen rollout belongs to another job')
    require(raw.get('method', 'causal_rollout15s') == request.get('method', 'causal_rollout15s')
            and raw.get('lineage', {}).get('checkpoint_sha256') == request['checkpoint_sha256'], 'frozen checkpoint/method binding changed')
    geometry = raw.get('geometry', {})
    require(all(geometry.get(key) == value for key, value in dict(width=832, height=480, fps=16, total_rgb_frames=241).items()),
            'recovery requires the fixed 832x480, 241-frame, 16-fps rollout')
    scenes = raw.get('scenes')
    require(isinstance(scenes, list) and len(scenes) == 1, 'recovery requires one frozen scene')
    scene = scenes[0]
    require(all(scene.get(key) == request[key] for key in ('scene_id', 'source_episode_id', 'seed', 'prompt', 'action_segments'))
            and Path(scene.get('initial_frame_path', '')).resolve() == directory / 'initial.npy', 'frozen scene binding changed')
    initial = np.load(directory / 'initial.npy', allow_pickle=False)
    actions = np.load(directory / 'actions.npy', allow_pickle=False)
    require(initial.dtype == np.uint8 and initial.shape == (480, 832, 3), 'invalid frozen initial image')
    require(actions.dtype == np.float32 and np.array_equal(actions, action_array(request['action_segments'])),
            'frozen action tensor differs from the request')
    video = directory / 'raw.mp4'
    require(not video.is_symlink() and contained(video, directory) == video and video.is_file()
            and video.stat().st_size > 0 and video.stat().st_nlink == 1, 'recovery accepts only this job\'s original unlinked raw.mp4')
    return directory, request, lease, initial, actions


def _decode_recovery_video(path):
    from training.eval.input_header import _probe_video
    probe = _probe_video(path, count_frames=True)
    require((probe['width'], probe['height'], probe['frames'], probe['fps']) == (832, 480, 241, 16),
            'raw video dimensions/frame count/fps are incomplete or invalid')
    # -xerror makes decoder errors fatal; software decode never opens CUDA.
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-hwaccel', 'none', '-noautorotate',
                    '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-f', 'null', '-'],
                   capture_output=True, check=True, timeout=120)
    return dict(width=832, height=480, frames=241, fps=16, complete_software_decode=True)


def prepare_postprocess_recovery(directory, project_root):
    """Prepare only this failed job's display video/receipt; leave status failed."""
    began = time.monotonic()
    directory, request, lease, initial, actions = _recovery_inputs(directory, project_root)
    status = _read_job_json(directory, 'status.json')
    require(status.get('job_id') == directory.name and status.get('status') == 'failed', 'prepare requires an already failed job')
    failed_bytes = (directory / 'status.json').read_bytes()
    failed_sha = sha256(directory / 'status.json')
    recovery_path = directory / 'recovery.json'
    if recovery_path.exists():
        recovery = _read_job_json(directory, 'recovery.json')
        require(recovery.get('state') == 'prepared' and recovery.get('failed_status_sha256') == failed_sha
                and recovery.get('request_sha256') == lease['request_sha256']
                and recovery.get('receipt_sha256') == sha256(directory / 'receipt.json'), 'existing recovery preparation changed')
        validate_result(directory, request)
        return recovery
    require(not (directory / 'inputs.mp4').exists() and not (directory / 'receipt.json').exists(),
            'existing postprocessing artifacts require inspection; refusing to overwrite them')
    raw_sha = sha256(directory / 'raw.mp4')
    decoded = _decode_recovery_video(directory / 'raw.mp4')
    require(sha256(directory / 'raw.mp4') == raw_sha, 'raw video changed while checking recovery')
    history_path = directory / 'recovery.failed-status.json'
    if history_path.exists():
        require(history_path.read_bytes() == failed_bytes, 'original failure history differs')
    else:
        with history_path.open('xb') as stream:
            stream.write(failed_bytes)
    started, finished = status.get('started_unix'), status.get('finished_unix')
    original_elapsed = (finished - started if type(started) in (int, float) and type(finished) in (int, float)
                        and math.isfinite(started) and math.isfinite(finished) and finished >= started else None)
    recovery = dict(source='cpu_postprocess_recovery', job_id=directory.name, request_sha256=lease['request_sha256'],
                    failed_status_file=history_path.name, failed_status_sha256=failed_sha,
                    raw_validation=decoded, raw_video_sha256=raw_sha, gpu_execution=False, new_gpu_seconds=0,
                    original_job_elapsed_seconds=original_elapsed,
                    original_job_elapsed_scope='original status wall time; includes CPU prechecks, generation and failed packaging; not model/GPU time')
    metadata_path = directory / 'generation.json'
    if metadata_path.exists():
        metadata = _read_job_json(directory, 'generation.json')
        require(metadata.get('outcome') == 'raw_generated', 'invalid saved generation metadata')
        recovery['generation_metadata_source'] = 'generation.json'
        recovery['generation_metadata_sha256'] = sha256(metadata_path)
    else:
        reason = 'The original worker failed before persisting generation metadata; model time, peak VRAM and generation details cannot be reconstructed.'
        metadata = dict(outcome='raw_generated', job_id=directory.name, request_sha256=lease['request_sha256'],
            **{key: request[key] for key in ('actions_sha256', 'initial_sha256', 'config_sha256', 'checkpoint_sha256')},
            lineage=None, gpu_snapshot=None, authorization_id=lease['authorization_id'], reservation_id=lease['reservation_id'],
            git_revision=None, generated_at_utc=None, total_rgb_frames=241, fps=16, future_duration_seconds=15,
            ground_truth_future_used=False, quality_evaluation='not_run',
            custom_action_ground_truth='not_available; no reference MSE is claimed', generation=None,
            method=request.get('method', 'causal_rollout15s'), elapsed_seconds=None,
            elapsed_seconds_scope='unavailable; original job wall time is recorded separately', peak_vram_bytes=None,
            missing_generation_metadata_reason=reason, videos={'raw.mp4': raw_sha})
        recovery['generation_metadata_source'] = None
        recovery['generation_metadata_sha256'] = None
        recovery['missing_generation_metadata_reason'] = reason
    package_video_result(directory, request, initial, actions, metadata, recovery=recovery)
    recovery.update(state='prepared', prepared_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    cpu_recovery_elapsed_seconds=time.monotonic() - began, receipt_sha256=sha256(directory / 'receipt.json'))
    require((directory / 'status.json').read_bytes() == failed_bytes, 'job status changed during recovery preparation')
    write_json(recovery_path, recovery)
    return recovery


@contextmanager
def _recovery_service_lock(jobs_root):
    require(os.name == 'posix', 'recovery finalization requires the Linux demo-service flock')
    import fcntl
    with (jobs_root / '.demo-service.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('demo service is running; keep prepared results and finalize after an idle shutdown') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def finalize_postprocess_recovery(directory, project_root):
    """Publish prepared completion only while owning the idle service queue."""
    directory = contained(directory, project_root)
    with _recovery_service_lock(directory.parent):
        for other in directory.parent.iterdir():
            if other.is_dir() and re.fullmatch('[0-9a-f]{32}', other.name) and (other / 'status.json').exists():
                require(_read_job_json(other, 'status.json').get('status') not in ('queued', 'running'),
                        'service queue still has queued/running jobs; no status was changed')
        directory, request, _, _, _ = _recovery_inputs(directory, project_root)
        recovery = _read_job_json(directory, 'recovery.json')
        require(recovery.get('source') == 'cpu_postprocess_recovery' and recovery.get('state') in ('prepared', 'finalized')
                and recovery.get('job_id') == directory.name and recovery.get('request_sha256') == sha256(directory / 'request.json')
                and recovery.get('receipt_sha256') == sha256(directory / 'receipt.json'), 'prepared recovery binding changed')
        require(sha256(directory / 'recovery.failed-status.json') == recovery['failed_status_sha256'], 'original failure history changed')
        validate_result(directory, request)
        status = _read_job_json(directory, 'status.json')
        if status.get('status') == 'completed':
            require(status.get('recovery', {}).get('source') == 'cpu_postprocess_recovery', 'job was completed by another operation')
            return recovery
        require(status.get('status') == 'failed' and sha256(directory / 'status.json') == recovery['failed_status_sha256'],
                'failed job status changed since preparation')
        recovery.update(state='finalized', finalized_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(directory / 'recovery.json', recovery)
        status.update(status='completed', error=None, failure_type=None, finished_unix=time.time(),
                      recovery=dict(source='cpu_postprocess_recovery', record='recovery.json',
                                    previous_status_file='recovery.failed-status.json', new_gpu_seconds=0))
        write_json(directory / 'status.json', status)
        return recovery
