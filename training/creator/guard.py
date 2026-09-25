"""Private-config GPU/budget bridge for the Linux Creator deployment.

No authority is created here. Both video jobs and model requests must use this
same ledger and lock_root. A persistent GPU marker is never reclaimed on age:
explicit settlement must first verify that the selected physical card is idle.
The global flock protects accounting; O_EXCL markers protect each UUID across
processes, including guards using different ledgers under the same lock_root.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from training import gpu_gate

# Includes cold local-weight I/O plus inference; the 3090 host needed ~180s
# just to load the visual model. Still finite and fully reserved in the ledger.
MAX_MODEL_SECONDS = 480
MAX_NEW_STORAGE_BYTES = 100_000_000_000
MAX_TOTAL_STORAGE_BYTES = 400_000_000_000
STORAGE_METADATA_ALLOWANCE = 64 * 1024 * 1024
RPC_SECONDS = 30.0


class GuardError(RuntimeError):
    """A denied operation leaves existing reservations intact."""


def _require(condition, message):
    if not condition:
        raise GuardError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, f'duplicate JSON field: {key}')
        value[key] = item
    return value


def _loads(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(GuardError(f'invalid JSON number: {value}')))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path, max_bytes=2_000_000):
    path = Path(path)
    _require(path.is_file() and path.stat().st_size <= max_bytes, 'missing or oversized JSON file')
    return _loads(path.read_text(encoding='utf-8'))


def _write(path, value, *, exclusive=False):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _within(path, root, *, equal=False):
    result, root = Path(path).resolve(), Path(root).resolve()
    _require(result.is_relative_to(root) and (equal or result != root), 'path must stay under the configured project/storage root')
    return result


def _require_linux():
    _require(sys.platform.startswith('linux'), 'the GPU guard requires Linux; CPU planning does not need this guard')


@contextmanager
def _file_lock(path):
    _require_linux()
    import fcntl
    with Path(path).open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GuardError('global GPU budget is locked by another operation') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _deadline():
    # The video CommandGuard permits 45 seconds including interpreter startup.
    # SIGALRM also bounds nvidia-smi's own longer timeout and filesystem scans.
    if not sys.platform.startswith('linux'):
        yield  # Pure-CPU tests replace physical checks and locking on Windows.
        return
    def expired(_signum, _frame):
        raise GuardError('guard operation exceeded its thirty-second deadline')
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, RPC_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def _directory_bytes(path):
    result = subprocess.run(['du', '--bytes', '--summarize', '--', str(path)],
                            capture_output=True, text=True, check=True, timeout=25)
    fields = result.stdout.split()
    _require(fields and fields[0].isdigit(), 'invalid storage inventory')
    return int(fields[0])


@dataclass(frozen=True)
class GuardConfig:
    authority: Path
    ledger: Path
    lock_root: Path
    project_root: Path
    python: Path
    model: Path
    prior_gpu_hours_upper: float
    max_round_gpu_hours: float
    prior_storage_bytes: int
    storage_root: Path
    storage_limit: int = MAX_TOTAL_STORAGE_BYTES
    gpu_index: int | None = None

    @classmethod
    def load(cls, path):
        raw = _read(path, 65536)
        _require(isinstance(raw, dict) and set(raw).issubset(cls.__dataclass_fields__), 'unknown guard configuration fields')
        required = set(cls.__dataclass_fields__) - {'storage_limit', 'gpu_index'}
        _require(required.issubset(raw), 'guard configuration is incomplete')
        for key in ('authority', 'ledger', 'lock_root', 'project_root', 'python', 'model', 'storage_root'):
            _require(isinstance(raw[key], str) and Path(raw[key]).is_absolute(), f'{key} must be an absolute path')
            # Resolving venv/bin/python's symlink selects the system interpreter
            # and loses the environment's installed torch/transformers packages.
            raw[key] = Path(raw[key]).absolute() if key == 'python' else Path(raw[key]).resolve()
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        _require(self.gpu_index is None or (type(self.gpu_index) is int and self.gpu_index >= 0),
                 'gpu_index must be an explicit nonnegative integer or null for a legacy single-card authority')
        _require(_number(self.prior_gpu_hours_upper) and 0 <= self.prior_gpu_hours_upper <= 160,
                 'invalid prior GPU-hour upper bound')
        _require(_number(self.max_round_gpu_hours) and 0 < self.max_round_gpu_hours <= 6,
                 'this Creator round has an operator ceiling of six GPU-hours')
        _require(type(self.prior_storage_bytes) is int and 0 <= self.prior_storage_bytes <= MAX_TOTAL_STORAGE_BYTES,
                 'invalid retained prior-storage accounting')
        _require(type(self.storage_limit) is int and self.storage_limit in (300_000_000_000, MAX_TOTAL_STORAGE_BYTES),
                 'storage ceiling must be 300GB or 400GB')
        _require(self.storage_root.is_absolute() and self.storage_root != Path(self.storage_root.anchor)
                 and self.storage_root.is_dir(), 'storage root must be an existing data directory')
        _within(self.project_root, self.storage_root, equal=True)
        _within(self.ledger, self.storage_root)
        _within(self.model, self.storage_root)
        _require(self.authority.is_absolute() and self.lock_root.is_absolute()
                 and self.lock_root != Path(self.lock_root.anchor), 'authority and lock root must be explicit paths')

    def binding(self):
        # Card selection changes the per-GPU marker, never the shared budget.
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()
                if key != 'gpu_index'}


class CreatorGuard:
    def __init__(self, config: GuardConfig):
        config.validate()
        self.config = config

    def _authority(self):
        cfg = self.config
        _require(cfg.authority.is_file() and cfg.authority.stat().st_size <= 65536, 'missing or oversized authorization')
        authority_bytes = cfg.authority.read_bytes()
        _require(len(authority_bytes) <= 65536, 'oversized authorization')
        authority = _loads(authority_bytes.decode('utf-8'))
        _require(isinstance(authority, dict) and authority.get('status') == 'ACTIVE'
                 and authority.get('project') == 'InterActWorld' and authority.get('until_user_stop') is True,
                 'an existing ACTIVE InterActWorld authorization is required')
        _require(isinstance(authority.get('authorization_id'), str) and authority['authorization_id'], 'authorization ID missing')
        _require(authority.get('total_gpu_hours_limit') == 160, 'authorization GPU-hour scope differs')
        limit = authority.get('total_storage_bytes_limit', 300_000_000_000)
        _require(type(limit) is int and limit in (300_000_000_000, MAX_TOTAL_STORAGE_BYTES)
                 and cfg.storage_limit <= limit, 'configured storage ceiling is not authorized')
        hosts = authority.get('hosts')
        host = hosts.get(gpu_gate.SHARED_PROFILE) if isinstance(hosts, dict) else None
        _require(isinstance(host, dict) and str(host.get('hostname', '')).lower() == socket.gethostname().lower(),
                 'shared-server authorization does not name this host')
        if 'gpus' in host:
            cards = host['gpus']
            _require(cfg.gpu_index is not None, 'multi-card authority requires an explicit configured gpu_index')
            _require(isinstance(cards, list) and cards and all(isinstance(card, dict) for card in cards),
                     'invalid authorized GPU list')
            _require(all(type(card.get('gpu_index')) is int and card['gpu_index'] >= 0
                         and isinstance(card.get('gpu_uuid'), str) for card in cards), 'invalid authorized GPU identity')
            _require(len({card['gpu_index'] for card in cards}) == len(cards)
                     and len({card['gpu_uuid'].lower() for card in cards}) == len(cards), 'duplicate authorized GPU identity')
            selected = [card for card in cards if card['gpu_index'] == cfg.gpu_index]
            _require(len(selected) == 1, 'configured GPU index is not authorized')
            host = {**selected[0], 'hostname': host['hostname']}
        _require(type(host.get('gpu_index')) is int and host['gpu_index'] >= 0, 'the user-authorized GPU index is missing')
        _require(cfg.gpu_index is None or host['gpu_index'] == cfg.gpu_index, 'configured GPU index is not authorized')
        gpu_uuid = host.get('gpu_uuid')
        _require(isinstance(gpu_uuid, str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu_uuid),
                 'a fixed authorized GPU UUID is required')
        threshold = host.get('desktop_memory_max_exclusive')
        _require(type(threshold) is int and 1 <= threshold <= 500, 'shared authorization requires memory below 500 MiB')
        return authority, host, hashlib.sha256(authority_bytes).hexdigest()

    def _ledger(self):
        if not self.config.ledger.exists():
            return dict(version=1, binding=self.config.binding(), storage_bytes_upper=0, reservations={})
        ledger = _read(self.config.ledger)
        _require(isinstance(ledger, dict) and ledger.get('version') == 1
                 and ledger.get('binding') == self.config.binding() and isinstance(ledger.get('reservations'), dict),
                 'ledger configuration differs; explicit accounting reconciliation is required')
        _require(type(ledger.get('storage_bytes_upper')) is int and ledger['storage_bytes_upper'] >= 0,
                 'invalid storage ledger')
        return ledger

    def _usage(self, ledger):
        seconds = 0.0
        for record in ledger['reservations'].values():
            _require(record.get('status') in ('reserved', 'settled'), 'unknown ledger reservation status')
            amount = record.get('charged_seconds') if record['status'] == 'settled' else record.get('reserved_seconds')
            _require(_number(amount) and amount >= 0, 'invalid GPU accounting record')
            seconds += amount
        return seconds / 3600

    def _storage_ok(self, storage):
        _require(storage <= MAX_NEW_STORAGE_BYTES, 'this round exceeds its 100GB added-storage ceiling')
        total = self.config.prior_storage_bytes + storage
        _require(total <= self.config.storage_limit, 'retained prior plus current storage exceeds the authorized ceiling')
        return total

    def _storage_increment(self, ledger):
        """Add every open job's positive growth exactly once, across all cards."""
        storage = ledger['storage_bytes_upper']
        for record in ledger['reservations'].values():
            if record['status'] != 'reserved':
                continue
            previous = record.get('job_bytes_accounted', record['job_bytes_base'])
            current = _directory_bytes(record['context']['job_directory'])
            storage += max(0, current - previous)
            # Deletions never reclaim the conservative recorded storage bound.
            record['job_bytes_accounted'] = max(previous, current)
        return storage

    def _marker(self, gpu_uuid):
        return self.config.lock_root / (gpu_uuid.lower() + '.lease.json')

    def _query(self, lease):
        return gpu_gate.query_shared_gpu(confirmed_index=lease['gpu_index'], confirmed_uuid=lease['gpu_uuid'],
                                         profile=lease['profile'], memory_max_exclusive=lease['desktop_memory_max_exclusive'])

    def _context(self, payload):
        cfg = self.config
        _require(isinstance(payload, dict), 'guard input must be an object')
        _require(Path(payload.get('project_root', '')).resolve() == cfg.project_root, 'guard project root mismatch')
        job = payload.get('job_id')
        _require(isinstance(job, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', job), 'invalid job ID')
        directory = _within(payload.get('job_directory', ''), cfg.project_root)
        _require(directory.is_dir() and directory.name == job, 'job directory identity mismatch')
        digest = payload.get('request_sha256')
        _require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest)
                 and _sha(directory / 'request.json') == digest, 'job request changed')
        maximum = payload.get('max_seconds')
        _require(type(maximum) is int and 1 <= maximum <= 3600, 'invalid worker time bound')
        return dict(job_id=job, request_sha256=digest, job_directory=str(directory),
                    project_root=str(cfg.project_root), max_seconds=maximum)

    def dispatch(self, payload):
        _require_linux()
        _require(isinstance(payload, dict) and payload.get('operation') in ('reserve', 'check', 'settle'), 'unknown guard operation')
        # Test/missing/paused authority is rejected before creating lease artifacts.
        if payload['operation'] == 'reserve':
            self._authority()
        self.config.lock_root.mkdir(parents=True, exist_ok=True)
        self.config.ledger.parent.mkdir(parents=True, exist_ok=True)
        with _deadline(), _file_lock(self.config.ledger.with_suffix(self.config.ledger.suffix + '.lock')):
            ledger = self._ledger()
            context = self._context(payload)
            if payload['operation'] == 'reserve':
                return self._reserve(ledger, context)
            return self._existing(ledger, context, payload)

    def _reserve(self, ledger, context):
        authority, host, authority_sha = self._authority()
        _require(not any(record['context']['job_id'] == context['job_id'] for record in ledger['reservations'].values()),
                 'this job already has a reservation; never launch it twice')
        marker = self._marker(host['gpu_uuid'])
        _require(not marker.exists(), 'GPU already has a persistent lease; expiry alone does not release it')
        _require(not any(record['status'] == 'reserved' and record['lease']['gpu_uuid'] == host['gpu_uuid']
                         for record in ledger['reservations'].values()),
                 'GPU has an unresolved budget reservation; explicit reconciliation is required')
        # Cover worker time, backend/group cleanup and a conservative settlement
        # allowance. This is stricter than CommandGuard's minimum +30 seconds.
        seconds = context['max_seconds'] + 60
        round_upper = self._usage(ledger) + seconds / 3600
        _require(round_upper <= self.config.max_round_gpu_hours, 'Creator round GPU-hour budget exhausted')
        global_upper = self.config.prior_gpu_hours_upper + round_upper
        _require(global_upper <= 160, 'global GPU-hour budget exhausted')
        # Small ledger/lease metadata lives outside the active job directory.
        # Reserve its growth explicitly so incremental checks stay conservative.
        incremented = self._storage_increment(ledger)
        job_bytes = _directory_bytes(context['job_directory'])
        storage = max(incremented, _directory_bytes(self.config.storage_root) + STORAGE_METADATA_ALLOWANCE)
        storage_total = self._storage_ok(storage)
        now = time.time()
        reservation = uuid.uuid4().hex
        lease = dict(status='reserved', reservation_id=reservation, authorization_id=authority['authorization_id'],
                     authorization_record=str(self.config.authority), authorization_sha256=authority_sha,
                     profile=gpu_gate.SHARED_PROFILE, gpu_index=host['gpu_index'], gpu_uuid=host['gpu_uuid'],
                     desktop_memory_max_exclusive=host['desktop_memory_max_exclusive'],
                     job_id=context['job_id'], request_sha256=context['request_sha256'],
                     gpu_hours_reserved=seconds / 3600, global_gpu_hours_upper=global_upper,
                     global_storage_bytes_upper=storage_total, start_before_unix=now + min(900, context['max_seconds'] + 30),
                     deadline_unix=now + context['max_seconds'] + 30)
        # Hold the global budget lock during the last physical check and claim.
        self._query(lease)
        marker_value = dict(reservation_id=reservation, ledger=str(self.config.ledger),
                            project_root=str(self.config.project_root), gpu_uuid=host['gpu_uuid'])
        try:
            _write(marker, marker_value, exclusive=True)
        except FileExistsError as error:
            raise GuardError('GPU lease was claimed concurrently') from error
        record = dict(status='reserved', context=context, lease=lease, created_unix=now,
                      reserved_seconds=seconds, storage_base=storage, job_bytes_base=job_bytes,
                      job_bytes_accounted=job_bytes,
                      marker=marker_value)
        ledger['reservations'][reservation] = record
        ledger['storage_bytes_upper'] = storage
        # If writing fails, deliberately retain the marker for reconciliation.
        _write(self.config.ledger, ledger)
        return lease

    def _existing(self, ledger, context, payload):
        submitted = payload.get('lease')
        _require(isinstance(submitted, dict), 'lease is required')
        _require(self.config.gpu_index is None or submitted.get('gpu_index') == self.config.gpu_index,
                 'lease belongs to another configured GPU')
        record = ledger['reservations'].get(submitted.get('reservation_id'))
        _require(isinstance(record, dict) and record.get('lease') == submitted and record.get('context') == context,
                 'lease/job/request does not match the persistent reservation')
        if record['status'] == 'settled':
            _require(payload['operation'] == 'settle', 'reservation is already settled')
            return record['settlement']
        marker = self._marker(submitted['gpu_uuid'])
        _require(_read(marker, 65536) == record['marker'], 'GPU lease marker changed or disappeared')
        if payload['operation'] == 'check':
            authority, host, digest = self._authority()
            _require(digest == submitted['authorization_sha256'] and authority['authorization_id'] == submitted['authorization_id']
                     and all(host[key] == submitted[key] for key in ('gpu_uuid', 'gpu_index', 'desktop_memory_max_exclusive')),
                     'authorization changed or revoked')
            _require(time.time() <= submitted['deadline_unix'], 'reservation deadline exceeded; cleanup and explicit settlement required')
            storage = self._storage_increment(ledger)
            total = self._storage_ok(storage)
            _require(self._usage(ledger) <= self.config.max_round_gpu_hours
                     and self.config.prior_gpu_hours_upper + self._usage(ledger) <= 160, 'GPU-hour budget exhausted')
            ledger['storage_bytes_upper'] = storage
            _write(self.config.ledger, ledger)
            return dict(status='allowed', reservation_id=submitted['reservation_id'], global_storage_bytes_upper=total)
        elapsed = payload.get('elapsed_seconds')
        _require(_number(elapsed) and elapsed >= 0, 'settlement needs finite elapsed_seconds')
        _require(payload.get('outcome') in ('completed', 'failed', 'interrupted'), 'invalid settlement outcome')
        _require(payload.get('worker_returncode') is None or type(payload['worker_returncode']) is int, 'invalid worker return code')
        # Settlement remains possible after revocation, but only for this exact
        # persisted lease and only once the selected card is independently idle.
        self._query(submitted)
        charge = max(time.time() - record['created_unix'], elapsed) + 30
        storage = self._storage_increment(ledger)
        settlement = dict(status='settled', reservation_id=submitted['reservation_id'],
                          charged_gpu_seconds=charge, outcome=payload['outcome'])
        record.update(status='settled', charged_seconds=charge, settlement=settlement)
        ledger['storage_bytes_upper'] = storage
        # The global lock excludes any reservation in this ledger during commit.
        # Keep the on-disk record reserved until marker removal succeeds. If the
        # final write fails, restore only our absent marker; an unresolved ledger
        # record also independently blocks this GPU on the next reserve attempt.
        _require(_read(marker, 65536) == record['marker'], 'GPU lease marker changed before release')
        marker.unlink()
        try:
            _write(self.config.ledger, ledger)
        except BaseException:
            try:
                _write(marker, record['marker'], exclusive=True)
            except (OSError, GuardError):
                pass  # Never replace a marker another ledger may have claimed.
            raise
        return settlement


def _terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=1)


def run_model(guard, request_path, output_path):
    """Run a local model with the same reservation path as video generation."""
    _require_linux()
    cfg = guard.config
    request_path = _within(request_path, cfg.project_root)
    output_path = _within(output_path, cfg.project_root)
    _require(request_path.name == 'request.json' and output_path.parent == request_path.parent
             and not output_path.exists(), 'model request/output must share a fresh request directory')
    request = _read(request_path, 262144)
    _require(isinstance(request, dict) and request.get('kind') in ('plan', 'inspect'), 'unsupported model request kind')
    if request['kind'] == 'inspect':
        video = _within(request.get('video_path', ''), cfg.project_root)
        _require(video.is_file() and video.name == 'raw.mp4', 'visual inspection requires a project raw.mp4')
    _require(cfg.python.is_file() and cfg.model.is_dir() and (cfg.model / 'config.json').is_file(),
             'guarded Python and offline model must already exist')
    timeout_program = shutil.which('timeout')
    _require(timeout_program is not None, 'GNU timeout is required for an independent finite model watchdog')
    context = dict(job_id=request_path.parent.name, request_sha256=_sha(request_path),
                   job_directory=str(request_path.parent), project_root=str(cfg.project_root), max_seconds=MAX_MODEL_SECONDS)
    lease = guard.dispatch(dict(operation='reserve', **context))
    started, process, outcome = time.monotonic(), None, 'failed'
    old_handlers = {}
    def interrupted(_signum, _frame):
        nonlocal outcome
        outcome = 'interrupted'
        raise GuardError('model guard interrupted; terminating its owned process group')
    try:
        for kind in (signal.SIGTERM, signal.SIGINT):
            old_handlers[kind] = signal.signal(kind, interrupted)
        runtime = request_path.parent
        cache = runtime / 'cache'
        cache.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=lease['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                           HF_HUB_DISABLE_TELEMETRY='1', WANDB_DISABLED='true', WANDB_MODE='disabled',
                           HF_HOME=str(cache / 'huggingface'), HF_HUB_CACHE=str(cache / 'huggingface' / 'hub'),
                           XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache / 'torch'),
                           CUDA_CACHE_PATH=str(cache / 'cuda'), TRITON_CACHE_DIR=str(cache / 'triton'),
                           TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                           PYTORCH_KERNEL_CACHE_PATH=str(cache / 'kernels'),
                           TMPDIR=str(runtime), TMP=str(runtime), TEMP=str(runtime), TOKENIZERS_PARALLELISM='false')
        command = [timeout_program, '--signal=TERM', '--kill-after=2s', f'{MAX_MODEL_SECONDS}s',
                   str(cfg.python), '-m', 'training.creator.model_worker', '--model', str(cfg.model),
                   '--request', str(request_path), '--output', str(output_path)]
        _require(time.time() <= lease['start_before_unix'], 'model did not start within the reservation start window')
        guard.dispatch(dict(operation='check', lease=lease, **context))
        with (runtime / 'guard-worker.log').open('xb') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=Path(__file__).resolve().parents[2], env=environment, start_new_session=True)
            last_check = time.monotonic()
            while process.poll() is None:
                _require(time.monotonic() - started < MAX_MODEL_SECONDS, f'model exceeded its {MAX_MODEL_SECONDS}-second bound')
                if time.monotonic() - last_check >= 30:
                    guard.dispatch(dict(operation='check', lease=lease, **context))
                    last_check = time.monotonic()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            _require(process.returncode == 0, 'model worker failed; see guard-worker.log')
        response = _read(output_path, 262144)
        _require(isinstance(response, dict) and 'error' not in response, 'model worker produced an invalid/error response')
        outcome = 'completed'
        return response
    finally:
        try:
            if process is not None:
                _terminate(process)
            # Failure here intentionally leaves the lease open and makes the
            # provider fail even if a model output had already been written.
            guard.dispatch(dict(operation='settle', lease=lease, **context, outcome=outcome,
                                elapsed_seconds=time.monotonic() - started,
                                worker_returncode=None if process is None else process.returncode))
        finally:
            for kind, handler in old_handlers.items():
                signal.signal(kind, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-model', action='store_true')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        guard = CreatorGuard(GuardConfig.load(args.config))
        if args.run_model:
            _require(args.request is not None and args.output is not None, '--run-model requires --request and --output')
            run_model(guard, args.request, args.output)
        else:
            _require(args.request is None and args.output is None, 'request/output arguments require --run-model')
            text = sys.stdin.read(65537)
            _require(len(text) <= 65536, 'guard input is too large')
            result = guard.dispatch(_loads(text))
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'denied', 'error': type(error).__name__, 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
