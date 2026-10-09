"""Bounded, loopback-only action continuation studio; model loading is lazy.

The outer studio worker owns the GPU lease and process watchdog. CPU tests inject
an executor; the production path only uses ReusableActionEngine and real inputs.
"""
from __future__ import annotations

import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit
import uuid


KEYS = tuple('WASDIJKL')
IDENTITY = re.compile(r'[0-9a-f]{32}')
ACTIVE = ('queued', 'running')
MAX_JSON = 16_384
MAX_TIMELINE_SEGMENTS = 64
MAX_EXPORT_SECONDS = 180
CFG_EXECUTIONS = ('exact', 'delta_cached_v1')
R005_CHECKPOINT_SHA = '9bc9f57cce1095cbbc472307bc3963da2e6153e9f0f19080b542b6bf118dfaee'


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def check(condition, message, status=400):
    if not condition:
        raise APIError(status, message)


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def studio_profile(request):
    """Operator-only profiles, never arbitrary browser-selected dimensions."""
    width, height = request.get('width', 832), request.get('height', 480)
    check(type(width) is int and type(height) is int and (width, height) in ((832, 480), (512, 288)),
          'studio dimensions must be 832x480 or the optional 512x288 preview')
    reduced = (width, height) == (512, 288)
    quality = request.get('quality_settings')
    allow_cfg_delta = request.get('allow_cfg_delta', False)
    check(type(allow_cfg_delta) is bool, 'allow_cfg_delta is an operator-only boolean')
    negative_cache = None
    if quality is not None:
        check(isinstance(quality, dict) and set(quality) == {'cfg_scale', 'lora_strength', 'action_scale'},
              'quality_settings requires cfg_scale, lora_strength and action_scale')
        for key, values in [('cfg_scale', (1, 3, 5)), ('lora_strength', (.5, 1)), ('action_scale', (.03,))]:
            check(type(quality[key]) in (int, float) and quality[key] in values, f'invalid quality {key}')
        check(not reduced, 'quality settings require the full 832x480 profile')
        check(request.get('fastpath', False) is False, 'quality text CFG cannot use fixed-text fastpath')
        negative_cache = request.get('negative_cache')
        check(isinstance(negative_cache, str) and (negative_cache.startswith('/') or Path(negative_cache).is_absolute())
              and Path(negative_cache).suffix == '.pt', 'quality settings require an explicit absolute negative_cache .pt path')
    if allow_cfg_delta:
        check((width, height) == (832, 480)
              and quality == dict(cfg_scale=5, lora_strength=1, action_scale=.03)
              and request.get('fastpath', False) is False,
              'optional CFG cache requires 832x480 / 40 Euler / CFG5 / LoRA1 / action0.03 / no fastpath')
    return dict(width=width, height=height,
                profile='quality_candidate' if quality is not None else 'low_resolution_preview' if reduced else 'full_resolution',
                experimental_label='40步画质候选 · 可能更慢 · 提升尚未证实 · 非实时' if quality is not None else
                    '低清预览候选 · 仍有形变与场景漂移 · 非实时' if reduced else '原分辨率 · 提速配置质量仍需观察 · 非实时',
                quality_settings=copy.deepcopy(quality), negative_cache=negative_cache,
                allow_cfg_delta=allow_cfg_delta,
                allowed_cfg_executions=list(CFG_EXECUTIONS) if allow_cfg_delta else ['exact'],
                default_cfg_execution='exact',
                default_steps=40 if quality is not None else 20,
                allowed_steps=[40] if quality is not None else [20] if reduced else [40, 20, 10])


def check_cfg_source(profile, source_identity):
    """Enabling the experimental branch needs the already-verified R005 pin."""
    if profile['allow_cfg_delta']:
        check(source_identity is not None and source_identity['checkpoint_sha256'] == R005_CHECKPOINT_SHA,
              'optional CFG cache requires the verified original R005 source identity', 500)


def segment_cfg_execution(segment):
    mode = segment.get('cfg_execution', 'exact')
    check(isinstance(mode, str) and mode in CFG_EXECUTIONS, 'invalid cfg_execution')
    return mode


def validated_source_identity(value):
    """Operator-owned binding; None is only for legacy stores/CPU fixtures."""
    if value is None:
        return None
    fields = {'schema_version', 'checkpoint_sha256', 'source_job', 'source_request_sha256',
              'initial_sha256', 'scene_id', 'prompt'}
    check(isinstance(value, dict) and set(value) == fields
          and type(value['schema_version']) is int and value['schema_version'] == 1,
          'invalid studio source identity schema', 500)
    check(all(isinstance(value[key], str) and re.fullmatch(r'[0-9a-f]{64}', value[key])
              for key in ('checkpoint_sha256', 'source_request_sha256', 'initial_sha256')),
          'invalid studio source identity digests', 500)
    check(isinstance(value['source_job'], str)
          and (value['source_job'].startswith('/') or Path(value['source_job']).is_absolute())
          and isinstance(value['scene_id'], str) and value['scene_id']
          and isinstance(value['prompt'], str), 'invalid studio source identity inputs', 500)
    return copy.deepcopy(value)


def source_identity_from_verified_record(request, record_bytes, lineage, scene):
    """Use read_source's already-verified pins; hash only the small request JSON.

    The request binds rollout configuration, initial RGB and original controls.
    A different source path/record intentionally requires a new session root;
    legacy sessions are not silently assigned a newly inferred provenance.
    """
    original = json.loads(record_bytes)
    check(original.get('checkpoint_sha256') == lineage['sha256']
          and original.get('prompt') == scene['prompt'],
          'verified source record does not match loaded checkpoint/prompt', 500)
    return validated_source_identity(dict(schema_version=1, checkpoint_sha256=lineage['sha256'],
        source_job=str(Path(request['source_job']).resolve()),
        source_request_sha256=hashlib.sha256(record_bytes).hexdigest(),
        initial_sha256=original.get('initial_sha256'), scene_id=scene['scene_id'], prompt=scene['prompt']))


class StudioStore:
    """Immutable completed segments; only the owned worker changes job status."""

    def __init__(self, directory, scene, *, deadline, minimum_task_seconds=600, width=832, height=480,
                 quality_settings=None, negative_cache=None, fastpath=False, source_identity=None,
                 allow_cfg_delta=False):
        self.directory = Path(directory).resolve()
        self.root = self.directory / 'sessions'
        self.root.mkdir(parents=True, exist_ok=True)
        self.scene = dict(scene)
        self.source_identity = validated_source_identity(source_identity)
        check(self.source_identity is None or (self.source_identity['scene_id'] == self.scene['scene_id']
              and self.source_identity['prompt'] == self.scene['prompt']),
              'studio source identity does not match the configured scene', 500)
        self.profile = studio_profile(dict(width=width, height=height, quality_settings=quality_settings,
                                           negative_cache=negative_cache, fastpath=fastpath,
                                           allow_cfg_delta=allow_cfg_delta))
        check_cfg_source(self.profile, self.source_identity)
        self.deadline = deadline
        self.minimum_task_seconds = minimum_task_seconds
        self.lock = threading.RLock()
        self.export_lock = threading.Lock()
        self.sessions = {}
        self.queue = queue.Queue()
        self.stopping = threading.Event()
        self.current = None
        self._restore()

    def _restore(self):
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or not IDENTITY.fullmatch(directory.name):
                continue
            source = directory / 'session.json'
            if not source.is_file():
                continue
            session = json.loads(source.read_text(encoding='utf-8'))
            check(session.get('session_id') == directory.name and session.get('scene_id') == self.scene['scene_id'],
                  'persisted session identity/scene mismatch', 500)
            check((session.get('width', 832), session.get('height', 480)) ==
                  (self.profile['width'], self.profile['height']) and session.get('steps') in self.profile['allowed_steps'],
                  'persisted session does not match the operator resolution profile', 500)
            check(session.get('quality_settings') == self.profile['quality_settings']
                  and session.get('negative_cache') == self.profile['negative_cache'],
                  'persisted session does not match the operator quality settings or negative cache', 500)
            check(session.get('source_identity') == self.source_identity,
                  'persisted session source identity mismatch or missing; use a new session directory, '
                  'legacy identity is never inferred', 500)
            session.update(copy.deepcopy(self.profile))
            check(isinstance(session.get('segments'), list), 'invalid persisted segments', 500)
            seen = set()
            for segment in session['segments']:
                segment_cfg_execution(segment)  # Missing legacy field means exact; never rewrite old results.
                identity = segment.get('segment_id', '')
                check(isinstance(identity, str) and IDENTITY.fullmatch(identity) and identity not in seen,
                      'invalid persisted segment identity', 500)
                parent = segment.get('parent_segment_id')
                check(parent is None or parent in seen, 'invalid persisted segment parent', 500)
                seen.add(identity)
                if segment.get('status') in ACTIVE:
                    segment.update(status='failed', phase='interrupted', video_url=None,
                                   error='服务曾中断；未完成任务不会自动重新运行。', ended_unix=time.time())
                    atomic_json(self.segment_directory(session['session_id'], identity) / 'job.json', segment)
            session['active_job_id'] = None
            self.sessions[directory.name] = session
            self._save(session)

    def _save(self, session):
        atomic_json(self.root / session['session_id'] / 'session.json', session)

    def _session(self, identity):
        check(isinstance(identity, str) and IDENTITY.fullmatch(identity), 'invalid session ID', 404)
        check(identity in self.sessions, 'session not found', 404)
        return self.sessions[identity]

    def get(self, identity):
        with self.lock:
            return copy.deepcopy(self._session(identity))

    def create(self, payload):
        check(isinstance(payload, dict) and set(payload) <= {'scene_id', 'seed', 'steps'}, 'invalid session fields')
        check(payload.get('scene_id') == self.scene['scene_id'], 'only the configured frozen scene is available')
        seed, steps = payload.get('seed'), payload.get('steps', self.profile['default_steps'])
        check(type(seed) is int and 0 <= seed <= 2**32 - 1, 'seed must be an unsigned 32-bit integer')
        check(type(steps) is int and steps in self.profile['allowed_steps'],
              'steps are not permitted by the operator resolution profile')
        with self.lock:
            check(not self.stopping.is_set() and time.monotonic() < self.deadline, 'bounded service has expired', 503)
            identity = uuid.uuid4().hex
            session = dict(session_id=identity, scene_id=self.scene['scene_id'], prompt=self.scene['prompt'],
                           seed=seed, steps=steps, created_unix=time.time(), segments=[], active_job_id=None)
            session.update(copy.deepcopy(self.profile))
            if self.source_identity is not None:
                session['source_identity'] = copy.deepcopy(self.source_identity)
            (self.root / identity).mkdir(exist_ok=False)
            self.sessions[identity] = session
            self._save(session)
            return copy.deepcopy(session)

    def segment_directory(self, session_id, segment_id):
        check(isinstance(session_id, str) and IDENTITY.fullmatch(session_id), 'invalid session ID', 404)
        check(isinstance(segment_id, str) and IDENTITY.fullmatch(segment_id), 'invalid segment ID', 404)
        path = self.root / session_id / segment_id
        check(path.resolve().is_relative_to(self.root.resolve()), 'path outside session root', 403)
        return path

    def submit(self, identity, payload):
        check(isinstance(payload, dict) and set(payload) <= {'parent_segment_id', 'duration_s', 'keys', 'request_id', 'cfg_execution'},
              'invalid segment fields')
        cfg_execution = segment_cfg_execution(payload)
        duration, keys = payload.get('duration_s'), payload.get('keys')
        check(type(duration) is int and duration in (3, 6), 'duration_s must be 3 or 6')
        check(isinstance(keys, list) and all(type(key) is str and key in KEYS for key in keys)
              and len(keys) == len(set(keys)), 'keys must be unique entries from WASDIJKL')
        check(not any(a in keys and b in keys for a, b in [('W', 'S'), ('A', 'D'), ('I', 'K'), ('J', 'L')]),
              'opposite keys cannot be held together')
        keys = [key for key in KEYS if key in keys]
        parent = payload.get('parent_segment_id')
        check(parent is None or isinstance(parent, str) and IDENTITY.fullmatch(parent), 'invalid parent segment')
        request_id = payload.get('request_id')
        check(request_id is None or isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{8,100}', request_id),
              'invalid request_id')
        with self.lock:
            session = self._session(identity)
            if request_id is not None:
                old = next((item for item in session['segments'] if item.get('request_id') == request_id), None)
                if old is not None:
                    check(old['parent_segment_id'] == parent and old['duration_s'] == duration and old['actions'] == keys
                          and segment_cfg_execution(old) == cfg_execution,
                          'request_id already used with different actions, parent or cfg_execution', 409)
                    return copy.deepcopy(old)
            check(cfg_execution in self.profile['allowed_cfg_executions'],
                  'CFG cache is not enabled by the operator', 403)
            check(not self.stopping.is_set() and self.deadline - time.monotonic() >= self.minimum_task_seconds,
                  '剩余服务时长不足以安全接收新任务；已有片段仍可查看。', 503)
            check(self.current is None and not any(s['active_job_id'] for s in self.sessions.values()),
                  '另一个生成任务仍在运行，请等待或取消。', 409)
            if parent is not None:
                prior = next((item for item in session['segments'] if item['segment_id'] == parent), None)
                check(prior is not None and prior['status'] == 'completed', 'parent must be completed in this session')
                check((self.segment_directory(identity, parent) / 'endpoint.npy').is_file(), 'parent endpoint is missing', 409)
            segment_id = uuid.uuid4().hex
            target = self.segment_directory(identity, segment_id)
            target.mkdir(exist_ok=False)
            segment = dict(segment_id=segment_id, parent_segment_id=parent, request_id=request_id,
                           status='queued', phase='queued', duration_s=duration, actions=keys,
                           cfg_execution=cfg_execution,
                           seed=session['seed'], steps=session['steps'], queued_unix=time.time(),
                           elapsed_seconds=None, error=None, video_url=None, cancel_requested=False)
            session['segments'].append(segment)
            session['active_job_id'] = segment_id
            atomic_json(target / 'job.json', segment)
            self._save(session)
            self.queue.put((identity, segment_id))
            return copy.deepcopy(segment)

    def _segment(self, session_id, segment_id):
        session = self._session(session_id)
        segment = next((item for item in session['segments'] if item['segment_id'] == segment_id), None)
        check(segment is not None, 'segment not found', 404)
        return session, segment

    def update(self, session_id, segment_id, **values):
        with self.lock:
            session, segment = self._segment(session_id, segment_id)
            segment.update(values)
            atomic_json(self.segment_directory(session_id, segment_id) / 'job.json', segment)
            self._save(session)

    def cancel(self, identity):
        with self.lock:
            session = self._session(identity)
            active = session['active_job_id']
            if active is None:
                return dict(status='no_active_job')
            _, segment = self._segment(identity, active)
            segment['cancel_requested'] = True
            if segment['status'] == 'queued':
                segment.update(status='cancelled', phase='cancelled_before_start', ended_unix=time.time())
                session['active_job_id'] = None
            else:
                segment['phase'] = 'cancel_requested_waiting_for_segment_end'
            atomic_json(self.segment_directory(identity, active) / 'job.json', segment)
            self._save(session)
            return dict(status=segment['status'], cancel_requested=True,
                        message='取消是协作式的，最迟当前片段生成结束生效；不会发布被取消的结果。')

    def stop(self):
        with self.lock:
            self.stopping.set()
            for identity in list(self.sessions):
                self.cancel(identity)

    def media(self, session_id, segment_id):
        with self.lock:
            _, segment = self._segment(session_id, segment_id)
            check(segment['status'] == 'completed', 'video is not a completed result', 404)
            return self.segment_directory(session_id, segment_id) / 'video.mp4'

    def timeline_sources(self, session_id, segment_id):
        """Snapshot one completed ancestry; never accept browser-supplied paths."""
        with self.lock:
            session = self._session(session_id)
            by_id = {item['segment_id']: item for item in session['segments']}
            check(len(by_id) == len(session['segments']), 'duplicate segment identities', 409)
            chain, seen = [], set()
            current = segment_id
            while current is not None:
                check(isinstance(current, str) and IDENTITY.fullmatch(current), 'invalid branch identity', 404)
                check(current not in seen, 'branch contains a parent cycle', 409)
                check(len(chain) < MAX_TIMELINE_SEGMENTS, 'branch exceeds the bounded export length', 413)
                seen.add(current)
                item = by_id.get(current)
                check(item is not None, 'branch parent is missing from this session', 409)
                check(item['status'] == 'completed', 'branch contains an incomplete segment', 409)
                check(type(item.get('duration_s')) is int and item['duration_s'] in (3, 6),
                      'branch has an invalid recorded duration', 409)
                source = self.segment_directory(session_id, current) / 'video.mp4'
                check(source.resolve() == source.absolute() and not source.is_symlink(),
                      'branch source must not redirect to another path', 403)
                check(source.is_file() and source.stat().st_size > 0, 'branch video is missing', 409)
                chain.append(dict(segment_id=current, duration_s=item['duration_s'], source=source))
                current = item.get('parent_segment_id')
            check(bool(chain), 'no completed branch selected', 404)
            return list(reversed(chain))

    def _run_export(self, command, deadline):
        """Only an owned CPU child; shutdown and the finite deadline stop it."""
        check(not self.stopping.is_set() and time.monotonic() < deadline, 'export window has ended', 503)
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            try:
                while True:
                    check(not self.stopping.is_set(), 'service stopped during branch export', 503)
                    remaining = deadline - time.monotonic()
                    check(remaining > 0, 'branch export exceeded its time limit', 504)
                    try:
                        stdout, stderr = process.communicate(timeout=min(.25, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode:
                    print('[studio-export-error] ' + stderr.decode('utf-8', errors='replace')[-2000:], flush=True)
                    raise APIError(500, '视频合并或解码检查失败；原片段保留，未发布不完整结果。')
                return stdout.decode('utf-8', errors='replace')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

    def timeline(self, session_id, segment_id):
        """CPU-only cached MP4; every child drops just its conditioning frame."""
        chain = self.timeline_sources(session_id, segment_id)
        target = self.segment_directory(session_id, segment_id)
        output = target / 'timeline.mp4'
        check(output.resolve() == output.absolute() and not output.is_symlink(), 'unsafe export target', 403)
        if output.is_file() and output.stat().st_size > 0:
            return output
        check(self.export_lock.acquire(blocking=False), '另一个完整分支正在合并，请稍后重试。', 409)
        temporary = None
        try:
            # A second HTTP request may have finished between snapshot and lock.
            if output.is_file() and output.stat().st_size > 0:
                return output
            check(not output.exists(), 'invalid existing export target', 409)
            deadline = min(self.deadline - 2, time.monotonic() + MAX_EXPORT_SECONDS)
            check(not self.stopping.is_set() and deadline - time.monotonic() >= 5,
                  '剩余服务时长不足以合并；原片段仍然保留。', 503)
            ffmpeg = shutil.which('ffmpeg')
            check(ffmpeg is not None, 'ffmpeg is not available for CPU export', 503)
            temporary = target / f'timeline-{uuid.uuid4().hex}.partial.mp4'
            check(temporary.resolve() == temporary.absolute(), 'unsafe temporary export target', 403)
            command = [ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-n', '-filter_complex_threads', '2']
            filters, labels = [], []
            for index, item in enumerate(chain):
                command.extend(['-threads', '2', '-i', str(item['source'])])
                trim = '' if index == 0 else 'trim=start_frame=1,'
                filters.append(f'[{index}:v:0]{trim}setpts=PTS-STARTPTS[part{index}]')
                labels.append(f'[part{index}]')
            filters.append(''.join(labels) + f'concat=n={len(chain)}:v=1:a=0[out]')
            command.extend(['-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                            '-r', '16', '-threads', '2', '-movflags', '+faststart',
                            '-progress', 'pipe:1', '-nostats', str(temporary)])
            started = time.monotonic()
            expected_frames = 1 + sum(item['duration_s'] * 16 for item in chain)
            encoded = self._run_export(command, deadline)
            frames = re.findall(r'^frame=\s*(\d+)\s*$', encoded, re.MULTILINE)
            check(frames and int(frames[-1]) == expected_frames, 'export frame count does not match branch', 500)
            decode_command = [ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
                              '-i', str(temporary), '-map', '0:v:0', '-an', '-threads', '2',
                              '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
            decoded = self._run_export(decode_command, deadline)
            frames = re.findall(r'^frame=\s*(\d+)\s*$', decoded, re.MULTILINE)
            check(frames and int(frames[-1]) == expected_frames, 'export full decode count does not match branch', 500)
            check(not self.stopping.is_set() and time.monotonic() < deadline, 'export window has ended', 503)
            check(temporary.is_file() and temporary.stat().st_size > 0, 'export produced no MP4', 500)
            atomic_json(target / 'timeline.json', dict(
                segment_ids=[item['segment_id'] for item in chain], frames=expected_frames, fps=16,
                duration_seconds=expected_frames / 16, child_conditioning_frames_dropped=len(chain) - 1,
                elapsed_seconds=time.monotonic() - started, created_unix=time.time(),
                commands=[command, decode_command], decode_check='all_expected_frames_decoded',
                source_clips_unchanged=True, generation_triggered=False))
            os.replace(temporary, output)
            return output
        finally:
            if temporary is not None and temporary.is_file():
                temporary.unlink()  # Only this attempt's unpublished derived file.
            self.export_lock.release()


class GenerationWorker(threading.Thread):
    """One executor at a time. Executors return JSON report + float CPU endpoint."""

    def __init__(self, store, executor, *, idle_seconds=180):
        super().__init__(name='interactworld-studio-generator', daemon=True)
        self.store, self.executor = store, executor
        self.idle_seconds = idle_seconds
        self.last_used = time.monotonic()
        self.fatal_error = None

    def run(self):
        try:
            while not self.store.stopping.is_set():
                try:
                    identity, segment_id = self.store.queue.get(timeout=.5)
                except queue.Empty:
                    if time.monotonic() - self.last_used >= self.idle_seconds:
                        self.executor.close()
                        self.last_used = time.monotonic()
                    continue
                try:
                    self._execute(identity, segment_id)
                finally:
                    self.store.queue.task_done()
                    self.last_used = time.monotonic()
        except BaseException as error:
            self.fatal_error = f'{type(error).__name__}: {error}'
            self.store.stop()
        finally:
            try:
                self.executor.close()
            except BaseException as error:
                self.fatal_error = self.fatal_error or f'engine close failed: {error}'

    def _execute(self, identity, segment_id):
        with self.store.lock:
            session, segment = self.store._segment(identity, segment_id)
            if segment['status'] != 'queued' or self.store.stopping.is_set():
                return
            self.store.current = (identity, segment_id)
            started = time.monotonic()
            self.store.update(identity, segment_id, status='running', phase='checking_authority', started_unix=time.time())
            frozen_session, frozen_segment = copy.deepcopy(session), copy.deepcopy(segment)
        target = self.store.segment_directory(identity, segment_id)
        try:
            def phase(value):
                with self.store.lock:
                    _, current = self.store._segment(identity, segment_id)
                    if not current['cancel_requested']:
                        self.store.update(identity, segment_id, phase=value)

            report, endpoint = self.executor(frozen_session, frozen_segment, target, phase)
            with self.store.lock:
                session, segment = self.store._segment(identity, segment_id)
                if segment['cancel_requested'] or self.store.stopping.is_set():
                    segment.update(status='cancelled', phase='cancelled', video_url=None,
                                   error='生成期间收到取消请求，结果未作为可续写片段发布。')
                else:
                    requested_cfg = segment_cfg_execution(segment)
                    actual_cfg = report.get('cfg_execution', 'exact')
                    check(actual_cfg == requested_cfg and (requested_cfg == 'exact' or 'cfg_execution' in report),
                          'actual generation CFG mode differs from submitted mode', 500)
                    check(report.get('outcome', 'completed') == 'completed',
                          'executor returned an unsuccessful generation report', 500)
                    if requested_cfg == 'delta_cached_v1':
                        check(report.get('actual_model_forward_calls') == 64
                              and report.get('approximate_inference') is True,
                              'cached generation lacks actual execution evidence', 500)
                    check((target / 'video.mp4').is_file() and (target / 'video.mp4').stat().st_size > 0,
                          'executor returned without a real video file', 500)
                    import numpy as np
                    values = endpoint.detach().cpu().float().numpy() if hasattr(endpoint, 'detach') else np.asarray(endpoint)
                    check(values.dtype == np.float32 and values.shape == (1, 3, 1, session['height'], session['width'])
                          and np.isfinite(values).all() and values.min() >= -1 and values.max() <= 1,
                          'invalid float RGB endpoint', 500)
                    temporary = target / 'endpoint.npy.tmp'
                    with temporary.open('xb') as stream:
                        np.save(stream, values, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target / 'endpoint.npy')
                    segment.update(status='completed', phase='completed',
                                   video_url=f'/media/sessions/{identity}/{segment_id}/video.mp4')
                segment['generation'] = report
        except Exception as error:
            with self.store.lock:
                session, segment = self.store._segment(identity, segment_id)
                segment.update(status='cancelled' if segment['cancel_requested'] else 'failed', phase='stopped',
                               video_url=None, error=f'{type(error).__name__}: {error}')
            self.executor.close()  # Failed runtime is never silently reused.
        finally:
            with self.store.lock:
                session, segment = self.store._segment(identity, segment_id)
                segment.update(elapsed_seconds=time.monotonic() - started, ended_unix=time.time())
                session['active_job_id'] = None
                self.store.current = None
                atomic_json(target / 'job.json', segment)
                self.store._save(session)


class RealExecutor:
    def __init__(self, assets, request, lease):
        self.assets, self.request, self.lease = assets, request, lease
        self.source_identity = validated_source_identity(assets.get('source_identity'))
        self.profile = studio_profile(request)
        check_cfg_source(self.profile, self.source_identity)
        self.engine = None
        self.quality_runtime = None
        self.negative_source = None
        self.ever_loaded = False

    def _check_device(self):
        from training.gpu_gate import _run_nvidia_smi
        identity = self.lease['gpu_uuid']
        rows = [line.strip() for line in _run_nvidia_smi([
            '--id=' + identity, '--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader,nounits']).splitlines() if line.strip()]
        check(len(rows) == 1, 'authorized GPU inventory is ambiguous', 503)
        fields = [part.strip() for part in rows[0].split(',')]
        check(len(fields) == 3 and int(fields[0]) == self.lease['gpu_index']
              and fields[1].lower() == identity.lower(), 'authorized GPU identity changed', 503)
        memory = int(fields[2])
        pids = []
        for line in _run_nvidia_smi(['--id=' + identity, '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits']).splitlines():
            if not line.strip():
                continue
            row = [part.strip() for part in line.split(',')]
            check(len(row) == 2 and row[0].lower() == identity.lower(), 'unexpected compute inventory', 503)
            pids.append(int(row[1]))
        check(all(pid == os.getpid() for pid in pids), 'authorized GPU has another compute process', 503)
        if not self.ever_loaded:
            check(not pids, 'first model load requires no existing compute process', 503)
        if self.engine is None:
            check(0 <= memory < self.lease['desktop_memory_max_exclusive'],
                  'model load/reload requires memory below authorized free-card threshold', 503)
        return dict(gpu_uuid=identity, gpu_index=self.lease['gpu_index'], memory_used_mib=memory,
                    compute_pids=pids, own_worker_pid=os.getpid(), model_already_loaded=self.engine is not None)

    def __call__(self, session, segment, target, phase):
        check(session.get('source_identity') == self.source_identity,
              'session source identity differs from the loaded executor', 409)
        import numpy as np
        import torch
        from training.demo.backend import validate_active_authority
        from training.demo.latency_engine import ReusableActionEngine
        check((session['width'], session['height']) == (self.profile['width'], self.profile['height'])
              and session['steps'] in self.profile['allowed_steps'], 'session resolution profile mismatch', 409)
        check(session.get('quality_settings') == self.profile['quality_settings']
              and session.get('negative_cache') == self.profile['negative_cache'], 'session quality profile mismatch', 409)
        cfg_execution = segment_cfg_execution(segment)
        check(cfg_execution in self.profile['allowed_cfg_executions'],
              'CFG cache is not enabled by the operator', 403)
        validate_active_authority(self.lease)
        atomic_json(target / 'gpu-state.json', self._check_device())
        if self.engine is None:
            phase('loading_model')
            self.ever_loaded = True
            if self.profile['quality_settings'] is None:
                self.engine = ReusableActionEngine(self.assets['state'], self.assets['config'],
                                                   residency=self.request.get('residency', 'phase'),
                                                   fastpath=self.request.get('fastpath', False),
                                                   width=self.profile['width'], height=self.profile['height'])
            else:
                from training.studio.quality_probe import load_negative, QualityRuntime
                negative, source = load_negative(self.profile['negative_cache'], self.assets['config'].base_model_path)
                residency = self.request.get('residency', 'phase')
                began = time.monotonic()
                runtime_class = QualityRuntime
                if self.profile['allow_cfg_delta']:
                    from training.studio.cfg_optional_runtime import OptionalCFGRuntime
                    runtime_class = OptionalCFGRuntime
                runtime = runtime_class(self.assets['state'], self.assets['config'], residency, negative)
                try:
                    runtime.synchronize()
                    cold_seconds = time.monotonic() - began
                    self.engine = ReusableActionEngine(self.assets['state'], self.assets['config'],
                                                       residency=residency, fastpath=False, _runtime=runtime)
                    self.engine.load_seconds = cold_seconds
                except BaseException:
                    runtime.close()
                    raise
                self.quality_runtime, self.negative_source = runtime, source
        parent = segment['parent_segment_id']
        if parent is None:
            initial = self.assets['initial']
        else:
            # Parent IDs are validated against this session, never browser paths.
            parent_file = target.parent / parent / 'endpoint.npy'
            initial = torch.from_numpy(np.load(parent_file, allow_pickle=False).copy())
        actions = np.zeros((segment['duration_s'] * 16, 8), dtype=np.float32)
        for key in segment['actions']:
            actions[:, KEYS.index(key)] = 1
        np.save(target / 'actions.npy', actions, allow_pickle=False)
        phase('generating_and_decoding')
        options = dict(initial=initial, prompt=self.assets['prompt'], actions=actions,
                       seed=session['seed'], output_path=target / 'video.mp4', steps=session['steps'])
        if self.quality_runtime is None:
            return self.engine.run_segment(**options)
        from training.studio.quality_probe import annotate_report
        case = dict(self.profile['quality_settings'], steps=session['steps'])
        if self.profile['allow_cfg_delta']:
            self.quality_runtime.configure(case, cfg_execution=cfg_execution)
        else:
            self.quality_runtime.configure(case)
        self.engine.action_scale = case['action_scale']
        self.engine.last_run_report = None
        try:
            report, endpoint = self.engine.run_segment(**options)
            expected = 64 if cfg_execution == 'delta_cached_v1' else session['steps'] * (1 if case['cfg_scale'] == 1 else 2)
            check(self.quality_runtime.forward_calls == expected, 'quality forward count differs from requested CFG', 500)
            return report, endpoint
        finally:
            if self.engine.last_run_report is not None:
                annotate_report(self.engine.last_run_report, case, self.quality_runtime,
                                self.negative_source, self.engine.load_seconds)
                if self.profile['allow_cfg_delta']:
                    from training.studio.cfg_optional_runtime import annotate_optional_cfg
                    annotate_optional_cfg(self.engine.last_run_report, self.quality_runtime)
                else:
                    self.engine.last_run_report['cfg_execution'] = 'exact'
                atomic_json(target / 'generation.json', self.engine.last_run_report)

    def close(self):
        self.quality_runtime, self.negative_source = None, None
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.close()


def make_http_server(store, *, port=8894, index_path=None):
    index_path = Path(index_path) if index_path else Path(__file__).with_name('index.html')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, template, *args):
            print(f'[studio-http] {self.client_address[0]} {template % args}', flush=True)

        def _origin(self):
            port = self.server.server_address[1]
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            check(self.headers.get('Host') in hosts, 'untrusted Host', 403)
            origin = self.headers.get('Origin')
            check(origin is None or origin in {f'http://{host}' for host in hosts}, 'cross-origin requests denied', 403)
            check(self.headers.get('Sec-Fetch-Site') != 'cross-site', 'cross-site requests denied', 403)

        def _headers(self, status, content_type, length, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _json(self, status, value):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._headers(status, 'application/json; charset=utf-8', len(content))
            if self.command != 'HEAD':
                self.wfile.write(content)

        def _payload(self):
            check(self.headers.get('Transfer-Encoding') is None, 'chunked requests are not accepted', 400)
            check(self.headers.get('Content-Type', '').split(';')[0].strip() == 'application/json',
                  'application/json required', 415)
            try:
                length = int(self.headers.get('Content-Length', '-1'))
            except ValueError:
                raise APIError(400, 'invalid Content-Length')
            check(0 <= length <= MAX_JSON, 'JSON body exceeds limit or length is missing', 413)
            raw = self.rfile.read(length)
            check(len(raw) == length, 'incomplete request body')
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError):
                raise APIError(400, 'invalid JSON')
            check(isinstance(payload, dict), 'JSON object required')
            return payload

        def _file(self, path, content_type):
            check(path.is_file(), 'media not found', 404)
            check(path.resolve().is_relative_to(store.directory) or path.resolve() == index_path.resolve(),
                  'media path outside root', 403)
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            headers = {'Accept-Ranges': 'bytes'}
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                if match is None or not any(match.groups()) or size == 0:
                    self._headers(416, content_type, 0, {'Content-Range': f'bytes */{size}'})
                    return
                lower, upper = match.groups()
                if lower:
                    start = int(lower)
                    end = min(int(upper), size - 1) if upper else size - 1
                else:
                    count = int(upper)
                    start, end = max(0, size - count), size - 1
                if start >= size or start > end or (not lower and int(upper) == 0):
                    self._headers(416, content_type, 0, {'Content-Range': f'bytes */{size}'})
                    return
                status = 206
                headers['Content-Range'] = f'bytes {start}-{end}/{size}'
            self._headers(status, content_type, max(0, end - start + 1), headers)
            if self.command == 'HEAD':
                return
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = stream.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def _dispatch(self):
            self._origin()
            path = urlsplit(self.path).path
            if self.command in ('GET', 'HEAD'):
                if path == '/':
                    return self._file(index_path, 'text/html; charset=utf-8')
                if path == '/api/config':
                    archive = (dict(read_only=True, archive_label=store.archive_label,
                                    saved_sessions=store.saved_sessions, default_session_id=store.default_session_id)
                               if getattr(store, 'read_only', False) else {})
                    return self._json(200, dict(scenes=[dict(scene_id=store.scene['scene_id'], prompt=store.scene['prompt'],
                                                             image_url='/media/initial.png')],
                                                **store.profile, **archive, experimental_steps=[step for step in (20, 10)
                                                                                   if step in store.profile['allowed_steps']], realtime=False,
                                                remaining_service_seconds=max(0, int(store.deadline - time.monotonic()))))
                match = re.fullmatch(r'/api/sessions/([0-9a-f]{32})', path)
                if match:
                    return self._json(200, store.get(match[1]))
                if path == '/media/initial.png':
                    return self._file(store.directory / 'initial.png', 'image/png')
                match = re.fullmatch(r'/media/sessions/([0-9a-f]{32})/([0-9a-f]{32})/video\.mp4', path)
                if match:
                    return self._file(store.media(match[1], match[2]), 'video/mp4')
                match = re.fullmatch(r'/media/sessions/([0-9a-f]{32})/([0-9a-f]{32})/timeline\.mp4', path)
                if match:
                    return self._file(store.timeline(match[1], match[2]), 'video/mp4')
                raise APIError(404, 'route not found')
            if self.command == 'POST':
                check(not getattr(store, 'read_only', False),
                      '只读真实结果回放；未连接 GPU，不接受创建、生成或取消请求。', 405)
                payload = self._payload()
                if path == '/api/sessions':
                    return self._json(201, store.create(payload))
                match = re.fullmatch(r'/api/sessions/([0-9a-f]{32})/(segments|cancel)', path)
                if match:
                    if match[2] == 'segments':
                        return self._json(202, store.submit(match[1], payload))
                    check(not payload, 'cancel accepts an empty JSON object')
                    return self._json(200, store.cancel(match[1]))
                raise APIError(404, 'route not found')
            raise APIError(405, 'method not allowed')

        def _handle(self):
            try:
                self._dispatch()
            except APIError as error:
                self.close_connection = True
                self._json(error.status, dict(error=str(error)))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception as error:
                self.close_connection = True
                print(f'[studio-http-error] {type(error).__name__}: {error}', flush=True)
                self._json(500, dict(error='服务内部错误，详见保留日志；请刷新状态，勿重复提交。'))

        do_GET = do_HEAD = do_POST = _handle

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.timeout = .5
    return server


def serve(request, directory, lease):
    """Serve until the finite lease window ends, then release our engine."""
    from PIL import Image
    import numpy as np
    from training.studio.worker import read_source
    began = time.monotonic()
    maximum = request.get('max_seconds')
    check(type(maximum) is int and maximum >= 60, 'finite max_seconds is required')
    deadline = began + maximum - 45
    port = request.get('port', 8894)
    check(type(port) is int and 1024 <= port <= 65535, 'invalid operator port')
    minimum = request.get('minimum_task_seconds', 600)
    check(type(minimum) is int and minimum >= 60, 'minimum_task_seconds must be at least 60')
    profile = studio_profile(request)
    directory = Path(directory).resolve()
    source_record_path = Path(request['source_job']) / 'request.json'
    source_record = source_record_path.read_bytes()
    state, config, prompt, lineage, initial, _, scene = read_source(request)
    check(source_record_path.read_bytes() == source_record,
          'source request record changed while loading; cannot bind session provenance', 500)
    source_identity = source_identity_from_verified_record(request, source_record, lineage, scene)
    if profile['profile'] == 'low_resolution_preview':
        initial = np.asarray(Image.fromarray(initial).resize(
            (profile['width'], profile['height']), Image.Resampling.LANCZOS), dtype=np.uint8).copy()
    check(initial.shape == (profile['height'], profile['width'], 3), 'source image does not match studio profile', 500)
    # Reject incompatible persisted prefixes BEFORE overwriting their displayed
    # initial image or provenance record. No legacy migration is implied.
    store = StudioStore(directory, scene, deadline=deadline, minimum_task_seconds=minimum,
                        width=profile['width'], height=profile['height'],
                        quality_settings=profile['quality_settings'], negative_cache=profile['negative_cache'],
                        fastpath=request.get('fastpath', False), source_identity=source_identity,
                        allow_cfg_delta=profile['allow_cfg_delta'])
    Image.fromarray(initial).save(directory / 'initial.png')
    atomic_json(directory / 'inputs.json', dict(scene_id=scene['scene_id'], prompt=scene['prompt'],
        checkpoint=lineage['path'], checkpoint_sha256=lineage['sha256'], source_job=request['source_job'],
        source_revision=request.get('source_revision'), experiment_revision=request.get('experiment_revision'),
        source_identity=source_identity,
        **profile, cached_prompt_only=True, input_loading_seconds=time.monotonic() - began))
    executor = RealExecutor(dict(state=state, config=config, prompt=prompt, initial=initial,
                                source_identity=source_identity), request, lease)
    server = make_http_server(store, port=port)
    worker = GenerationWorker(store, executor)
    worker.start()
    atomic_json(directory / 'service.json', dict(status='serving', url=f'http://127.0.0.1:{port}/',
        started_unix=time.time(), maximum_service_seconds=maximum - 45, idle_unload_seconds=180,
        source_revision=request.get('source_revision'), realtime=False))
    lifecycle = 'bounded_window_ended'
    try:
        while time.monotonic() < deadline and not store.stopping.is_set():
            if (directory / 'STOP').is_file():
                lifecycle = 'operator_stop'
                break
            server.handle_request()
    finally:
        store.stop()
        server.server_close()
        if store.export_lock.acquire(timeout=6):
            store.export_lock.release()
        worker.join(timeout=38)  # Export cleanup + generator wait stay inside the 45s reserve.
        atomic_json(directory / 'service.json', dict(status='stopped' if not worker.is_alive() else 'shutdown_pending',
            url=f'http://127.0.0.1:{port}/', elapsed_seconds=time.monotonic() - began,
            error=worker.fatal_error, realtime=False))
    check(not worker.is_alive(), 'active generation did not finish cooperative shutdown; outer watchdog remains in force', 500)
    check(worker.fatal_error is None, worker.fatal_error or 'generation worker failed', 500)
    counts = {}
    for session in store.sessions.values():
        for segment in session['segments']:
            counts[segment['status']] = counts.get(segment['status'], 0) + 1
    return dict(status='completed', mode='serve', service_lifecycle=lifecycle,
                session_count=len(store.sessions), segment_status_counts=counts,
                elapsed_seconds=time.monotonic() - began, realtime=False, quality_evaluation='not_run')
