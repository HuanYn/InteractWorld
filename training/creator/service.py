"""Versioned creation sessions over the unchanged asynchronous video queue."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import secrets
import threading
import time
import uuid
from urllib.parse import urlsplit

from training.demo.contracts import require, write_json
from training.demo.service import make_server as make_demo_server, QueueFullError
from training.creator.planner import plan_request


class CreatorService:
    def __init__(self, demo, *, provider=None, sessions_root=None, visual_revision_enabled=False):
        self.demo = demo
        self.provider = provider
        require(type(visual_revision_enabled) is bool, 'visual_revision_enabled must be an operator boolean')
        self.visual_revision_enabled = visual_revision_enabled
        self.root = Path(sessions_root or demo.deployment.project_root / 'sessions')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sessions = {}
        self.operations = {}
        for path in self.root.glob('*.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            require(record['session_id'] == path.stem, 'session identity mismatch')
            self.sessions[path.stem] = record

    def _save(self, session):
        write_json(self.root / (session['session_id'] + '.json'), session)

    def _session(self, sid):
        require(isinstance(sid, str) and sid in self.sessions, 'unknown session')
        return self.sessions[sid]

    def _version(self, session, vid):
        for version in session['versions']:
            if version['version_id'] == vid:
                return version
        raise ValueError('unknown plan version')

    def snapshot(self, session):
        value = copy.deepcopy(session)
        jobs = {j['job_id']: j for j in self.demo.list_jobs()}
        for version in value['versions']:
            review = version.get('review')
            if isinstance(review, dict) and review.get('source') == 'model_assessment':
                review.setdefault('model_decision', review.get('decision'))
                review.setdefault('model_revision_text', review.get('revision_text', ''))
                if not self.visual_revision_enabled:
                    # Apply current operator policy to this deep-copy only.
                    # Persisted evidence and human reviews remain untouched.
                    review.update(decision='ask_user', visual_revision_enabled=False,
                        downgrade_reason='视觉模型尚未校准，操作方未启用视觉反馈修订；原始观察与建议仅供核对，请人工审核。')
            if version.get('job_id'):
                job = jobs.get(version['job_id'], self.demo.jobs.get(version['job_id'], {}))
                version['job_status'] = job.get('status', 'unavailable')
                version['job_error'] = job.get('error', '')
                if job.get('started_unix') and job.get('finished_unix'):
                    version['elapsed_seconds'] = job['finished_unix'] - job['started_unix']
        return value

    def list_sessions(self):
        with self.lock:
            return [self.snapshot(s) for s in sorted(self.sessions.values(), key=lambda s: s['created_at'], reverse=True)]

    def plan(self, payload):
        require(isinstance(payload, dict), 'request must be an object')
        require(set(payload) <= {'session_id', 'scene_id', 'seed', 'text', 'request_id'}, 'unexpected plan fields')
        text = payload.get('text', '')
        rid = payload.get('request_id')
        require(isinstance(text, str) and 0 < len(text.strip()) <= 2000, 'instruction needs1..2000 characters')
        require(isinstance(rid, str) and re.fullmatch(r'[A-Za-z0-9_-]{8,100}', rid), 'request_id required')
        with self.lock:
            for existing in self.sessions.values():
                for version in existing['versions']:
                    if version.get('request_id') == rid:
                        require(version['text'] == text and (not payload.get('session_id') or payload['session_id'] == existing['session_id']), 'request_id reused with different input')
                        require(payload.get('scene_id', existing['scene_id']) == existing['scene_id'] and payload.get('seed', existing['seed']) == existing['seed'], 'request_id condition mismatch')
                        return self.snapshot(existing)
            require(rid not in self.operations, 'this request is already being planned')
            if payload.get('session_id'):
                session = self._session(payload['session_id'])
                require(payload.get('scene_id', session['scene_id']) == session['scene_id'] and payload.get('seed', session['seed']) == session['seed'], 'scene/seed are immutable in this session')
            else:
                require(payload.get('scene_id') in self.demo.catalog.scenes, 'unknown scene')
                seed = payload.get('seed', 42)
                require(type(seed) is int and 0 <= seed <= 2**32 - 1, 'invalid seed')
                session = dict(session_id=uuid.uuid4().hex, scene_id=payload['scene_id'], seed=seed,
                               created_at=time.time(), accepted_version=None, versions=[])
                self.sessions[session['session_id']] = session
                self._save(session)
            previous_version = next((v for v in reversed(session['versions']) if v['plan']['status'] == 'ready'), None)
            previous = copy.deepcopy(previous_version['plan']) if previous_version else None
            head = session['versions'][-1]['version_id'] if session['versions'] else None
            self.operations[rid] = session['session_id']
        try:
            proposal = self.provider.plan(text, previous) if self.provider else None
            plan = plan_request(text, previous=previous, proposal=proposal)
            with self.lock:
                require((session['versions'][-1]['version_id'] if session['versions'] else None) == head, 'plan changed while model was working; submit your edit again')
                vid = uuid.uuid4().hex
                session['versions'].append(dict(version_id=vid, parent_version=previous_version['version_id'] if previous_version else None,
                    text=text, plan=plan, job_id=None, review=None, created_at=time.time(), request_id=rid,
                    root_request=vid, automatic_revisions=0, origin='user',
                    provider=getattr(self.provider, 'name', 'local_model_command') if self.provider else 'rule_fallback'))
                self._save(session)
                return self.snapshot(session)
        finally:
            with self.lock:
                self.operations.pop(rid, None)

    def generate(self, payload):
        with self.lock:
            session = self._session(payload['session_id'])
            version = self._version(session, payload['version_id'])
            require(version['plan']['status'] == 'ready', 'clarify unsupported/ambiguous requests before generation')
            if version.get('job_id'):
                return self.snapshot(session)
            job = self.demo.submit(dict(scene_id=session['scene_id'], seed=session['seed'], action_segments=version['plan']['action_segments']))
            version['job_id'] = job['job_id']
            self._save(session)
            return self.snapshot(session)

    def _completed(self, version):
        job = self.demo.jobs.get(version.get('job_id'), {})
        require(job.get('status') == 'completed', 'a completed real generation is required')
        return self.demo.deployment.jobs_root / version['job_id']

    def retry(self, payload):
        """Explicit bounded retry of a failed job, without another model plan."""
        rid = payload.get('request_id')
        require(isinstance(rid, str) and re.fullmatch(r'[A-Za-z0-9_-]{8,100}', rid), 'request_id required')
        with self.lock:
            session = self._session(payload['session_id'])
            previous = self._version(session, payload['version_id'])
            existing = next((v for v in session['versions'] if v.get('request_id') == rid), None)
            if existing:
                require(existing.get('retry_of') == previous['version_id'], 'request_id reused for a different operation')
                return self.snapshot(session)
            require(self.demo.jobs.get(previous.get('job_id'), {}).get('status') == 'failed', 'only failed jobs may be retried')
            # A new request ID must not fork another GPU job from the same
            # failed parent. Its one retry child remains the canonical result.
            if any(v.get('retry_of') == previous['version_id'] for v in session['versions']):
                return self.snapshot(session)
            require(previous.get('retry_count', 0) < 2, 'two retries exhausted; resolve the failure before a new request')
            require(previous['plan']['status'] == 'ready', 'cannot retry a non-executable plan')
            job = self.demo.submit(dict(scene_id=session['scene_id'], seed=session['seed'], action_segments=previous['plan']['action_segments']))
            version = copy.deepcopy(previous)
            version.update(version_id=uuid.uuid4().hex, parent_version=previous['version_id'],
                retry_of=previous['version_id'], retry_count=previous.get('retry_count', 0) + 1,
                request_id=rid, job_id=job['job_id'], review=None, origin='retry', created_at=time.time())
            session['versions'].append(version)
            self._save(session)
            return self.snapshot(session)

    def review(self, payload):
        with self.lock:
            session = self._session(payload['session_id'])
            version = self._version(session, payload['version_id'])
            self._completed(version)
            require(payload.get('verdict') in ('satisfied', 'unsatisfied', 'uncertain'), 'invalid verdict')
            evidence = payload.get('evidence', '')
            require(isinstance(evidence, str) and 0 < len(evidence.strip()) <= 2000, 'please give an observation, not just a score')
            previous = version.get('review')
            version['review'] = dict(verdict=payload['verdict'], evidence=evidence, source='human', decision='ask_user',
                                     created_at=time.time(), previous_assessment=previous if previous and previous.get('source') != 'human' else None)
            self._save(session)
            return self.snapshot(session)

    def inspect(self, payload):
        require(self.provider is not None, 'visual model not configured; use clearly labelled human review')
        with self.lock:
            session = self._session(payload['session_id'])
            version = self._version(session, payload['version_id'])
            require(not version.get('review') or version['review'].get('source') != 'human', 'human review already exists; automatic review cannot overwrite it')
            directory = self._completed(version)
            op = 'inspect:' + version['version_id']
            require(op not in self.operations, 'inspection already running')
            self.operations[op] = True
        try:
            result = self.provider.inspect(str(directory / 'raw.mp4'), version['plan']['goals'])
            require(isinstance(result, dict) and result.get('verdict') in ('satisfied', 'unsatisfied', 'uncertain'), 'invalid visual assessment')
            require(result.get('decision') in ('accept', 'revise', 'ask_user', 'stop'), 'invalid visual decision')
            result['model_decision'] = result['decision']
            result['model_revision_text'] = result.get('revision_text', '')
            require(isinstance(result.get('evidence'), list) and
                    (result['evidence'] or result['verdict'] == 'uncertain'), 'definite visual assessment must cite temporal observations')
            for evidence in result['evidence']:
                require(isinstance(evidence, dict) and type(evidence.get('time_seconds')) in (int, float)
                        and 0 <= evidence['time_seconds'] <= 15.1 and isinstance(evidence.get('observation'), str), 'invalid temporal evidence')
            if result['verdict'] == 'uncertain':
                result['decision'] = 'ask_user'
            require((result['decision'] == 'accept') == (result['verdict'] == 'satisfied'), 'accept requires a satisfied verdict')
            if result['decision'] == 'revise':
                require(result['verdict'] == 'unsatisfied' and isinstance(result.get('revision_text'), str) and result['revision_text'].strip(), 'revision requires an observed failure and specific edit')
            result.update(source='model_assessment', provider=getattr(self.provider, 'name', 'local_model_command'),
                          calibrated=False, created_at=time.time(), automatic_acceptance=False)
            with self.lock:
                require(not version.get('review') or version['review'].get('source') != 'human', 'human review arrived; model assessment discarded')
                version['review'] = result
                self._save(session)
                return self.snapshot(session)
        finally:
            with self.lock:
                self.operations.pop(op, None)

    def revise(self, payload):
        require(self.provider is not None, 'automatic revision needs a model provider')
        require(self.visual_revision_enabled, 'visual revision is disabled; uncalibrated observations require human review')
        with self.lock:
            session = self._session(payload['session_id'])
            version = self._version(session, payload['version_id'])
            self._completed(version)
            root = version['root_request']
            existing = next((v for v in session['versions'] if v['root_request'] == root and v['origin'] == 'visual_feedback'), None)
            if existing:
                return self.snapshot(session)
            require(session['versions'][-1]['version_id'] == version['version_id'],
                    'a newer user plan exists; visual revision cannot replace it')
            require(version['automatic_revisions'] == 0, 'maximum one feedback revision reached')
            review = version.get('review') or {}
            require(review.get('source') == 'model_assessment' and review.get('decision') == 'revise'
                    and review.get('verdict') == 'unsatisfied', 'no supported visual revision decision')
            op = 'revise:' + root
            require(op not in self.operations, 'revision is already being planned')
            self.operations[op] = True
            text = review['revision_text']
            previous = copy.deepcopy(version['plan'])
            head = session['versions'][-1]['version_id']
        try:
            proposal = self.provider.plan(text, previous)
            plan = plan_request(text, previous=previous, proposal=proposal)
            with self.lock:
                require(session['versions'][-1]['version_id'] == head, 'a newer user edit arrived; automatic revision discarded')
                require(version.get('review') == review, 'review changed while planning; automatic revision discarded')
                session['versions'].append(dict(version_id=uuid.uuid4().hex, parent_version=version['version_id'],
                    text=text, plan=plan, job_id=None, review=None, created_at=time.time(), request_id=payload.get('request_id', uuid.uuid4().hex),
                    root_request=root, automatic_revisions=1, origin='visual_feedback'))
                self._save(session)
                return self.snapshot(session)
        finally:
            with self.lock:
                self.operations.pop(op, None)

    def accept(self, payload):
        with self.lock:
            session = self._session(payload['session_id'])
            version = self._version(session, payload['version_id'])
            self._completed(version)
            session['accepted_version'] = version['version_id']
            self._save(session)
            return self.snapshot(session)


def make_server(creator, port=None):
    server = make_demo_server(creator.demo, port)
    Base = server.RequestHandlerClass

    class Handler(Base):
        def do_GET(self):
            from training.creator import page
            route = urlsplit(self.path).path
            try:
                self.same_host()
                if route in ('/', '/app.js', '/style.css'):
                    content, kind = {'/': (page.HTML, 'text/html'), '/app.js': (page.JS, 'text/javascript'), '/style.css': (page.CSS, 'text/css')}[route]
                    self.send_data(content.encode(), content_type=kind + '; charset=utf-8')
                elif route == '/api/sessions':
                    self.json(dict(sessions=creator.list_sessions()))
                elif route == '/api/config':
                    self.json(dict(csrf_token=creator.demo.csrf_token, scenes=creator.demo.catalog.public(),
                        generation_enabled=bool(creator.demo.deployment.guard_command), planner_kind='local_model' if creator.provider else 'rule_fallback',
                        observer_kind='uncalibrated_local_vlm' if creator.provider else 'human_only', max_auto_revisions=1,
                        visual_revision_enabled=creator.visual_revision_enabled,
                        fps=16, future_frames=240, prompt_editing=False, realtime=False))
                else:
                    super().do_GET()
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.json({'error': str(error)}, status=400)

        def do_POST(self):
            routes = {'/api/plan': creator.plan, '/api/generate': creator.generate, '/api/review': creator.review,
                      '/api/inspect': creator.inspect, '/api/revise': creator.revise, '/api/accept': creator.accept,
                      '/api/retry': creator.retry}
            if self.path not in routes:
                return super().do_POST()
            try:
                origin = self.same_host()
                require(self.headers.get('Origin') == origin, 'cross-origin requests are forbidden')
                require(secrets.compare_digest(self.headers.get('X-InterActWorld-CSRF', ''), creator.demo.csrf_token), 'invalid CSRF token')
                require(self.headers.get_content_type() == 'application/json', 'JSON content type required')
                require(self.headers.get('Transfer-Encoding') is None, 'chunked requests are not accepted')
                size = int(self.headers.get('Content-Length', '0'))
                require(0 < size <= 65536, 'request body exceeds64KB or is empty')
                payload = json.loads(self.rfile.read(size))
                self.json(routes[self.path](payload))
            except QueueFullError as error:
                self.json({'error': str(error)}, status=429)
            except (ValueError, TypeError, KeyError) as error:
                self.json({'error': str(error)}, status=400)
            except Exception as error:
                print(f'Creator operation failed: {type(error).__name__}: {error}', flush=True)
                self.json({'error': '模型或服务执行失败，未推进任务成功状态；请查看本次运行日志。'}, status=503)

    server.RequestHandlerClass = Handler
    return server
