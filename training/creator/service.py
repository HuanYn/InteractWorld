"""Versioned creation sessions over the unchanged asynchronous video queue."""
from __future__ import annotations

import copy
from contextlib import contextmanager
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
from training.creator.planner import plan_request, expand_segments, KEYS


class CreatorService:
    def __init__(self, demo, *, provider=None, sessions_root=None, visual_revision_enabled=False,
                 serialize_model_and_video=False):
        self.demo = demo
        self.provider = provider
        require(type(visual_revision_enabled) is bool, 'visual_revision_enabled must be an operator boolean')
        self.visual_revision_enabled = visual_revision_enabled
        require(type(serialize_model_and_video) is bool, 'serialize_model_and_video must be an operator boolean')
        self.serialize_model_and_video = serialize_model_and_video
        self._model_active = False
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

    @contextmanager
    def _model_slot(self):
        """Fail fast for a single-card deployment; GPU authority stays in its guard."""
        if not self.serialize_model_and_video:
            yield
            return
        with self.lock:
            if self._model_active or any(job.get('status') in ('queued', 'running')
                                         for job in self.demo.list_jobs()):
                raise QueueFullError('单卡串行模式：模型或视频任务正在运行，请等待结束后再提交。')
            self._model_active = True
        try:
            yield
        finally:
            with self.lock:
                self._model_active = False

    def _submit_video(self, payload):
        with self.lock:
            if self.serialize_model_and_video and self._model_active:
                raise QueueFullError('单卡串行模式：模型正在工作，尚未创建视频任务，请稍后再提交。')
            return self.demo.submit(payload)

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
        require(set(payload) <= {'session_id', 'scene_id', 'seed', 'text', 'request_id', 'planner',
                                'base_version_id'}, 'unexpected plan fields')
        requested_base = payload.get('base_version_id')
        if 'base_version_id' in payload:
            require(isinstance(requested_base, str) and requested_base.strip(),
                    'base_version_id must be a nonempty version ID')
            require(payload.get('session_id'), 'base_version_id requires session_id')
        planner = payload.get('planner', 'local_model' if self.provider else 'rule_fallback')
        require(planner in ('local_model', 'rule_fallback'), 'unknown planner mode')
        require(planner != 'local_model' or self.provider is not None, 'local model is not configured; explicitly choose rule_fallback')
        provider = self.provider if planner == 'local_model' else None
        text = payload.get('text', '')
        rid = payload.get('request_id')
        require(isinstance(text, str) and 0 < len(text.strip()) <= 2000, 'instruction needs1..2000 characters')
        require(isinstance(rid, str) and re.fullmatch(r'[A-Za-z0-9_-]{8,100}', rid), 'request_id required')
        with self.lock:
            for existing in self.sessions.values():
                for version in existing['versions']:
                    if version.get('request_id') == rid:
                        require(version.get('origin', 'user') == 'user',
                                'request_id reused for a different operation')
                        previous_mode = version.get('planner_kind', 'rule_fallback' if version.get('provider') == 'rule_fallback' else 'local_model')
                        require(previous_mode == planner, 'request_id reused with different planner mode')
                        require(version['text'] == text and (not payload.get('session_id') or payload['session_id'] == existing['session_id']), 'request_id reused with different input')
                        require(payload.get('scene_id', existing['scene_id']) == existing['scene_id'] and payload.get('seed', existing['seed']) == existing['seed'], 'request_id condition mismatch')
                        # Bind retries to the original selection, not today's
                        # latest ready plan. Records predating explicit bases
                        # were necessarily implicit and retain that behavior.
                        require(requested_base == version.get('requested_base_version_id'),
                                'request_id reused with different base_version_id')
                        result = self.snapshot(existing)
                        result['planned_version_id'] = version['version_id']
                        return result
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
            if requested_base is not None:
                previous_version = self._version(session, requested_base)
                require(previous_version['plan']['status'] == 'ready',
                        'base_version_id must reference a ready plan')
            else:
                previous_version = next((v for v in reversed(session['versions']) if v['plan']['status'] == 'ready'), None)
            previous = copy.deepcopy(previous_version['plan']) if previous_version else None
            actual_base = previous_version['version_id'] if previous_version else None
            head = session['versions'][-1]['version_id'] if session['versions'] else None
            self.operations[rid] = session['session_id']
        try:
            proposal = None
            if provider:
                with self._model_slot():
                    proposal = provider.plan(text, previous)
            plan = plan_request(text, previous=previous, proposal=proposal)
            with self.lock:
                require((session['versions'][-1]['version_id'] if session['versions'] else None) == head, 'plan changed while model was working; submit your edit again')
                vid = uuid.uuid4().hex
                session['versions'].append(dict(version_id=vid, parent_version=actual_base,
                    base_version_id=actual_base, requested_base_version_id=requested_base,
                    text=text, plan=plan, job_id=None, review=None, created_at=time.time(), request_id=rid,
                    root_request=vid, automatic_revisions=0, origin='user',
                    planner_kind=planner,
                    provider=getattr(provider, 'name', 'local_model_command') if provider else 'rule_fallback'))
                self._save(session)
                result = self.snapshot(session)
                result['planned_version_id'] = vid
                return result
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
            job = self._submit_video(dict(scene_id=session['scene_id'], seed=session['seed'], action_segments=version['plan']['action_segments']))
            version['job_id'] = job['job_id']
            self._save(session)
            return self.snapshot(session)

    def _completed(self, version):
        job = self.demo.jobs.get(version.get('job_id'), {})
        require(job.get('status') == 'completed', 'a completed real generation is required')
        return self.demo.deployment.jobs_root / version['job_id']

    def _reference_version(self, session, version):
        """Find the completed parent plan, including its successful retry.

        A retry is the same plan, not a distinct visual-edit baseline. Never
        silently use another scene/session or an arbitrary earlier candidate.
        """
        logical = version
        seen = set()
        while logical.get('retry_of'):
            require(logical['version_id'] not in seen, 'cyclic retry lineage')
            seen.add(logical['version_id'])
            logical = self._version(session, logical['retry_of'])
        if not logical.get('parent_version'):
            return None
        parent = self._version(session, logical['parent_version'])
        candidates = [v for v in session['versions']
                      if v['version_id'] != version['version_id']
                      and v['root_request'] == parent['root_request']
                      and v['plan']['action_segments'] == parent['plan']['action_segments']
                      and self.demo.jobs.get(v.get('job_id'), {}).get('status') == 'completed']
        return max(candidates, key=lambda v: v['created_at']) if candidates else None

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
            job = self._submit_video(dict(scene_id=session['scene_id'], seed=session['seed'], action_segments=previous['plan']['action_segments']))
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
            criteria = payload.get('criteria', {})
            require(isinstance(criteria, dict) and set(criteria) <= {
                'movement_response', 'camera_response', 'temporal_stability'}, 'invalid human review criteria')
            require(all(value in ('satisfied', 'unsatisfied', 'uncertain') for value in criteria.values()),
                    'invalid human criterion verdict')
            previous = version.get('review')
            if previous:
                version.setdefault('review_history', []).append(copy.deepcopy(previous))
            assessment = (previous if previous and previous.get('source') != 'human'
                          else previous.get('previous_assessment') if previous else None)
            version['review'] = dict(verdict=payload['verdict'], evidence=evidence, source='human', decision='ask_user',
                                     criteria=copy.deepcopy(criteria), created_at=time.time(),
                                     previous_assessment=copy.deepcopy(assessment))
            self._save(session)
            return self.snapshot(session)

    def inspect(self, payload):
        require(self.provider is not None, 'visual model not configured; use clearly labelled human review')
        with self.lock:
            session = self._session(payload['session_id'])
            version = self._version(session, payload['version_id'])
            require(not version.get('review') or version['review'].get('source') != 'human', 'human review already exists; automatic review cannot overwrite it')
            directory = self._completed(version)
            reference = self._reference_version(session, version)
            # Keep the actual user's constraint even if a model-written goal
            # accidentally omitted "continuous" or a relative duration edit.
            goals = list(dict.fromkeys([*version['plan']['goals'], '用户当前请求：' + version['text']]))
            op = 'inspect:' + version['version_id']
            require(op not in self.operations, 'inspection already running')
            self.operations[op] = True
        try:
            with self._model_slot():
                if reference is None:
                    result = self.provider.inspect(str(directory / 'raw.mp4'), goals)
                else:
                    result = self.provider.inspect(str(directory / 'raw.mp4'), goals,
                        reference_video_path=str(self._completed(reference) / 'raw.mp4'))
            require(isinstance(result, dict) and result.get('verdict') in ('satisfied', 'unsatisfied', 'uncertain'), 'invalid visual assessment')
            require(result.get('decision') in ('accept', 'revise', 'ask_user', 'stop'), 'invalid visual decision')
            original_judgment = result.get('original_model_judgment')
            original_judgment = original_judgment if isinstance(original_judgment, dict) else result
            result['model_decision'] = original_judgment['decision']
            result['model_revision_text'] = original_judgment.get('revision_text', '')
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
                          calibrated=False, created_at=time.time(), automatic_acceptance=False,
                          reference_available=reference is not None,
                          reference_used=result.get('reference_used') is True,
                          reference_version_id=reference['version_id'] if reference else None,
                          reference_job_id=reference['job_id'] if reference else None)
            with self.lock:
                require(not version.get('review') or version['review'].get('source') != 'human', 'human review arrived; model assessment discarded')
                if version.get('review'):
                    version.setdefault('review_history', []).append(copy.deepcopy(version['review']))
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
            reference = self._reference_version(session, version)
            user_text = version['text']
            head = session['versions'][-1]['version_id']
        try:
            with self._model_slot():
                proposal = self.provider.plan(text, previous)
            plan = plan_request(text, previous=previous, proposal=proposal)
            require(plan['status'] == 'ready', 'visual revision did not produce an executable plan; ask the user')
            before, after = expand_segments(previous['action_segments']), expand_segments(plan['action_segments'])
            require(before != after, 'visual revision made no input change; no new version was created')
            scope = previous.get('edit_scope', 'all')
            protected = set(KEYS[:4] if scope == 'camera' else KEYS[4:] if scope == 'movement' else ())
            require(all(set(a) & protected == set(b) & protected for a, b in zip(before, after)),
                    'visual revision changed controls protected by the user edit scope')
            # A feedback edit must still meet the user's original edit request.
            # A shortened action must not be lengthened back to its parent span.
            user_check = plan_request(user_text, previous=reference['plan'] if reference else None,
                proposal={**plan, 'edit_scope': previous.get('edit_scope', 'all')})
            require(user_check['status'] == 'ready', 'visual revision contradicts the original user instruction')
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
                        rule_planner_available=True,
                        observer_kind='uncalibrated_local_vlm' if creator.provider else 'human_only', max_auto_revisions=1,
                        visual_revision_enabled=creator.visual_revision_enabled,
                        serialize_model_and_video=creator.serialize_model_and_video,
                        fps=16, future_frames=240, prompt_editing=False, realtime=False))
                else:
                    super().do_GET()
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.json({'error': str(error)}, status=400)

        def do_POST(self):
            routes = {'/api/plan': creator.plan, '/api/generate': creator.generate, '/api/review': creator.review,
                      '/api/inspect': creator.inspect, '/api/revise': creator.revise, '/api/accept': creator.accept,
                      '/api/retry': creator.retry}
            if creator.serialize_model_and_video:
                # The inherited demo submission endpoint shares this same gate.
                routes['/api/jobs'] = creator._submit_video
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
                self.json(routes[self.path](payload), status=202 if self.path == '/api/jobs' else 200)
            except QueueFullError as error:
                self.json({'error': str(error)}, status=429)
            except (ValueError, TypeError, KeyError) as error:
                self.json({'error': str(error)}, status=400)
            except Exception as error:
                print(f'Creator operation failed: {type(error).__name__}: {error}', flush=True)
                self.json({'error': '模型或服务执行失败，未推进任务成功状态；请查看本次运行日志。'}, status=503)

    server.RequestHandlerClass = Handler
    return server
