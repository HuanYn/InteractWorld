"""Bounded serial queue and strict loopback HTTP API; no public file server."""
from __future__ import annotations

import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import re
import secrets
import threading
import time
from urllib.parse import urlsplit
import uuid

from training.demo.backend import SubprocessBackend
from training.demo.contracts import Catalog, contained, load_initial, require, write_json
from training.demo import page


class QueueFullError(RuntimeError):
    pass


class DemoService:
    def __init__(self, deployment, *, executor=None, start_worker=True):
        self.deployment = deployment
        self.catalog = Catalog(deployment)
        self.executor = executor if executor is not None else SubprocessBackend(deployment)
        self.pending = queue.Queue(maxsize=deployment.max_pending)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.csrf_token = secrets.token_urlsafe(32)
        self.jobs = {}
        deployment.jobs_root.mkdir(parents=True, exist_ok=True)
        self._process_lock = None
        if os.name == 'posix':
            import fcntl
            self._process_lock = (deployment.jobs_root / '.demo-service.lock').open('a')
            try:
                fcntl.flock(self._process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._process_lock.close()
                raise RuntimeError('another demo service owns this job queue')
        for directory in deployment.jobs_root.iterdir():
            if not directory.is_dir() or not re.fullmatch('[0-9a-f]{32}', directory.name):
                continue
            path = directory / 'status.json'
            if path.is_file():
                try:
                    status = json.loads(path.read_text(encoding='utf-8'))
                    require(status['job_id'] == directory.name, 'saved job identity mismatch')
                    if status['status'] in ('queued', 'running'):
                        status.update(status='failed', error='服务重启中断了任务；没有自动重放或复用旧视频。')
                        write_json(path, status)
                    self.jobs[directory.name] = status
                except (ValueError, KeyError):
                    continue
        self.thread = threading.Thread(target=self._work, name='interactworld-demo-worker', daemon=True)
        if start_worker:
            self.thread.start()

    def submit(self, payload):
        if not self.deployment.guard_command:
            raise ValueError('未配置 GPU 授权与预算门禁，生成尚未启用。')
        with self.lock:
            if self.stop_event.is_set():
                raise ValueError('service is stopping')
            if self.pending.full():
                raise QueueFullError('队列已满，请等待当前任务结束后再提交。')
            job_id = uuid.uuid4().hex
            directory = contained(self.deployment.jobs_root / job_id, self.deployment.project_root)
            request = self.catalog.freeze(payload, directory, job_id)
            status = dict(job_id=job_id, scene_id=request['scene_id'], seed=request['seed'],
                          status='queued', created_unix=time.time(), error=None)
            self.jobs[job_id] = status
            write_json(directory / 'status.json', status)
            self.pending.put_nowait(job_id)
            return dict(status)

    def _update(self, job_id, **fields):
        with self.lock:
            self.jobs[job_id].update(fields)
            write_json(self.deployment.jobs_root / job_id / 'status.json', self.jobs[job_id])

    def _work(self):
        while not self.stop_event.is_set():
            try:
                job_id = self.pending.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                self._update(job_id, status='running', started_unix=time.time())
                self.executor(self.deployment.jobs_root / job_id, self.stop_event)
                self._update(job_id, status='completed', finished_unix=time.time())
            except Exception as error:
                self._update(job_id, status='failed', error=str(error)[:1000],
                             failure_type=type(error).__name__, finished_unix=time.time())
            finally:
                self.pending.task_done()

    def list_jobs(self):
        with self.lock:
            return [dict(value) for value in sorted(self.jobs.values(), key=lambda j: j['created_unix'], reverse=True)[:50]]

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=40)
        if self._process_lock is not None:
            self._process_lock.close()


def make_server(service, port=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'InterActWorldDemo/1'

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass

        def security_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

        def send_data(self, data, *, status=200, content_type='application/json; charset=utf-8', headers=None):
            self.send_response(status)
            self.security_headers()
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def json(self, value, status=200):
            self.send_data(json.dumps(value, ensure_ascii=False).encode(), status=status)

        def same_host(self):
            expected = f'127.0.0.1:{self.server.server_address[1]}'
            require(self.headers.get('Host') == expected, 'unexpected Host; DNS rebinding blocked')
            return 'http://' + expected

        def do_POST(self):
            try:
                origin = self.same_host()
                require(self.headers.get('Origin') == origin, 'cross-origin requests are forbidden')
                require(secrets.compare_digest(self.headers.get('X-InterActWorld-CSRF', ''), service.csrf_token), 'invalid CSRF token')
                require(self.headers.get_content_type() == 'application/json', 'JSON content type required')
                require(self.path == '/api/jobs', 'unknown API route')
                require(self.headers.get('Transfer-Encoding') is None, 'chunked requests are not accepted')
                size = int(self.headers.get('Content-Length', '0'))
                require(0 < size <= 65536, 'request body exceeds64KB or is empty')
                payload = json.loads(self.rfile.read(size))
                self.json(service.submit(payload), status=202)
            except QueueFullError as error:
                self.json({'error': str(error)}, status=429)
            except (ValueError, TypeError, KeyError) as error:
                self.json({'error': str(error)}, status=400)

        def do_OPTIONS(self):
            self.json({'error': 'cross-origin API use is disabled'}, status=405)

        def do_HEAD(self):
            self.do_GET()

        def send_file(self, path, *, download=False):
            require(path.is_file(), 'artifact not found')
            size = path.stat().st_size
            start, end = 0, size - 1
            status = 200
            range_header = self.headers.get('Range')
            if range_header:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', range_header)
                require(match is not None, 'unsupported byte range')
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
                require(0 <= start <= end < size, 'byte range outside file')
                status = 206
            self.send_response(status)
            self.security_headers()
            self.send_header('Content-Type', 'video/mp4' if path.suffix == '.mp4' else 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            if download:
                self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
            self.end_headers()
            if self.command != 'HEAD':
                with path.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        def do_GET(self):
            try:
                self.same_host()
                url = urlsplit(self.path)
                if url.path in ('/', '/app.js', '/style.css'):
                    content, kind = {'/': (page.HTML, 'text/html'), '/app.js': (page.JS, 'text/javascript'),
                                     '/style.css': (page.CSS, 'text/css')}[url.path]
                    self.send_data(content.encode(), content_type=kind + '; charset=utf-8')
                elif url.path == '/api/config':
                    self.json(dict(scenes=service.catalog.public(), csrf_token=service.csrf_token,
                                   generation_enabled=bool(service.deployment.guard_command), action_keys=['W','A','S','D','I','J','K','L'],
                                   prompt_editing=False, fps=16, future_frames=240, max_pending=service.deployment.max_pending,
                                   supported_stages=['action_teacher_lora_v1','causal_teacher_forcing_v1','longforcing_lite_v1']))
                elif url.path == '/api/jobs':
                    self.json({'jobs': service.list_jobs()})
                elif match := re.fullmatch(r'/api/jobs/([0-9a-f]{32})', url.path):
                    require(match[1] in service.jobs, 'unknown job')
                    with service.lock:
                        self.json(dict(service.jobs[match[1]]))
                elif match := re.fullmatch(r'/api/scenes/([A-Za-z0-9_-]{1,100})/initial.png', url.path):
                    require(match[1] in service.catalog.scenes, 'unknown scene')
                    from PIL import Image
                    output = io.BytesIO()
                    Image.fromarray(load_initial(service.catalog.scenes[match[1]]['initial_frame_path'])).save(output, format='PNG')
                    self.send_data(output.getvalue(), content_type='image/png')
                elif match := re.fullmatch(r'/api/jobs/([0-9a-f]{32})/files/(raw\.mp4|inputs\.mp4|actions\.npy|request\.json|receipt\.json)', url.path):
                    job_id, name = match.groups()
                    require(job_id in service.jobs and service.jobs[job_id]['status'] == 'completed', 'job is not completed')
                    directory = contained(service.deployment.jobs_root / job_id, service.deployment.project_root)
                    self.send_file(contained(directory / name, directory), download=url.query == 'download=1')
                else:
                    self.json({'error': 'not found'}, status=404)
            except (ValueError, KeyError, OSError) as error:
                self.json({'error': str(error)}, status=400)

    server = ThreadingHTTPServer((service.deployment.host, service.deployment.port if port is None else port), Handler)
    server.daemon_threads = True
    return server
