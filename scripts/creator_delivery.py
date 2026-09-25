"""Run one bounded Creator API case and retain actual responses on the data disk.

No model imports, fallback media, automatic request retries, automatic revisions, or
GPU commands are used here. The running service owns authorization and execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid


class DeadlineExceeded(TimeoutError):
    pass


class APIError(RuntimeError):
    def __init__(self, status, response):
        self.status, self.response = status, response
        super().__init__(f'HTTP {status}: {json.dumps(response, ensure_ascii=False)}')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def loopback_url(value):
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
        loopback = host == 'localhost' or (host is not None and ipaddress.ip_address(host).is_loopback)
    except ValueError:
        loopback = False
    if (not loopback or parsed.scheme != 'http' or parsed.username is not None
            or parsed.password is not None or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError('--url must be an HTTP loopback origin, e.g. http://127.0.0.1:8765')
    return f'{parsed.scheme}://{parsed.netloc}'


def bounded_seconds(value):
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('--max-seconds must be an integer in 1..3600') from error
    if not 1 <= seconds <= 3600:
        raise argparse.ArgumentTypeError('--max-seconds must be in 1..3600')
    return seconds


class Delivery:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.max_seconds
        self.output = args.output.resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.token = ''
        self.attempted_posts = set()
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.report = dict(schema='creator_delivery_v1', run_id=uuid.uuid4().hex,
                           source='actual_creator_http_api', case=args.case, url=args.url,
                           started_at=utc_now(), status='running', max_seconds=args.max_seconds,
                           poll_interval_seconds=30, automatic_revision=False,
                           visual_quality_verified=False, session_id=args.session_id,
                           version_id=args.version_id, job_id=None, requests=[], polls=[])
        # Reserve a new evidence path before any request; never overwrite a prior run.
        with self.output.open('x', encoding='utf-8') as stream:
            json.dump(self.report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')

    def save(self):
        self.report['elapsed_seconds'] = round(time.monotonic() - self.started, 3)
        fd, name = tempfile.mkstemp(prefix='.' + self.output.name + '-', suffix='.tmp', dir=self.output.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(self.report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            os.replace(name, self.output)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def log(self, event, **fields):
        print(json.dumps(dict(time=utc_now(), event=event, **fields), ensure_ascii=False, separators=(',', ':')), flush=True)

    def remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise DeadlineExceeded('client deadline reached; this does not cancel a server-side operation')
        return seconds

    def api(self, path, payload=None):
        remaining = self.remaining()
        method = 'POST' if payload is not None else 'GET'
        headers = {'Accept': 'application/json', 'Origin': self.args.url}
        data = None
        if payload is not None:
            if path in self.attempted_posts:
                raise RuntimeError(f'refusing to repeat a mutation: {path}')
            self.attempted_posts.add(path)
            headers.update({'Content-Type': 'application/json', 'X-InterActWorld-CSRF': self.token})
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        receipt = dict(method=method, path=path, started_at=utc_now(), state='sent_once' if data else 'reading')
        if payload is not None:
            receipt['payload'] = payload
        self.report['requests'].append(receipt)
        self.save()
        self.log('request', method=method, path=path)
        result = queue.Queue(maxsize=1)

        def read_response():
            try:
                request = Request(self.args.url + path, data=data, headers=headers, method=method)
                try:
                    with self.opener.open(request, timeout=remaining) as response:
                        status = response.status
                        raw = response.read(8 * 1024 * 1024 + 1)
                except HTTPError as error:
                    with error:
                        status = error.code
                        raw = error.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError('API response exceeded 8 MiB')
                try:
                    body = json.loads(raw.decode('utf-8'))
                except (UnicodeError, ValueError) as error:
                    raise ValueError(f'HTTP {status} did not return valid JSON: {raw[:1000]!r}') from error
                if not 200 <= status < 300:
                    raise APIError(status, body)
                if not isinstance(body, dict):
                    raise ValueError('API response must be a JSON object')
                result.put((body, status, None))
            except Exception as error:
                result.put((None, None, error))

        # The wall deadline also bounds stalled response reads. A timeout never
        # retries a POST; the daemon request may already have reached the server.
        threading.Thread(target=read_response, daemon=True).start()
        try:
            body, status, error = result.get(timeout=self.remaining())
            if error is not None:
                raise error
        except queue.Empty as error:
            receipt.update(state='client_timeout', finished_at=utc_now())
            raise DeadlineExceeded(f'client deadline while awaiting {path}; server outcome is unknown') from error
        except Exception as error:
            receipt.update(state='error', finished_at=utc_now(), error_type=type(error).__name__, error=str(error))
            if isinstance(error, APIError):
                receipt.update(http_status=error.status, response=error.response)
            self.save()
            raise
        receipt.update(state='responded', finished_at=utc_now(), http_status=status)
        self.save()
        self.log('response', method=method, path=path, http_status=status)
        return body

    def session(self, session_id):
        for value in self.api('/api/sessions').get('sessions', []):
            if value.get('session_id') == session_id:
                return value
        raise ValueError(f'session not found: {session_id}')

    @staticmethod
    def version(session, version_id):
        for value in session.get('versions', []):
            if value.get('version_id') == version_id:
                return value
        raise ValueError(f'version not found: {version_id}')

    def finish(self, status, **fields):
        self.report.update(status=status, finished_at=utc_now(), **fields)
        self.save()
        self.log('finished', status=status, session_id=self.report['session_id'],
                 version_id=self.report['version_id'], job_id=self.report['job_id'], output=str(self.output))

    def run(self):
        config = self.api('/api/config')
        self.token = config.get('csrf_token', '')
        if not isinstance(self.token, str) or not self.token:
            raise ValueError('configuration did not provide a CSRF token')
        self.report['config'] = {key: value for key, value in config.items() if key != 'csrf_token'}
        self.save()
        if self.args.case == 'inspect':
            before = self.session(self.args.session_id)
            version = self.version(before, self.args.version_id)
            self.report.update(session_before=before, job_id=version.get('job_id'))
            self.save()
            if version.get('job_status') != 'completed':
                raise ValueError('inspect requires the specified version to have a completed real job')
            inspected = self.api('/api/inspect', dict(session_id=self.args.session_id, version_id=self.args.version_id))
            self.report['session'] = inspected
            review = self.version(inspected, self.args.version_id).get('review')
            self.report['review'] = review
            if not review or review.get('source') != 'model_assessment':
                raise ValueError('inspect did not return a model_assessment; no visual completion is claimed')
            self.finish('inspection_returned')
            return 0

        if not config.get('generation_enabled'):
            raise ValueError('server generation is disabled; no plan or generation was submitted')
        if self.args.case == 'retry':
            before = self.session(self.args.session_id)
            failed = self.version(before, self.args.version_id)
            self.report.update(session_before=before, retry_of=self.args.version_id)
            self.save()
            if failed.get('job_status') != 'failed' or failed.get('plan', {}).get('status') != 'ready':
                raise ValueError('retry requires a failed job with an already validated ready plan')
            if failed.get('retry_count', 0) >= 2:
                raise ValueError('maximum two retries already reached')
            request_id = uuid.uuid4().hex
            generated = self.api('/api/retry', dict(session_id=self.args.session_id,
                                version_id=self.args.version_id, request_id=request_id))
            self.report['retry_response'] = generated
            self.save()
            matches = [v for v in generated.get('versions', []) if v.get('request_id') == request_id]
            if len(matches) != 1:
                raise ValueError('retry response did not identify exactly one new version for this request')
            version = matches[0]
            if (version.get('origin') != 'retry' or version.get('retry_of') != self.args.version_id
                    or version.get('plan') != failed.get('plan')
                    or generated.get('scene_id') != before.get('scene_id')
                    or generated.get('seed') != before.get('seed')):
                raise ValueError('retry response changed the frozen plan/scene/seed or omitted retry provenance')
            if not version.get('job_id') or version.get('job_id') == failed.get('job_id'):
                raise ValueError('retry response did not provide a new job')
            return self.poll_job(generated, version['version_id'])
        text = self.args.text or ('一直前进，后半段抬头' if self.args.case == 'initial' else '保留前进，只缩短抬头')
        if self.args.case == 'initial':
            scenes = config.get('scenes', [])
            if not scenes or not scenes[0].get('scene_id'):
                raise ValueError('configuration has no available scene')
            payload = dict(scene_id=scenes[0]['scene_id'], seed=42, text=text)
        else:
            before = self.session(self.args.session_id)
            self.report['session_before'] = before
            payload = dict(session_id=before['session_id'], scene_id=before['scene_id'], seed=before['seed'], text=text)
        payload['request_id'] = uuid.uuid4().hex
        planned = self.api('/api/plan', payload)
        matches = [v for v in planned.get('versions', []) if v.get('request_id') == payload['request_id']]
        if len(matches) != 1:
            self.report['plan_response'] = planned
            raise ValueError('plan response did not identify exactly one version for this new request')
        version = matches[0]
        self.report.update(session_id=planned['session_id'], version_id=version['version_id'],
                           plan_response=planned, session=planned)
        self.save()
        if version.get('plan', {}).get('status') != 'ready':
            self.finish('plan_not_ready', explanation=version.get('plan', {}).get('explanation'))
            return 1
        generated = self.api('/api/generate', dict(session_id=planned['session_id'], version_id=version['version_id'],
                                                    request_id=uuid.uuid4().hex))
        return self.poll_job(generated, version['version_id'])

    def poll_job(self, generated, version_id):
        session_id = generated['session_id']
        generated_version = self.version(generated, version_id)
        job_id = generated_version.get('job_id')
        self.report.update(generation_response=generated, session=generated, session_id=session_id,
                           version_id=version_id, job_id=job_id)
        self.save()
        if not job_id:
            raise ValueError('generation response did not provide a job id')
        self.log('submitted', session_id=session_id, version_id=version_id, job_id=job_id)
        while True:
            jobs = self.api('/api/jobs').get('jobs', [])
            job = next((item for item in jobs if item.get('job_id') == job_id), None)
            if job is None:
                raise ValueError(f'submitted job is absent from the live API: {job_id}')
            self.report['job'] = job
            self.report['polls'].append(dict(observed_at=utc_now(), status=job.get('status'), error=job.get('error')))
            self.save()
            self.log('job', job_id=job_id, status=job.get('status'), error=job.get('error'))
            if job.get('status') == 'completed':
                self.report['session'] = self.session(session_id)
                self.report['version'] = self.version(self.report['session'], version_id)
                self.finish('generation_completed')
                return 0
            if job.get('status') in ('failed', 'cancelled'):
                self.finish('job_failed', error=job.get('error') or job['status'])
                return 1
            if job.get('status') not in ('queued', 'running'):
                raise ValueError(f'unrecognized job status: {job.get("status")!r}')
            time.sleep(min(30, self.remaining()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, type=loopback_url, help='HTTP loopback origin of the real Creator service')
    parser.add_argument('--output', required=True, type=Path, help='new report.json path on the operator data disk')
    parser.add_argument('--case', required=True, choices=('initial', 'edit', 'inspect', 'retry'))
    parser.add_argument('--session-id')
    parser.add_argument('--version-id')
    parser.add_argument('--text', help='override initial/edit instruction, 1..2000 characters')
    parser.add_argument('--max-seconds', type=bounded_seconds, default=1800, help='total client deadline, 1..3600 (default 1800)')
    args = parser.parse_args(argv)
    if args.case in ('edit', 'inspect', 'retry') and not args.session_id:
        parser.error('--session-id is required for edit/inspect/retry')
    if args.case in ('inspect', 'retry') and not args.version_id:
        parser.error('--version-id is required for inspect/retry')
    if args.case == 'initial' and args.session_id:
        parser.error('initial creates a new session; --session-id is not accepted')
    if args.case not in ('inspect', 'retry') and args.version_id:
        parser.error('--version-id is only used for inspect/retry')
    if args.text is not None and (args.case in ('inspect', 'retry') or not 1 <= len(args.text.strip()) <= 2000):
        parser.error('--text is a nonempty initial/edit instruction of at most 2000 characters')
    delivery = None
    try:
        delivery = Delivery(args)
        return delivery.run()
    except KeyboardInterrupt:
        if delivery:
            delivery.finish('interrupted', error='client interrupted; server operation is not cancelled',
                            server_operation_may_continue=True)
        return 130
    except Exception as error:
        if delivery:
            timed_out = isinstance(error, (DeadlineExceeded, TimeoutError, socket.timeout)) or (
                isinstance(error, URLError) and isinstance(error.reason, (TimeoutError, socket.timeout)))
            delivery.finish('timed_out' if timed_out else 'error', error_type=type(error).__name__, error=str(error),
                            server_operation_may_continue=bool(delivery.attempted_posts))
        else:
            print(json.dumps(dict(event='error', error_type=type(error).__name__, error=str(error)), ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
