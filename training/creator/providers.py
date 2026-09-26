"""JSON command bridge for operator-configured, externally guarded models.

This module never imports a model, grants GPU access, or builds a shell command.
The configured executable must own authorization, budget and GPU leases. Request,
response and diagnostic files are retained below ``runtime_root`` for auditing.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile


CAPABILITIES = {
    "fps": 16,
    "future_frames": 240,
    "movement": {"W": "forward", "A": "left", "S": "backward", "D": "right"},
    "camera": {"I": "look up", "J": "look left", "K": "look down", "L": "look right"},
    "timeline_edit": {"op": "replace_intervals", "time_unit": "seconds", "interval": "half_open",
                      "frame_period_seconds": 0.0625, "duration_seconds": 15,
                      "preservation": "all unedited control keys at every frame",
                      "base_version": "previous_plan is chosen by the user/application, never by the model"},
    "unsupported": ["jump", "interact", "attack", "change scene or character", "guaranteed object interaction"],
    "limits": "A proposed action input is not evidence that the generated video will realize it.",
}


class ProviderError(RuntimeError):
    """The real provider failed; no synthetic plan or assessment is substituted."""


def validated_planning_trace(value):
    """Copy bounded worker observations, never use them as execution authority.

    A malformed trace can be omitted by the service without changing the
    independent plan validation. No arbitrary model metadata is persisted.
    """
    if not isinstance(value, dict) or set(value) != {
        'schema_version', 'max_revisions', 'model_load_count', 'outcome',
        'attempts', 'revision_count',
    }:
        raise ValueError('invalid planning trace fields')
    for key, minimum, maximum in [('schema_version', 1, 1), ('max_revisions', 1, 1),
                                  ('model_load_count', 0, 1), ('revision_count', 0, 1)]:
        if type(value[key]) is not int or not minimum <= value[key] <= maximum:
            raise ValueError('invalid planning trace integer')
    if value['outcome'] not in ('first_pass', 'repaired', 'failed', 'needs_clarification', 'unsupported'):
        raise ValueError('invalid planning trace outcome')
    attempts = value['attempts']
    if not isinstance(attempts, list) or not 0 <= len(attempts) <= 2:
        raise ValueError('invalid planning trace attempts')
    if value['revision_count'] != max(0, len(attempts) - 1):
        raise ValueError('planning trace revision count differs from attempts')
    if value['model_load_count'] != int(bool(attempts)):
        raise ValueError('planning trace model load count differs from attempts')
    clean_attempts = []
    for number, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or set(attempt) != {'attempt', 'status', 'feedback', 'elapsed_seconds'}:
            raise ValueError('invalid planning attempt fields')
        if type(attempt['attempt']) is not int or attempt['attempt'] != number:
            raise ValueError('invalid planning attempt number')
        if attempt['status'] not in ('ready', 'clarify', 'unsupported'):
            raise ValueError('invalid planning attempt status')
        elapsed = attempt['elapsed_seconds']
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or not 0 <= elapsed <= 3600:
            raise ValueError('invalid planning attempt elapsed time')
        feedback = attempt['feedback']
        if not isinstance(feedback, dict) or not {'code', 'repairable', 'message'} <= set(feedback) \
                or set(feedback) - {'code', 'repairable', 'message', 'missing_keys', 'extra_keys', 'protected_keys'}:
            raise ValueError('invalid planning feedback fields')
        if not isinstance(feedback['code'], str) or not 1 <= len(feedback['code']) <= 80 \
                or not all(char.isascii() and (char.isalnum() or char == '_') for char in feedback['code']):
            raise ValueError('invalid planning feedback code')
        if type(feedback['repairable']) is not bool or not isinstance(feedback['message'], str) \
                or len(feedback['message']) > 2000 or '\x00' in feedback['message']:
            raise ValueError('invalid planning feedback message')
        for key in ('missing_keys', 'extra_keys', 'protected_keys'):
            if key in feedback and (not isinstance(feedback[key], list) or len(feedback[key]) > 8
                    or any(not isinstance(control, str) or control not in tuple('WASDIJKL')
                           for control in feedback[key]) or len(set(feedback[key])) != len(feedback[key])):
                raise ValueError('invalid planning feedback controls')
        clean_attempts.append({**attempt, 'feedback': {**feedback,
            **{key: list(feedback[key]) for key in ('missing_keys', 'extra_keys', 'protected_keys')
               if key in feedback}}})
    if value['outcome'] == 'repaired' and len(attempts) != 2:
        raise ValueError('repaired outcome requires two attempts')
    if value['outcome'] == 'first_pass' and len(attempts) != 1:
        raise ValueError('first pass outcome requires one attempt')
    return {**value, 'attempts': clean_attempts}


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _terminate_owned_group(process):
    """Terminate only this invocation's new session/process tree."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # The leader can exit while descendants still hold the GPU or log files.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    elif process.poll() is None:
        # CREATE_NEW_PROCESS_GROUP isolates this command. taskkill /T operates
        # on this exact owned PID and descendants, never an executable name.
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=10, check=False,
        )
        if result.returncode and process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def _tail(path, limit=8000):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace").strip()


class CommandProvider:
    """Invoke an argv template containing literal {request} and {output}.

    For example, the operator may supply a guard executable followed by its
    worker arguments. Both placeholders are replaced as single argv values (or
    within a value); no shell parsing, formatting evaluation or fallback occurs.
    """

    def __init__(self, argv: list[str], runtime_root: Path, timeout_seconds: int = 180):
        if not isinstance(argv, list) or not argv or not all(
            isinstance(value, str) and value and "\x00" not in value for value in argv
        ):
            raise ValueError("provider command must be a non-empty argv list")
        if not all(any(marker in value for value in argv) for marker in ("{request}", "{output}")):
            raise ValueError("provider argv must contain {request} and {output} placeholders")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
            raise ValueError("provider timeout must be an integer in 1..3600 seconds")
        self.argv = tuple(argv)
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def _call(self, request):
        directory = Path(tempfile.mkdtemp(prefix=request["kind"] + "-", dir=self.runtime_root))
        request_path, output_path = directory / "request.json", directory / "output.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        command = [value.replace("{request}", str(request_path)).replace("{output}", str(output_path))
                   for value in self.argv]
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        options = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        }
        process = None
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                                           stderr=stderr, shell=False, **options)
                try:
                    returncode = process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired as error:
                    _terminate_owned_group(process)
                    raise ProviderError(f"model command timed out after {self.timeout_seconds}s; artifacts: {directory}") from error
                finally:
                    if os.name == "posix":
                        _terminate_owned_group(process)
            if returncode != 0:
                detail = _tail(stderr_path) or _tail(stdout_path)
                raise ProviderError(f"model command exited {returncode}; artifacts: {directory}\n{detail}")
            if not output_path.is_file() or output_path.stat().st_size > 262144:
                raise ProviderError(f"model command returned no response or an oversized response; artifacts: {directory}")
            response = json.loads(output_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if not isinstance(response, dict) or "error" in response:
                raise ProviderError(f"model command returned an invalid/error response; artifacts: {directory}: {response!r}")
            return response
        except (OSError, ValueError) as error:
            raise ProviderError(f"model provider failed; artifacts: {directory}: {error}") from error
        finally:
            if process is not None and process.poll() is None:
                _terminate_owned_group(process)

    def plan(self, text, previous=None):
        # Legacy visual-revision consumers require proposal fields only.
        return self.plan_with_trace(text, previous)['proposal']

    def plan_with_trace(self, text, previous=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("planning text must be non-empty")
        if previous is not None and not isinstance(previous, dict):
            raise ValueError("previous plan must be an object or null")
        response = self._call({"kind": "plan", "text": text, "previous_plan": previous,
                               "capabilities": CAPABILITIES})
        return {'proposal': {key: value for key, value in response.items() if key != 'planning_trace'},
                'planning_trace': response.get('planning_trace')}

    def inspect(self, video_path, goals, *, reference_video_path=None):
        path = Path(video_path).resolve()
        if not path.is_file():
            raise ValueError("inspection requires an existing raw video")
        if not isinstance(goals, list) or not goals or not all(isinstance(goal, str) and goal.strip() for goal in goals):
            raise ValueError("inspection goals must be a non-empty list of strings")
        request = {"kind": "inspect", "video_path": str(path), "goals": goals}
        if reference_video_path is not None:
            reference = Path(reference_video_path).resolve()
            if not reference.is_file() or reference == path:
                raise ValueError("comparison needs a different existing raw reference video")
            request['reference_video_path'] = str(reference)
        return self._call(request)
