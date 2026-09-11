"""Immutable browser-job inputs and operator-only deployment configuration."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

KEYS = ('W', 'A', 'S', 'D', 'I', 'J', 'K', 'L')
ACTION_STAGE = 'action_teacher_lora_v1'
SUPPORTED_STAGES = (ACTION_STAGE, 'causal_teacher_forcing_v1', 'longforcing_lite_v1')
ADAPTER = 'training.eval.wan_causal_adapter:create_wan_causal_adapter'
ACTION_ADAPTER = 'training.demo.action_backend:generate_action_video'
ACTION_METHOD = 'action_teacher_chunked_ar15s_ui_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def contained(path, root):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(Path(root).resolve()) and resolved != Path(root).resolve(),
            'path must stay inside the configured project directory')
    return resolved


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@dataclass(frozen=True)
class Deployment:
    rollout_config: Path
    project_root: Path
    jobs_root: Path
    guard_command: tuple[str, ...] = ()
    host: str = '127.0.0.1'
    port: int = 8765
    scene_count: int = 1
    max_pending: int = 2
    max_job_seconds: int = 900
    python_executable: str = sys.executable

    def validate(self):
        require(self.host == '127.0.0.1', 'demo service is loopback-only; use an SSH tunnel')
        require(type(self.port) is int and 0 <= self.port <= 65535, 'invalid port')
        require(type(self.scene_count) is int and 1 <= self.scene_count <= 3, 'scene_count must be1..3')
        require(type(self.max_pending) is int and 1 <= self.max_pending <= 8, 'max_pending must be1..8')
        require(type(self.max_job_seconds) is int and 30 <= self.max_job_seconds <= 3600, 'invalid job cap')
        require(self.project_root.is_dir(), 'project root does not exist')
        contained(self.jobs_root, self.project_root)
        require(Path(self.python_executable).is_absolute(), 'python_executable must be absolute')
        require(not self.guard_command or (Path(self.guard_command[0]).is_absolute()
                and all(isinstance(part, str) and part and '\x00' not in part for part in self.guard_command)),
                'guard_command must be an operator-provided absolute executable argv list')

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        allowed = set(cls.__dataclass_fields__)
        require(isinstance(raw, dict) and set(raw).issubset(allowed), 'unknown deployment fields')
        for key in ('rollout_config', 'project_root', 'jobs_root'):
            raw[key] = Path(raw[key]).resolve()
        command = raw.get('guard_command', [])
        require(isinstance(command, list), 'guard_command must be argv, never a shell string')
        raw['guard_command'] = tuple(command)
        result = cls(**raw)
        result.validate()
        return result


def action_array(segments):
    require(isinstance(segments, list) and 1 <= len(segments) <= 240, 'action timeline needs1..240 segments')
    rows = []
    for segment in segments:
        require(isinstance(segment, dict) and set(segment) == {'frames', 'keys'}, 'invalid action segment fields')
        frames, keys = segment['frames'], segment['keys']
        require(type(frames) is int and 1 <= frames <= 240, 'segment frames must be integer1..240')
        require(isinstance(keys, list) and all(type(key) is str for key in keys)
                and len(set(keys)) == len(keys) and set(keys).issubset(KEYS), 'invalid or duplicate action keys')
        require(not any({a, b}.issubset(keys) for a, b in [('W', 'S'), ('A', 'D'), ('I', 'K'), ('J', 'L')]),
                'opposing simultaneous keys are not supported')
        rows.extend([[float(key in keys) for key in KEYS]] * frames)
        require(len(rows) <= 240, 'timeline exceeds240 frames /15 seconds')
    require(len(rows) == 240, 'timeline must cover exactly240 frames at16 fps')
    return np.asarray(rows, dtype=np.float32)


def load_initial(path):
    if Path(path).suffix.lower() == '.npy':
        image = np.load(path, allow_pickle=False)
    else:
        from PIL import Image
        with Image.open(path) as source:
            image = np.asarray(source.convert('RGB'))
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3,
            'initial condition must be actual uint8 RGB')
    return image


class Catalog:
    def __init__(self, deployment):
        deployment.validate()
        self.deployment = deployment
        self.raw = yaml.safe_load(deployment.rollout_config.read_text(encoding='utf-8'))
        require(isinstance(self.raw, dict), 'invalid rollout YAML')
        lineage = self.raw.get('lineage', {})
        require(lineage.get('expected_stage') in SUPPORTED_STAGES,
                'unsupported self-trained checkpoint stage')
        is_action = lineage['expected_stage'] == ACTION_STAGE
        require(self.raw.get('adapter_factory') == (ACTION_ADAPTER if is_action else ADAPTER),
                'checkpoint stage does not match its concrete self-trained adapter')
        require(re.fullmatch('[0-9a-fA-F]{64}', str(lineage.get('checkpoint_sha256', ''))),
                'pin the exact self-trained checkpoint SHA before serving')
        require(Path(lineage['checkpoint_path']).name != 'best.pt', 'serve an immutable step checkpoint, not mutable best.pt')
        geometry = self.raw.get('geometry', {})
        fixed = (dict(width=832, height=480, fps=16, total_rgb_frames=241, num_chunks=5,
                      chunk_rgb_frames=49, chunk_future_frames=48) if is_action else
                 dict(fps=16, total_rgb_frames=241, num_chunks=20, latent_frames_per_chunk=3, rgb_frames_per_latent=4))
        require(all(geometry.get(key) == value for key, value in fixed.items()), 'unsupported timing contract')
        if is_action:
            require(self.raw.get('method') == ACTION_METHOD, 'Action UI method must be explicit')
            require(self.raw.get('sampler') == dict(solver='flow_euler', steps=40, shift=5.0, cfg='none'),
                    'Action UI uses the verified40-step flow Euler solver only')
        scenes = self.raw.get('scenes', [])
        require(1 <= len(scenes) <= 3 and deployment.scene_count <= len(scenes), 'invalid scene count')
        ids = [scene.get('scene_id') for scene in scenes]
        require(all(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value) for value in ids)
                and len(set(ids)) == len(ids), 'unsafe or duplicate scene ID')
        # V1 deliberately requires the exact precomputed static-text binding.
        # Full artifact/checkpoint validation is repeated by the actual worker.
        artifacts = lineage.get('artifact_paths', {})
        require('prompt_cache_receipt' in artifacts and 'prompt_cache' in artifacts,
                'first demo requires a bound static prompt cache; prompt editing is disabled')
        prompt_receipt = json.loads(Path(artifacts['prompt_cache_receipt']).read_text(encoding='utf-8'))
        require(prompt_receipt.get('prompt_policy') == 'scene_static_only_v1', 'unsupported prompt policy')
        self.scenes = {}
        for original in scenes[:deployment.scene_count]:
            scene = copy.deepcopy(original)
            binding = prompt_receipt.get('episodes', {}).get(scene.get('source_episode_id'))
            require(binding and binding.get('split') == 'dev' and binding.get('prompt') == scene.get('prompt'),
                    'scene prompt/source does not match the actual static embedding receipt')
            require(type(scene.get('seed')) is int and 0 <= scene['seed'] <= 2**32 - 1, 'invalid preset seed')
            action_array(scene['action_segments'])
            initial = load_initial(scene['initial_frame_path'])
            require(initial.shape == (geometry.get('height'), geometry.get('width'), 3), 'initial geometry mismatch')
            scene['initial_sha256'] = sha256(scene['initial_frame_path'])
            self.scenes[scene['scene_id']] = scene
        self.config_sha256 = sha256(deployment.rollout_config)

    def public(self):
        return [dict(scene_id=s['scene_id'], prompt=s['prompt'], seed=s['seed'],
                     source_episode_id=s['source_episode_id'], initial_sha256=s['initial_sha256'],
                     method=self.raw.get('method', 'causal_rollout15s'),
                     action_segments=s['action_segments'], initial_url=f"/api/scenes/{s['scene_id']}/initial.png")
                for s in self.scenes.values()]

    def freeze(self, payload, directory, job_id):
        require(isinstance(payload, dict) and set(payload) == {'scene_id', 'seed', 'action_segments'},
                'submit only scene_id, seed and action_segments; text/model/GPU fields are operator-only')
        scene = self.scenes.get(payload['scene_id'])
        require(scene is not None, 'unknown scene')
        require(type(payload['seed']) is int and 0 <= payload['seed'] <= 2**32 - 1, 'seed must be uint32')
        actions = action_array(payload['action_segments'])
        require(sha256(self.deployment.rollout_config) == self.config_sha256, 'operator rollout config changed; restart service')
        require(sha256(scene['initial_frame_path']) == scene['initial_sha256'], 'preset initial frame changed')
        initial = load_initial(scene['initial_frame_path'])
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / 'initial.npy', initial, allow_pickle=False)
        np.save(directory / 'actions.npy', actions, allow_pickle=False)
        selected = {key: copy.deepcopy(value) for key, value in scene.items() if key != 'initial_sha256'}
        selected.update(seed=payload['seed'], action_segments=copy.deepcopy(payload['action_segments']),
                        initial_frame_path=str(directory / 'initial.npy'))
        frozen = copy.deepcopy(self.raw)
        frozen.update(run_id=job_id, scenes=[selected], output_root=str(directory))
        config_path = directory / 'rollout.yaml'
        config_path.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding='utf-8')
        request = dict(schema_version=1, job_id=job_id, scene_id=selected['scene_id'],
                       source_episode_id=selected['source_episode_id'], prompt=selected['prompt'],
                       seed=selected['seed'], action_keys=list(KEYS), action_shape=[240, 8], action_dtype='float32',
                       actions_sha256=sha256(directory / 'actions.npy'), initial_sha256=sha256(directory / 'initial.npy'),
                       source_initial_sha256=scene['initial_sha256'], source_rollout_sha256=self.config_sha256,
                       config_sha256=sha256(config_path), checkpoint_sha256=frozen['lineage']['checkpoint_sha256'],
                       checkpoint_stage=frozen['lineage']['expected_stage'],
                       method=frozen.get('method', 'causal_rollout15s'),
                       action_segments=payload['action_segments'], fps=16, future_frames=240,
                       prompt_editing=False, ground_truth_future_used=False)
        write_json(directory / 'request.json', request)
        return request
