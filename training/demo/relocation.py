"""Explicit read-only relocation of immutable Action training artifacts.

Stored YAML/checkpoint/receipt bytes remain untouched. Only path comparison and
file opening use the operator's prefix map; hashes still cover the original
bytes. This mirrors train_action_teacher._manifest_hashes and its receipt
validators, including schema, prompt policy, source episodes and text hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
import warnings

from training.demo.contracts import require, sha256


def _absolute(value):
    require(isinstance(value, (str, Path)) and str(value), 'relocation paths must be nonempty strings')
    text = str(value)
    pure = PureWindowsPath(text) if PureWindowsPath(text).drive else PurePosixPath(text)
    require(pure.is_absolute() and '..' not in pure.parts, 'relocation paths must be absolute without traversal')
    return pure


def validate_relocations(value):
    """An explicit map, never heuristic basename matching or string replacement."""
    require(isinstance(value, dict), 'path_relocations must be an operator prefix mapping')
    for source, destination in value.items():
        source_path, destination_path = _absolute(source), _absolute(destination)
        require(len(source_path.parts) > 1 and len(destination_path.parts) > 1,
                'path_relocations must not map a filesystem root')
    return dict(value)


def relocate_path(value, relocations):
    source_path = _absolute(value)
    # One pass, longest component-prefix wins. /old/root never matches
    # /old/root-other, and the destination is never recursively remapped.
    for source, destination in sorted(relocations.items(), key=lambda item: len(_absolute(item[0]).parts), reverse=True):
        prefix = _absolute(source)
        if type(source_path) is type(prefix) and source_path.is_relative_to(prefix):
            return Path(str(_absolute(destination).joinpath(*source_path.relative_to(prefix).parts)))
    return Path(str(source_path))


def artifact_paths(config, config_path, relocations):
    manifest = _absolute(config.data.manifest_path)
    index = manifest.parent.parent / 'features' / f'{manifest.stem}.features.jsonl'
    result = dict(dataset_manifest=relocate_path(config.data.manifest_path, relocations),
                  feature_index=relocate_path(str(index), relocations),
                  feature_receipt=relocate_path(str(index.with_suffix('.jsonl.receipt.json')), relocations),
                  training_config=Path(config_path))
    if config.data.prompt_cache_path is not None:
        prompt = _absolute(config.data.prompt_cache_path)
        result.update(prompt_cache=relocate_path(str(prompt), relocations),
                      prompt_cache_receipt=relocate_path(str(prompt.with_suffix('.pt.receipt.json')), relocations))
    return result


class ArtifactHashes:
    """Optional operator-owned stat-keyed SHA cache; never supplies expected pins.

    Without a cache path this retains only per-validation deduplication. On-disk
    entries are reusable only for the same resolved path and complete stat
    identity. Invalid cache content is discarded, so original files are hashed
    again. Callers must still compare every returned SHA with their pinned SHA.
    """
    def __init__(self, verification_cache=None, *, protected_paths=()):
        self.values = {}
        self.cache_path = None
        if verification_cache is not None:
            require(isinstance(verification_cache, (str, Path)) and str(verification_cache)
                    and Path(verification_cache).is_absolute(), 'verification_cache must be an absolute operator path')
            self.cache_path = Path(verification_cache).resolve()
            require(self.cache_path not in {Path(path).resolve() for path in protected_paths},
                    'verification_cache must not overwrite an Action artifact')
            self.values = self._read_cache()

    def _read_cache(self):
        try:
            if not self.cache_path.is_file() or self.cache_path.stat().st_size > 4 * 1024 * 1024:
                return {}
            raw = json.loads(self.cache_path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict) or set(raw) != {'version', 'artifacts'} or type(raw['version']) is not int or raw['version'] != 1:
                return {}
            entries = raw['artifacts']
            if not isinstance(entries, dict) or len(entries) > 4096:
                return {}
            result = {}
            for path, entry in entries.items():
                if (not isinstance(path, str) or not Path(path).is_absolute()
                        or not isinstance(entry, dict) or set(entry) != {'identity', 'sha256'}):
                    return {}
                identity, digest = entry['identity'], entry['sha256']
                if (not isinstance(identity, list) or len(identity) != 5
                        or not all(type(value) is int for value in identity)
                        or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None):
                    return {}
                result[Path(path)] = tuple(identity), digest
            return result
        except (OSError, ValueError, TypeError):
            return {}

    def _save(self):
        if self.cache_path is None:
            return
        temporary = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            value = dict(version=1, artifacts={str(path): dict(identity=list(identity), sha256=digest)
                                              for path, (identity, digest) in self.values.items()})
            descriptor, temporary = tempfile.mkstemp(prefix='.' + self.cache_path.name + '-', dir=self.cache_path.parent)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cache_path)
        except OSError as error:
            # Cache availability is not permission to skip content verification.
            warnings.warn(f'Action verification cache unavailable; verified hashes remain uncached: {error}', RuntimeWarning)
            self.cache_path = None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def __call__(self, value):
        path = Path(value).resolve()
        require(path != self.cache_path, 'verification_cache must not overwrite an Action artifact')
        before = self._identity(path)
        cached = self.values.get(path)
        if cached and cached[0] == before:
            require(self._identity(path) == before, f'Action artifact changed during cache lookup: {path}')
            return cached[1]
        digest = sha256(path)
        require(self._identity(path) == before, f'Action artifact changed during hashing: {path}')
        self.values[path] = before, digest
        self._save()
        return digest


def relocated_manifest_hashes(config, config_path, relocations, *, verifier=None, verification_cache=None):
    """Equivalent immutable bindings with explicitly relocated receipt paths.

    Returns the original six-key checkpoint hash schema for static-prompt data,
    or the original four-key schema for legacy data. Does not load any tensors,
    feature shards, videos, or GPU models.
    """
    relocations = validate_relocations(relocations)
    paths = artifact_paths(config, config_path, relocations)
    require(verifier is None or verification_cache is None, 'provide one artifact verifier/cache source')
    digest = verifier if verifier is not None else ArtifactHashes(verification_cache, protected_paths=paths.values())
    hashes = {key: digest(path) for key, path in paths.items()}
    expected_manifest = config.data.manifest_sha256
    require(expected_manifest is None or hashes['dataset_manifest'] == expected_manifest,
            'dataset manifest hash mismatch')
    feature = json.loads(paths['feature_receipt'].read_text(encoding='utf-8'))
    require(isinstance(feature, dict) and feature.get('schema_version') == 1,
            'unsupported feature-cache receipt schema')
    for field, key in (('index', 'feature_index'), ('manifest', 'dataset_manifest')):
        require(relocate_path(feature.get(field), relocations).resolve() == paths[key].resolve(),
                f'feature-cache receipt {field} path mismatch')
    require(feature.get('index_sha256') == hashes['feature_index'], 'feature-cache index hash mismatch')
    require(feature.get('manifest_sha256') == hashes['dataset_manifest'], 'feature-cache manifest hash mismatch')
    if config.data.num_frames == 97:
        from training.data.action_dataset import validate_window97_contract
        validate_window97_contract(feature)
    if config.data.prompt_cache_path is None:
        return hashes
    receipt = json.loads(paths['prompt_cache_receipt'].read_text(encoding='utf-8'))
    require(isinstance(receipt, dict) and receipt.get('schema_version') == 1,
            'unsupported scene-static prompt receipt schema')
    require(receipt.get('kind') == 'scene_static_prompt_cache'
            and receipt.get('prompt_policy') == 'scene_static_only_v1', 'scene-static prompt kind/policy mismatch')
    require(relocate_path(receipt.get('cache_path'), relocations).resolve() == paths['prompt_cache'].resolve(),
            'scene-static prompt cache path mismatch')
    for field, key in (('manifest_sha256', 'dataset_manifest'), ('feature_index_sha256', 'feature_index'),
                       ('feature_receipt_sha256', 'feature_receipt'), ('cache_sha256', 'prompt_cache')):
        require(receipt.get(field) == hashes[key], f'scene-static prompt {field} mismatch')
    require(isinstance(receipt.get('encoder'), dict) and receipt['encoder'],
            'scene-static prompt receipt has no encoder provenance')
    source_episodes = {}
    with paths['feature_index'].open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f'feature index line {number} must be an object')
            identity, split = row.get('episode_id'), row.get('split')
            require(isinstance(identity, str) and identity and isinstance(split, str) and split
                    and identity not in source_episodes, 'feature index contains invalid or duplicate episode IDs')
            source_episodes[identity] = split
    episodes = receipt.get('episodes')
    require(isinstance(episodes, dict) and episodes, 'scene-static prompt receipt has no episode bindings')
    for identity, binding in episodes.items():
        require(identity in source_episodes and isinstance(binding, dict),
                f'scene-static prompt has unknown/unbound episode: {identity}')
        require(binding.get('split') == source_episodes[identity], f'scene-static prompt split mismatch: {identity}')
        prompt = binding.get('prompt')
        require(isinstance(prompt, str) and prompt.strip(), f'scene-static prompt text is missing: {identity}')
        require(binding.get('prompt_sha256') == hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                f'scene-static prompt text hash mismatch: {identity}')
    return hashes
