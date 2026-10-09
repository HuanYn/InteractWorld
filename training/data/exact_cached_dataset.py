"""Opt-in exact-window reader; candidate metadata is NOT visual qualification.

Import with InterActWorld-public on PYTHONPATH. No raw media, model, encoder,
network, cache writes, or launch path. Only the old noise/flow math is inherited;
the old episode-random window sampler and whole-shard hashing are never called.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
from operator import index as integer_index
from pathlib import Path

import torch

from training.data.action_dataset import (
    FeatureCacheError, FROZEN_DEMO_EPISODE, SCENE_STATIC_PROMPT_POLICY,
    WINDOW97_CACHE_KIND, WINDOW97_ENCODING_POLICY, _read_jsonl, _seed64,
    action_teacher_shapes, sha256_file, validate_feature_cache_binding,
    validate_window97_contract,
)
from training.data.action_resampled import ResampledActionDataset

NAMESPACE = "expanded_exact_cached_windows_v1"
ORIGINS = frozenset(("native97_pool256", "balanced_train6"))
KEYS = "WASDIJKL"
IDENTITY_FIELDS = ("cache_origin", "episode_id", "window_index", "source_start")
QUALIFICATION_EPISODES = frozenset((
    "063168aac541196a82d74a399c633dc6", "00afe1074cb9437f5fc87870dec48263",
    "06737e69c9479b803087f53be0485f12", FROZEN_DEMO_EPISODE,
))


def require(condition, message):
    if not condition:
        raise FeatureCacheError(message)


def window_key(row):
    """JSON subset rows need these four fields, not episode-only identities."""
    key = tuple(row[name] for name in IDENTITY_FIELDS)
    require(key[0] in ORIGINS and isinstance(key[1], str) and bool(key[1])
            and all(type(v) is int and v >= 0 for v in key[2:]), "invalid exact window identity")
    return key


def actions_from_runs(row):
    """Reconstruct the recorded canonical matrix; NEVER synthesize from target."""
    recorded = row["actions"]
    require(recorded["frames"] == 96, "exact 96 action rows required")
    values, position = [], 0
    for run in recorded["runs"]:
        start, end = run["start_future_index"], run["end_future_index_exclusive"]
        keys = "" if run["keys"] == "NONE" else run["keys"]
        require(type(start) is int and type(end) is int and start == position
                and start < end <= 96 and run["frames"] == end - start
                and isinstance(keys, str) and len(set(keys)) == len(keys)
                and set(keys) <= set(KEYS), "action runs must cover 0..96 exactly, without gaps/overlap")
        values.extend([[int(k in keys) for k in KEYS] for _ in range(end - start)])
        position = end
    require(position == 96, "incomplete action runs")
    require({k: sum(v[i] for v in values) for i, k in enumerate(KEYS)} == recorded["key_frames"],
            "action runs/counts differ")
    if "future_actions_96x8" in row:
        require(values == row["future_actions_96x8"], "action runs/matrix differ")
    return values


class ExactWindowSelection:
    """Explicit subset + canonical J/L schedule, usable without tensor reads.

    No default 'all' subset: the caller must name each intended window. This is
    selection, not an approval mechanism; visual review and job gates stay outside.
    """
    def __init__(self, manifest_path, *, subset):
        raw = Path(manifest_path).read_bytes()
        manifest = json.loads(raw)
        require(manifest.get("schema") == "expanded_cached_metadata_selection_v1", "wrong candidate schema")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        candidates = {window_key(r): r for r in manifest["selected"]}
        require(len(candidates) == len(manifest["selected"]), "duplicate candidate window")
        require(subset is not None, "explicit subset required; no implicit all-candidate training")
        wanted = [window_key(r) for r in subset]
        require(wanted and len(wanted) == len(set(wanted)), "subset must be nonempty and unique")
        require(set(wanted) <= candidates.keys(), "subset contains an unselected/unknown window")
        excluded = QUALIFICATION_EPISODES | set(manifest.get("excluded_unique_episode_ids", []))
        chosen = [copy.deepcopy(candidates[k]) for k in sorted(wanted)]
        physical = {(r["episode_id"], r["source_start"]) for r in chosen}
        require(len(physical) == len(chosen), "same physical window selected more than once")
        for row in chosen:
            require(row["episode_id"] not in excluded and row["target"] in ("J", "L"),
                    "qualification/demo episode or invalid direction in subset")
            require(row.get("annotation_cache_actions_equal_recorded") is True, "missing recorded action equality")
            row["future_actions_96x8"] = actions_from_runs(row)
            require(set(row["shard_stat"]) == {"bytes", "mtime_ns"}
                    and all(type(v) is int and v > 0 for v in row["shard_stat"].values()), "invalid shard stat")
        sides = {d: [r for r in chosen if r["target"] == d] for d in ("J", "L")}
        require(len(sides["J"]) == len(sides["L"]) > 0, "subset must explicitly retain equal nonzero J/L counts")
        self.rows = tuple(r for pair in zip(sides["J"], sides["L"]) for r in pair)
        self.contract = dict(namespace=NAMESPACE, candidate_manifest_sha256=self.manifest_sha256,
            selected_windows=[dict(zip(IDENTITY_FIELDS, window_key(r)), target=r["target"]) for r in self.rows],
            target_schedule=[r["target"] for r in self.rows],
            selection_sha256=hashlib.sha256(json.dumps(chosen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            window_policy="exact_selected_windows_JL_round_robin_no_episode_resampling",
            visual_qualification_claimed=False, training_authorized=False)


def _inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "cache path escapes bound index root")
    return path


def _unique_rows(path):
    rows = _read_jsonl(path)
    mapped = {r["episode_id"]: r for r in rows}
    require(len(mapped) == len(rows), "duplicate episode in metadata")
    return mapped


def _read_bound_source(info):
    """Hash only small metadata once; never scan a feature/weight payload."""
    index, manifest = Path(info["path"]).resolve(), Path(info["manifest"]).resolve()
    receipt_path = index.with_suffix(index.suffix + ".receipt.json")
    require(Path(info["receipt"]).resolve() == receipt_path, "receipt path differs")
    require(sha256_file(receipt_path) == info["receipt_sha256"], "receipt pin differs")
    receipt = validate_feature_cache_binding(index, manifest)
    validate_window97_contract(receipt)
    require(receipt["index_sha256"] == info["sha256"]
            and receipt["manifest_sha256"] == info["manifest_sha256"], "index/manifest pin differs")
    original = Path(receipt["source_manifest"]).resolve()
    require(sha256_file(original) == receipt["source_manifest_sha256"], "original manifest differs")
    return index, _unique_rows(index), _unique_rows(manifest), _unique_rows(original)


class ExactCachedDataset(ResampledActionDataset):
    """Read exact selected windows with the existing native97 TI2V objective.

    cache_bindings uses the existing result.indexes.train vocabulary per origin:
    path, sha256, receipt, receipt_sha256, manifest, manifest_sha256. Caller pins
    these to trusted completed-cache records. Only selected episode receipts are
    opened. At most two read-only memory-mapped shards are retained per worker.
    """
    def __init__(self, selection, *, cache_bindings, num_samples, seed=42):
        require(isinstance(selection, ExactWindowSelection), "explicit ExactWindowSelection required")
        require(type(num_samples) is int and num_samples > 0 and type(seed) is int, "invalid sample count/seed")
        self.num_samples, self.seed, self.timestep_shift = num_samples, seed, 5.0
        self.data_seed = _seed64(seed, NAMESPACE)
        self.expected_latent_shape, self.expected_action_shape = action_teacher_shapes(97)
        self.episodes = copy.deepcopy(selection.rows)  # each slot is an exact WINDOW, not an episode sampler
        self._shards, self._verified = OrderedDict(), set()
        origins = {r["cache_origin"] for r in self.episodes}
        require(origins <= cache_bindings.keys(), "missing cache origin binding")
        sources = {origin: _read_bound_source(cache_bindings[origin]) for origin in origins}
        episode_receipts = {}
        for row in self.episodes:
            index, entries, manifests, originals = sources[row["cache_origin"]]
            identity = row["episode_id"]
            require(identity in entries and identity in manifests and identity in originals, "selected episode absent from cache")
            entry, manifest, original = entries[identity], manifests[identity], originals[identity]
            require(entry.get("kind") == WINDOW97_CACHE_KIND
                    and all(r.get("split") == "train" for r in (entry, manifest, original))
                    and all(row[k] == manifest[k] == original[k] for k in ("video_path", "annotations_path")),
                    "selected original train provenance differs")
            receipt_key = row["cache_origin"], identity
            if receipt_key not in episode_receipts:
                bound = entry["episode_receipt"]
                path = _inside(index.parent, bound["path"])
                require(sha256_file(path) == bound["sha256"], "episode receipt pin differs")
                receipt = json.loads(path.read_text(encoding="utf-8"))
                validate_window97_contract({**receipt, "config": receipt["cache_binding"]["config"]})
                require(all(receipt[k] == entry[k] for k in ("episode_id", "split", "shards", "num_windows")),
                        "episode receipt/index differs")
                starts = receipt["cache_binding"]["selected_source_starts"]
                require(starts == manifest["selected_source_starts"] and len(starts) == entry["num_windows"],
                        "receipt/manifest window offsets differ")
                episode_receipts[receipt_key] = starts
            starts = episode_receipts[receipt_key]
            require(row["window_index"] < len(starts) and starts[row["window_index"]] == row["source_start"],
                    "exact window_index/source_start differs")
            require(all(type(s["samples"]) is int and s["samples"] > 0 for s in entry["shards"])
                    and sum(s["samples"] for s in entry["shards"]) == len(starts), "invalid indexed shard counts")
            offset = row["window_index"]
            for shard in entry["shards"]:
                if offset < shard["samples"]:
                    path = _inside(index.parent, shard["path"])
                    require(path == Path(row["shard_path"]).resolve()
                            and shard["sha256"] == row["recorded_shard_sha256"], "exact shard path/recorded SHA differs")
                    row.update(_path=path, _offset=offset, _count=shard["samples"])
                    self._check_stat(row)
                    break
                offset -= shard["samples"]
        self.contract = {**selection.contract, "cache_bindings": copy.deepcopy(cache_bindings),
            "training_seed": seed, "data_seed": self.data_seed, "virtual_samples": num_samples,
            "diffusion_seed_policy": "seed64(data_seed,diffusion,absolute_sample_index)",
            "shard_verification": "recorded_SHA_binding_plus_bytes_mtime_each_access;no_new_full_hash"}

    @staticmethod
    def _check_stat(row):
        stat = row["_path"].stat()
        require(dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) == row["shard_stat"], "selected cached shard changed")

    def sampling_spec(self, index):
        require(not isinstance(index, bool), "sample index cannot be bool")
        index = integer_index(index)
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        slot = index % len(self.episodes)
        row = self.episodes[slot]
        return dict(absolute_sample_index=index, episode_slot=slot,
            **dict(zip(IDENTITY_FIELDS, window_key(row))), source_target=row["target"],
            index_namespace=NAMESPACE, diffusion_seed=_seed64(self.data_seed, "diffusion", index))

    def _cached_window(self, row, window_index):
        require(window_index == row["window_index"], "unselected window requested")
        self._check_stat(row)
        path = row["_path"]
        if path not in self._shards:
            self._shards[path] = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if len(self._shards) > 2:
                self._shards.popitem(last=False)
        self._shards.move_to_end(path)
        payload = self._shards[path]
        count, offset = row["_count"], row["_offset"]
        require(payload.get("schema_version") == 1 and payload.get("kind") == WINDOW97_CACHE_KIND
                and payload.get("episode_id") == row["episode_id"]
                and payload.get("encoding_policy") == WINDOW97_ENCODING_POLICY
                and payload.get("prompt_policy") == SCENE_STATIC_PROMPT_POLICY
                and "y" not in payload and "clip_fea" not in payload, "unexpected cache identity/conditioning")
        require(tuple(payload["clean_latents"].shape) == (count, *self.expected_latent_shape)
                and tuple(payload["actions"].shape) == (count, *self.expected_action_shape)
                and tuple(payload["source_starts"].shape) == (count,)
                and int(payload["source_starts"][offset]) == row["source_start"], "cached tensor shape/source_start differs")
        # Only slice the named window; sibling windows are never sampled or scanned.
        latent, actions, prompt = (payload["clean_latents"][offset], payload["actions"][offset], payload["prompt_embeds"])
        require(all(torch.is_tensor(t) and t.device.type == "cpu" and t.is_floating_point() for t in (latent, actions, prompt))
                and prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096,
                "invalid native97 CPU feature dtype/static prompt shape")
        if window_key(row) not in self._verified:
            require(all(bool(torch.isfinite(t).all()) for t in (latent, actions, prompt))
                    and torch.equal(actions.float(), torch.tensor(row["future_actions_96x8"], dtype=torch.float32)),
                    "selected tensors are nonfinite or actual 96x8 actions differ")
            self._verified.add(window_key(row))
        return dict(clean_latents=latent.to(dtype=torch.float32, copy=True),
            actions=actions.to(dtype=torch.float32, copy=True), prompt_embeds=prompt.to(dtype=torch.float32, copy=True))
