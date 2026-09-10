"""CPU integration of static demo text and actual tiny checkpoint lineage."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest
import torch
import yaml

from scripts import prepare_abot_demo_assets as assets
from training.causal_tf import CausalTeacherForcingConfig
from training.eval.rollout15s import (
    ActionSegment, LineageSpec, SceneSpec, load_rollout_config, verify_checkpoint_lineage,
)
from training.longforcing_lite import LongForcingConfig, ParentLineage
from training.paths import PINNED_MODEL_DIR
from training.runtime import sha256_file

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "configs/eval/rollout15s_v1.yaml"


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, *, stage="longforcing", static=True):
    data = tmp_path / "data"
    (data / "manifests").mkdir(parents=True)
    (data / "features").mkdir()
    manifest, index = data / "manifests/train.jsonl", data / "features/train.features.jsonl"
    records, bindings = [], {}
    for number in range(3):
        identity = f"episode-{number}"
        archive = data / f"{identity}.tar"
        text = f"A quiet stone courtyard {number}."
        action = {"fps": 16, "frames": [{"frame_id": i, "keys": {"W": i < 120}} for i in range(241)]}
        caption = {"scene_static": text, "narrative": "Turn right and walk forward."}
        with tarfile.open(archive, "w") as stream:
            for name, value in (("action.json", action), ("caption.json", caption)):
                raw = json.dumps(value).encode()
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                stream.addfile(member, io.BytesIO(raw))
        records.append({"episode_id": identity, "split": "dev", "annotations_path": str(archive),
                        "video_path": str(data / f"{identity}.mp4")})
        bindings[identity] = {"split": "dev", "prompt": text, "annotations_sha256": sha256_file(archive),
                              "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    index.write_text(manifest.read_text(), encoding="utf-8")
    feature_receipt = index.with_suffix(".jsonl.receipt.json")
    _write_json(feature_receipt, {"schema_version": 1, "index": str(index), "manifest": str(manifest),
                                "index_sha256": sha256_file(index), "manifest_sha256": sha256_file(manifest)})
    cache = data / "features/static.pt"
    torch.save({"schema_version": 1, "kind": "scene_static_prompt_cache",
                "prompt_embeds": {identity: torch.ones(2, 4096) for identity in bindings}}, cache)
    receipt = {"schema_version": 1, "kind": "scene_static_prompt_cache", "prompt_policy": "scene_static_only_v1",
               "cache_path": str(cache), "cache_sha256": sha256_file(cache), "manifest_sha256": sha256_file(manifest),
               "feature_index_sha256": sha256_file(index), "feature_receipt_sha256": sha256_file(feature_receipt),
               "encoder": {"kind": "unit-test"}, "episodes": bindings}
    _write_json(cache.with_suffix(".pt.receipt.json"), receipt)
    config = CausalTeacherForcingConfig() if stage == "causal" else LongForcingConfig()
    config.model.base_model_path = str(tmp_path / "models" / PINNED_MODEL_DIR)
    config.model.action_scale = 0.03 if static else 1.0
    config.data.manifest_path, config.data.feature_index_path = str(manifest), str(index)
    config.data.feature_receipt_path = str(feature_receipt)
    config.data.prompt_cache_path = str(cache) if static else None
    config.training.output_dir = str(tmp_path / "run")
    teacher_path, causal_path = tmp_path / "teacher.pt", tmp_path / "causal.pt"
    if stage == "causal":
        config.lineage.checkpoint_path = str(teacher_path)
    else:
        config.lineage.teacher_checkpoint_path = str(teacher_path)
        config.lineage.causal_checkpoint_path = str(causal_path)
        for field, suffix in (("long_feature_index_path", ""), ("long_feature_receipt_path", ".receipt.json")):
            path = data / ("features/train.long241.features.jsonl" + suffix)
            path.write_text("{}", encoding="utf-8")
            setattr(config.data, field, str(path))
    training_path = tmp_path / "training.yaml"
    training_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    lineage = assets.lineage_from_training(manifest, training_path, stage)
    shared_keys = ("dataset_manifest", "feature_index", "feature_receipt") + (("prompt_cache", "prompt_cache_receipt") if static else ())
    shared = {key: sha256_file(lineage["artifact_paths"][key]) for key in shared_keys}
    state = {"model.act_control_adapter.weight": torch.ones(1)}
    teacher = {"format_version": 1, "stage": "action_teacher_lora_v1", "step": 20,
               "trainable_model": state, "config": config.to_dict(), "manifest_hashes": shared}
    torch.save(teacher, teacher_path)
    parent = ParentLineage(str(teacher_path), sha256_file(teacher_path), teacher["stage"], 20, shared).as_dict()
    causal = {"format_version": 1, "stage": "causal_teacher_forcing_v1", "step": 20,
              "trainable_model": state, "parent_teacher": parent, "config": config.to_dict(),
              "manifest_hashes": {**shared, "training_config": sha256_file(training_path), "teacher_checkpoint": sha256_file(teacher_path)}}
    causal["config"]["model"]["independent_first_frame"] = True
    torch.save(causal, causal_path)
    hashes = {key: sha256_file(path) for key, path in lineage["artifact_paths"].items()}
    payload = {**causal, "config": config.to_dict(), "manifest_hashes": hashes}
    if stage == "longforcing":
        payload.update(stage="longforcing_lite_v1", method="LongForcing-lite", is_dmd=False,
                       parent_causal=ParentLineage(str(causal_path), sha256_file(causal_path), causal["stage"],
                                                  20, causal["manifest_hashes"]).as_dict())
    checkpoint = Path(lineage["checkpoint_path"])
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload, checkpoint)
    lineage["checkpoint_sha256"] = sha256_file(checkpoint)
    scenes = tuple(SceneSpec(scene_id=f"scene-{i}", source_episode_id=row["episode_id"] if static else None,
                            prompt=bindings[row["episode_id"]]["prompt"] if static else "Legacy narrative text.",
                            initial_frame_path=str(tmp_path / f"initial-{i}.npy"),
                            reference_frames_path=str(tmp_path / f"anchors-{i}.npz"), seed=42 + i,
                            action_segments=(ActionSegment(120, ("W",)), ActionSegment(120, ())))
                   for i, row in enumerate(records))
    evaluation = replace(load_rollout_config(TEMPLATE), lineage=LineageSpec(**lineage), scenes=scenes,
                         output_root=str(tmp_path / "eval"))
    evaluation.validate()
    return evaluation, payload, records, receipt, training_path


@pytest.mark.parametrize("stage", ["causal", "longforcing"])
@pytest.mark.parametrize("static", [False, True])
def test_actual_checkpoint_accepts_optional_static_pair_and_legacy(tmp_path, stage, static):
    config, _, _, _, _ = _fixture(tmp_path, stage=stage, static=static)
    result = verify_checkpoint_lineage(config)
    assert result["prompt_contract"]["policy"] == ("scene_static_only_v1" if static else "original_feature_cache_prompt")
    pair = {"prompt_cache", "prompt_cache_receipt"}
    assert pair.issubset(result["manifest_hashes"]) is static


@pytest.mark.parametrize("change", ["omitted", "hash", "path", "text", "source", "parent_scale"])
def test_evaluation_rejects_condition_mismatches(tmp_path, change):
    config, payload, _, _, _ = _fixture(tmp_path)
    checkpoint = Path(config.lineage.checkpoint_path)
    if change == "omitted":
        artifacts = {key: value for key, value in config.lineage.artifact_paths.items() if not key.startswith("prompt_cache")}
        config = replace(config, lineage=replace(config.lineage, artifact_paths=artifacts))
    elif change == "hash":
        Path(config.lineage.artifact_paths["prompt_cache"]).write_bytes(b"changed cache")
    elif change == "path":
        payload["config"]["data"]["prompt_cache_path"] = str(tmp_path / "other.pt")
    elif change in ("text", "source"):
        first = replace(config.scenes[0], **({"prompt": "Walk right."} if change == "text" else {"source_episode_id": "unknown"}))
        config = replace(config, scenes=(first, *config.scenes[1:]))
    else:
        path = Path(config.lineage.artifact_paths["causal_checkpoint"])
        causal = torch.load(path, map_location="cpu", weights_only=False)
        causal["config"]["model"]["action_scale"] = 1.0
        torch.save(causal, path)
        payload["manifest_hashes"]["causal_checkpoint"] = sha256_file(path)
        payload["parent_causal"]["sha256"] = sha256_file(path)
    if change in ("path", "parent_scale"):
        torch.save(payload, checkpoint)
        config = replace(config, lineage=replace(config.lineage, checkpoint_sha256=sha256_file(checkpoint)))
    with pytest.raises(ValueError, match="action_scale" if change == "parent_scale" else "prompt|artifact"):
        verify_checkpoint_lineage(config)


def test_config_rejects_half_pair_and_missing_source_identity(tmp_path):
    config, _, _, _, _ = _fixture(tmp_path)
    artifacts = dict(config.lineage.artifact_paths)
    artifacts.pop("prompt_cache_receipt")
    with pytest.raises(ValueError, match="both cache and receipt"):
        replace(config, lineage=replace(config.lineage, artifact_paths=artifacts)).validate()
    with pytest.raises(ValueError, match="source_episode_id"):
        replace(config, scenes=(replace(config.scenes[0], source_episode_id=None), *config.scenes[1:])).validate()


def test_prepare_reads_exact_static_sidecar_text_and_serializes_source_identity(tmp_path, monkeypatch):
    config, _, records, expected, training_path = _fixture(tmp_path)
    monkeypatch.setattr(assets, "decode_video_window", lambda *args, **kwargs: (torch.zeros(3, 241, 2, 2, dtype=torch.uint8), "unit-test-media"))
    result = assets.prepare(Path(config.lineage.artifact_paths["dataset_manifest"]), tmp_path / "demo", TEMPLATE,
                            training_config=training_path, stage="longforcing", project_root=tmp_path)
    output = load_rollout_config(result["config"])
    assert result["prompt_policy"] == "scene_static_only_v1"
    assert set(result["prompt_artifacts"]) == {"prompt_cache", "prompt_cache_receipt"}
    for scene, record, emitted in zip(output.scenes, records, result["scenes"], strict=True):
        binding = expected["episodes"][record["episode_id"]]
        assert scene.source_episode_id == record["episode_id"]
        assert scene.prompt == binding["prompt"]
        assert "Turn right" not in scene.prompt
        assert emitted["prompt_sha256"] == binding["prompt_sha256"]


@pytest.mark.parametrize("change", ["missing_static", "missing_binding", "changed_annotation"])
def test_static_prompt_has_no_narrative_fallback(tmp_path, change):
    config, _, records, _, _ = _fixture(tmp_path)
    receipt = assets.prompt_receipt_from_lineage({"artifact_paths": config.lineage.artifact_paths})
    record = records[0]
    caption = {"scene_static": receipt["episodes"][record["episode_id"]]["prompt"], "narrative": "Walk forward."}
    assert assets.prompt_for_episode(record, caption, receipt) == caption["scene_static"]
    assert "Walk forward." in assets.prompt_for_episode(record, caption, None)
    if change == "missing_static":
        caption.pop("scene_static")
    elif change == "missing_binding":
        receipt["episodes"].pop(record["episode_id"])
    else:
        Path(record["annotations_path"]).write_bytes(b"changed annotations")
    with pytest.raises(ValueError):
        assets.prompt_for_episode(record, caption, receipt)
