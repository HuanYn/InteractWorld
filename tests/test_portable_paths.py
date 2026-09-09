from pathlib import Path, PurePosixPath

import pytest
import yaml

from scripts.cache_abot_features import parse_args as cache_args
from scripts.cache_abot_long_features import parse_args as long_cache_args
from scripts.fetch_abot_subset import _assert_under_execution_root, parse_args as fetch_args
from scripts.prepare_abot_demo_assets import lineage_from_training
from training.paths import PINNED_MODEL_DIR, is_pinned_base_model, pinned_base_model_path, project_root


ROOT = Path(__file__).parents[1]


def test_project_root_is_explicit_and_environment_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTWORLD_ROOT", str(tmp_path / "data"))
    assert project_root() == tmp_path / "data"
    assert project_root(tmp_path / "other") == tmp_path / "other"
    assert pinned_base_model_path() == tmp_path / "data/models" / PINNED_MODEL_DIR
    for invalid in ("/", "C:\\", "relative", "~/data", "/data/../elsewhere"):
        with pytest.raises(ValueError, match="absolute non-root"):
            project_root(invalid)
    assert is_pinned_base_model("/data/reproduction/models/" + PINNED_MODEL_DIR)
    assert is_pinned_base_model("D:/reproduction/models/" + PINNED_MODEL_DIR)
    assert not is_pinned_base_model("models/" + PINNED_MODEL_DIR)
    assert not is_pinned_base_model("/data/models/Wan2.2-TI2V-5B@wrong")


def test_fetch_and_cache_cli_defaults_follow_selected_data_root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("INTERACTWORLD_ROOT", str(data))
    fetched = fetch_args([])
    assert fetched.metadata == data / "data/index/metadata.jsonl"
    assert fetched.payload_root == data / "data/ABot-World-Explorer-500h"
    explicit = fetch_args(["--project-root", str(tmp_path / "explicit")])
    assert explicit.metadata == tmp_path / "explicit/data/index/metadata.jsonl"
    _assert_under_execution_root(data / "payload", data)
    with pytest.raises(SystemExit, match="must stay under"):
        _assert_under_execution_root(tmp_path / "outside", data)
    for parse in (cache_args, long_cache_args):
        args = parse(["--manifest", str(data / "data/manifests/train.jsonl")])
        assert args.vae_path == (data / "models" / PINNED_MODEL_DIR / "Wan2.2_VAE.pth").as_posix()


@pytest.mark.parametrize("stage,template", [
    ("causal", "causal_teacher_forcing_5090_week.yaml"),
    ("longforcing", "longforcing_lite_5090_week.yaml"),
])
def test_demo_lineage_uses_actual_relocated_config_and_no_invented_hash(tmp_path, stage, template):
    data = tmp_path / "data"
    manifest = data / "data/manifests/train.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")
    raw = yaml.safe_load((ROOT / "configs/train" / template).read_text(encoding="utf-8"))
    old_root = str(PurePosixPath(raw["model"]["base_model_path"]).parent.parent)

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(old_root):
            return str(data) + value[len(old_root):]
        return value

    raw = relocate(raw)
    config_path = tmp_path / f"actual-{stage}.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    original = config_path.read_bytes()
    lineage = lineage_from_training(manifest, config_path, stage)
    assert lineage["artifact_paths"]["training_config"] == str(config_path.resolve())
    assert lineage["expected_base_model_path"] == str(data) + "/models/" + PINNED_MODEL_DIR
    assert lineage["checkpoint_path"] == str(Path(raw["training"]["output_dir"]) / "checkpoints/best.pt")
    assert lineage["checkpoint_sha256"] is None
    assert ("causal_checkpoint" in lineage["artifact_paths"]) == (stage == "longforcing")
    if stage == "longforcing":
        assert lineage["artifact_paths"]["teacher_checkpoint"] == raw["lineage"]["teacher_checkpoint_path"]
        assert lineage["artifact_paths"]["causal_checkpoint"] == raw["lineage"]["causal_checkpoint_path"]
    else:
        assert lineage["artifact_paths"]["teacher_checkpoint"] == raw["lineage"]["checkpoint_path"]
    assert config_path.read_bytes() == original
    with pytest.raises(ValueError, match="exact manifest"):
        lineage_from_training(data / "wrong.jsonl", config_path, stage)
