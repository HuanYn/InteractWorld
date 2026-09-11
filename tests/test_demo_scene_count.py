"""Bounded scene counts across CPU plans, source assets, and stage transitions."""
import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts import generate_abot_demo as generate
from scripts import prepare_abot_demo_assets as assets
from training.eval.rollout15s import load_rollout_config


TEMPLATE = Path(__file__).parents[1] / "configs/eval/rollout15s_v1.yaml"


def write_config(tmp_path, count, *, duplicate=False):
    raw = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    originals = raw["scenes"]
    raw["scenes"] = [{**originals[index % 3], "scene_id": f"scene-{index}"} for index in range(count)]
    if duplicate:
        raw["scenes"][1]["scene_id"] = raw["scenes"][0]["scene_id"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path, raw


@pytest.mark.parametrize("count", [1, 2, 3])
def test_config_and_generator_cpu_plan_accept_exact_requested_count(tmp_path, monkeypatch, capsys, count):
    config_path, _ = write_config(tmp_path, count)
    assert len(load_rollout_config(config_path).scenes) == count
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU plan must not query the GPU")
    monkeypatch.setattr("training.gpu_gate.query_dedicated_gpu", forbidden)
    monkeypatch.setattr(sys, "argv", ["generate_abot_demo.py", "--config", str(config_path),
                                      "--output", str(tmp_path / "output"), "--scene-count", str(count)])
    generate.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "cpu_plan" and plan["scenes"] == count
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("count,duplicate", [(0, False), (4, False), (2, True)])
def test_config_rejects_out_of_range_count_and_duplicate_ids(tmp_path, count, duplicate):
    config_path, _ = write_config(tmp_path, count, duplicate=duplicate)
    with pytest.raises(ValueError, match="one to three uniquely named scenes"):
        load_rollout_config(config_path)


def test_generator_rejects_request_larger_than_config_without_silent_truncation(tmp_path, monkeypatch):
    config_path, _ = write_config(tmp_path, 1)
    monkeypatch.setattr(sys, "argv", ["generate_abot_demo.py", "--config", str(config_path),
                                      "--output", str(tmp_path / "output"), "--scene-count", "2"])
    with pytest.raises(ValueError, match="requested 2 scenes.*only 1"):
        generate.main()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("count", [0, 4])
def test_asset_preparation_rejects_invalid_count_before_reading_or_writing(tmp_path, count):
    with pytest.raises(ValueError, match="scene_count must be an integer from 1 to 3"):
        assets.prepare(tmp_path / "missing.jsonl", tmp_path / "output", TEMPLATE, scene_count=count)
    assert not (tmp_path / "output").exists()

