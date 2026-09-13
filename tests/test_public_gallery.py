"""CPU publication checks for explicit generated-gallery files; no GPU/model."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "assets/demos/manifest.json").read_text(encoding="utf-8"))

@pytest.mark.parametrize("clip", MANIFEST["clips"], ids=lambda clip: clip["id"])
def test_selected_media_bytes_and_geometry(clip):
    for field, size, digest in (
        ("video", "video_bytes", "video_sha256"),
        ("poster", "poster_bytes", "poster_sha256"),
    ):
        relative = Path(clip[field])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.parts[:2] == ("assets", "demos")
        payload = (ROOT / relative).read_bytes()
        assert len(payload) == clip[size]
        assert hashlib.sha256(payload).hexdigest() == clip[digest]
    assert clip["fps"] == "16/1"
    assert (clip["width"], clip["height"]) == (832, 528)
    assert clip["frames"] == (49 if clip["id"] == "action-r003-240-3s" else 241)
    assert abs(clip["duration_seconds"] - clip["frames"] / 16) < .01
    assert clip["audio_included"] is False
    assert "source_path" not in clip

def test_gallery_selection_and_preview():
    ids = [clip["id"] for clip in MANIFEST["clips"]]
    assert len(ids) == len(set(ids)) == 17
    historical = [clip_id for clip_id in ids if not clip_id.startswith("pair-")]
    assert len(historical) == 13
    page = (ROOT / "docs/demo-gallery.html").read_text(encoding="utf-8")
    assert page.count("<video ") == 13
    for clip_id in historical:
        assert f"../assets/demos/{clip_id}.mp4" in page
    for preview in MANIFEST["additional_previews"]:
        payload = (ROOT / preview["path"]).read_bytes()
        assert len(payload) == preview["bytes"]
        assert hashlib.sha256(payload).hexdigest() == preview["sha256"]

def test_archived_pairs_are_same_condition_with_changed_controls():
    showcase = MANIFEST["paired_showcase"]
    assert showcase["checkpoint_step"] == 1040
    assert showcase["checkpoint_sha256"] == "9bc9f57cce1095cbbc472307bc3963da2e6153e9f0f19080b542b6bf118dfaee"
    assert showcase["method"] == "action_teacher_window6_15s_ui_v1"
    assert showcase["custom_action_ground_truth_available"] is False
    assert len(showcase["pairs"]) == 2
    images = []
    archive = (ROOT / "docs/action-control-pairs.md").read_text(encoding="utf-8")
    for pair in showcase["pairs"]:
        left, right = pair["clips"]
        for field in ("initial_sha256", "prompt", "seed", "source_episode_id", "noise_sha256_float32", "first_latent_sha256_float32"):
            assert left[field] == right[field]
        assert left["seed"] == 42
        assert left["actions_sha256"] != right["actions_sha256"]
        images.append(left["initial_sha256"])
        for clip in pair["clips"]:
            assert clip["weighted_rgb_mse"] is None
            assert sum(segment["frames"] for segment in clip["action_segments"]) == 240
            assert f"../assets/demos/{clip['id']}-preview.gif" in archive
            assert f"../assets/demos/{clip['id']}.mp4" in archive
    assert images[0] != images[1]

def test_homepage_is_visually_curated_not_a_hidden_pair_comparison():
    selection = MANIFEST["homepage_selection"]
    assert selection["visual_selection_not_average_quality"] is True
    assert selection["full_videos_unmodified"] is True
    assert selection["clip_ids"] == ["pair-b-forward-up", "pair-a-forward-left", "action-r005-1040-window6"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Publication gate" not in readme
    assert "docs/action-control-pairs.md" in readme
    assert "docs/demo-gallery.md" in readme
    assert readme.count("-preview.gif") == 3
    for clip_id in selection["clip_ids"]:
        assert f"assets/demos/{clip_id}-preview.gif" in readme
        assert f"assets/demos/{clip_id}.mp4" in readme
    for clip_id in ("pair-a-backward-right", "pair-b-backward-down", "causal-clean60", "causal-recycling60"):
        assert f"assets/demos/{clip_id}-preview.gif" not in readme

def test_public_scores_do_not_claim_the_web_clip_passed():
    summary = json.loads((ROOT / "docs/version-results.json").read_text(encoding="utf-8"))
    selected = summary["selected_demo"]
    assert selected["weighted_rgb_mse_for_this_published_web_clip"] is None
    assert selected["exact_matched_zero_shuffled_available"] is False
    assert selected["realtime_passed"] is False
    assert selected["precise_control_passed"] is False
