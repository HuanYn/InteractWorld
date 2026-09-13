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
    assert len(ids) == len(set(ids)) == 13
    page = (ROOT / "docs/demo-gallery.html").read_text(encoding="utf-8")
    assert page.count("<video ") == 13
    for clip_id in ids:
        assert f"../assets/demos/{clip_id}.mp4" in page
    for preview in MANIFEST["additional_previews"]:
        payload = (ROOT / preview["path"]).read_bytes()
        assert len(payload) == preview["bytes"]
        assert hashlib.sha256(payload).hexdigest() == preview["sha256"]

def test_public_scores_do_not_claim_the_web_clip_passed():
    summary = json.loads((ROOT / "docs/version-results.json").read_text(encoding="utf-8"))
    selected = summary["selected_demo"]
    assert selected["weighted_rgb_mse_for_this_published_web_clip"] is None
    assert selected["exact_matched_zero_shuffled_available"] is False
    assert selected["realtime_passed"] is False
    assert selected["precise_control_passed"] is False
