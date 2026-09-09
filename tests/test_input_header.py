from fractions import Fraction
import hashlib
import io
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from training.eval.input_header import (
    HEADER_HEIGHT, KEY_ACTIVE, KEY_IDLE, InputHeaderRenderer, _load_font, annotate_video,
)


def _inputs():
    frame = np.random.default_rng(42).integers(0, 256, (32, 832, 3), dtype=np.uint8)
    actions = np.zeros((3, 8), dtype=np.float32)
    actions[0, 0] = 1
    actions[1, 1] = 1
    return frame, actions


def test_header_preserves_body_and_action_frame_offset():
    frame, actions = _inputs()
    renderer = InputHeaderRenderer(initial_frame=frame, prompt="Walk forward", actions=actions, seed=7)
    for index, expected in enumerate(((), ("W",), ("A",), ())):
        rendered = renderer.render(frame, index)
        assert rendered.shape == (32 + HEADER_HEIGHT, 832, 3)
        np.testing.assert_array_equal(rendered[HEADER_HEIGHT:], frame)
        assert renderer.input_state(index) == {"label": "INIT" if index == 0 else "KEYS", "keys": expected}
        for key in ("W", "A"):
            x, y, _, _ = renderer.key_boxes[key]
            np.testing.assert_array_equal(rendered[y + 3, x + 3], KEY_ACTIVE if key in expected else KEY_IDLE)
    assert renderer.metadata()["future_action_frame_offset"] == 1


def test_long_prompt_scrolls_and_complete_original_is_retained():
    frame, actions = _inputs()
    prompt = "\n".join(["Walk toward the trees and turn left at the next street."] * 12)
    renderer = InputHeaderRenderer(initial_frame=frame, prompt=prompt, actions=actions, seed=7)
    assert renderer.scroll_distance > 0
    first = renderer.render(frame, 0)
    last = renderer.render(frame, 3)
    assert not np.array_equal(first[19:45, renderer.prompt_left:renderer.keys_left - 12],
                              last[19:45, renderer.prompt_left:renderer.keys_left - 12])
    assert renderer.metadata()["prompt"] == prompt
    assert renderer.metadata()["prompt_display"] == "scrolling_excerpt"
    assert renderer.metadata()["prompt_scroll_pixels_per_second"] <= 32
    assert renderer.metadata()["actions_sha256"] == hashlib.sha256(actions.tobytes()).hexdigest()
    with patch("training.eval.input_header._font_candidates", return_value=[]):
        with pytest.raises(RuntimeError, match="No available font covers"):
            _load_font("向前走，左转")


def test_invalid_actions_and_existing_output_do_not_overwrite(tmp_path):
    frame, actions = _inputs()
    bad = actions.copy()
    bad[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        InputHeaderRenderer(initial_frame=frame, prompt="Walk", actions=bad, seed=0)
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"source evidence")
    output.write_bytes(b"existing output")
    with pytest.raises(FileExistsError):
        annotate_video(source, output, initial_frame=frame, prompt="Walk", actions=actions, seed=0)
    assert source.read_bytes() == b"source evidence"
    assert output.read_bytes() == b"existing output"


def test_encoder_failure_kills_both_processes_and_preserves_source(tmp_path):
    frame, actions = _inputs()
    source, output = tmp_path / "source.mp4", tmp_path / "display.mp4"
    source.write_bytes(b"original source evidence")

    class BrokenInput(io.BytesIO):
        def write(self, data):
            raise BrokenPipeError("simulated encoder failure")

    class Process:
        def __init__(self, stdin=None, stdout=None):
            self.stdin, self.stdout = stdin, stdout
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            return -9 if self.killed else 0

    decoder = Process(stdout=io.BytesIO(frame.tobytes()))
    encoder = Process(stdin=BrokenInput())
    with patch("training.eval.input_header._probe_video", return_value={"width": 832, "height": 32, "fps": Fraction(16)}), \
         patch("training.eval.input_header._font_candidates", return_value=[]), \
         patch("training.eval.input_header.subprocess.Popen", side_effect=[decoder, encoder]):
        with pytest.raises(BrokenPipeError):
            annotate_video(source, output, initial_frame=frame, prompt="Walk", actions=actions, seed=0)
    assert decoder.killed and encoder.killed
    assert not output.exists()
    assert source.read_bytes() == b"original source evidence"
    assert list(tmp_path.glob(".input-header-*")) == []


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="CPU ffmpeg tools unavailable")
def test_real_cpu_ffmpeg_annotation_and_frame_count(tmp_path):
    frame, actions = _inputs()
    source, output = tmp_path / "source.mp4", tmp_path / "display.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "832x32", "-r", "16", "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        input=np.repeat(frame[None], 4, axis=0).tobytes(), check=True, timeout=30,
    )
    original = source.read_bytes()
    result = annotate_video(source, output, initial_frame=frame, prompt="Walk forward, then left", actions=actions, seed=7)
    assert (result["width"], result["height"], result["frames"], result["fps"]) == (832, 80, 4, 16)
    assert result["video"] == "display.mp4"
    assert result["video_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert source.read_bytes() == original
    assert result["source_video_sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="more frames"):
        annotate_video(source, tmp_path / "wrong-count.mp4", initial_frame=frame, prompt="Walk", actions=actions[:1], seed=7)
    assert not (tmp_path / "wrong-count.mp4").exists()
