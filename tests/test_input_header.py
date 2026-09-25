from fractions import Fraction
import hashlib
import io
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training.eval.input_header import (
    ARROW_LABELS, HEADER_HEIGHT, HUD_KEY_ALPHA, KEY_ACTIVE, KEY_IDLE,
    InputHeaderRenderer, _load_font, _probe_video, annotate_video,
)


def _inputs():
    frame = np.random.default_rng(42).integers(0, 256, (32, 832, 3), dtype=np.uint8)
    actions = np.zeros((3, 8), dtype=np.float32)
    actions[0, 0] = 1
    actions[1, 1] = 1
    return frame, actions


@pytest.mark.parametrize("count_frames", [False, True])
@pytest.mark.parametrize("average_rate", [Fraction(30000, 1001), None])
def test_probe_without_ffprobe_uses_pyav_and_counts_decoded_frames(tmp_path, count_frames, average_rate):
    # Unit doubles verify fallback contracts, not a generated model video.
    stream = SimpleNamespace(width=832, height=480, average_rate=average_rate,
                             base_rate=Fraction(16), frames=999)
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.decode.return_value = iter([object(), object(), object(), object()])
    av_module = SimpleNamespace(open=MagicMock(return_value=container))
    source = tmp_path / "unit-only.mp4"
    with patch("training.eval.input_header.shutil.which", return_value=None), \
         patch.dict(sys.modules, {"av": av_module}), \
         patch("training.eval.input_header.subprocess.run") as run:
        result = _probe_video(source, count_frames=count_frames)
    assert result == dict(width=832, height=480, fps=average_rate or Fraction(16),
                          frames=4 if count_frames else None)
    assert isinstance(result["fps"], Fraction)
    av_module.open.assert_called_once_with(str(source), mode="r")
    if count_frames:
        container.decode.assert_called_once_with(stream)
    else:
        container.decode.assert_not_called()
    container.__exit__.assert_called_once()
    run.assert_not_called()


def test_probe_without_ffprobe_or_av_gives_explicit_dependency_error(tmp_path):
    with patch("training.eval.input_header.shutil.which", return_value=None), \
         patch.dict(sys.modules, {"av": None}):
        with pytest.raises(RuntimeError, match="ffprobe is unavailable.*PyAV 'av'"):
            _probe_video(tmp_path / "unit-only.mp4")


@pytest.mark.parametrize("streams,message", [
    ([], "expected one selected video stream"),
    ([SimpleNamespace(width=832, height=480, average_rate=None, base_rate=None)], "no valid frame rate"),
])
def test_pyav_probe_rejects_missing_video_or_frame_rate(tmp_path, streams, message):
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = streams
    with patch("training.eval.input_header.shutil.which", return_value=None), \
         patch.dict(sys.modules, {"av": SimpleNamespace(open=MagicMock(return_value=container))}):
        with pytest.raises(ValueError, match=message):
            _probe_video(tmp_path / "unit-only.mp4", count_frames=True)
    container.decode.assert_not_called()
    container.__exit__.assert_called_once()


def test_available_ffprobe_keeps_existing_command_and_fraction(tmp_path):
    source = tmp_path / "unit-only.mp4"
    response = SimpleNamespace(stdout='{"streams":[{"width":832,"height":480,"avg_frame_rate":"0/0",'
                                        '"r_frame_rate":"30000/1001","nb_read_frames":"4"}]}')
    with patch("training.eval.input_header.shutil.which", return_value="/existing/ffprobe"), \
         patch.dict(sys.modules, {"av": None}), \
         patch("training.eval.input_header.subprocess.run", return_value=response) as run:
        result = _probe_video(source, count_frames=True)
    assert result == dict(width=832, height=480, fps=Fraction(30000, 1001), frames=4)
    run.assert_called_once_with(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames", "-of", "json", str(source)],
        capture_output=True, text=True, check=True, timeout=60,
    )


def test_pyav_decode_failure_propagates_without_a_synthetic_count(tmp_path):
    stream = SimpleNamespace(width=832, height=480, average_rate=Fraction(16), base_rate=None)
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.decode.side_effect = RuntimeError("unit-test decode failure")
    with patch("training.eval.input_header.shutil.which", return_value=None), \
         patch.dict(sys.modules, {"av": SimpleNamespace(open=MagicMock(return_value=container))}):
        with pytest.raises(RuntimeError, match="unit-test decode failure"):
            _probe_video(tmp_path / "unit-only.mp4", count_frames=True)
    container.__exit__.assert_called_once()


def test_legacy_inline_header_preserves_body_and_action_frame_offset():
    frame, actions = _inputs()
    renderer = InputHeaderRenderer(initial_frame=frame, prompt="Walk forward", actions=actions, seed=7, layout="inline")
    for index, expected in enumerate(((), ("W",), ("A",), ())):
        rendered = renderer.render(frame, index)
        assert rendered.shape == (32 + HEADER_HEIGHT, 832, 3)
        np.testing.assert_array_equal(rendered[HEADER_HEIGHT:], frame)
        assert renderer.input_state(index) == {"label": "INIT" if index == 0 else "KEYS", "keys": expected}
        for key in ("W", "A"):
            x, y, _, _ = renderer.key_boxes[key]
            np.testing.assert_array_equal(rendered[y + 3, x + 3], KEY_ACTIVE if key in expected else KEY_IDLE)
    assert renderer.metadata()["future_action_frame_offset"] == 1


def test_split_hud_maps_all_real_action_keys_and_changes_only_two_regions():
    frame = np.random.default_rng(41).integers(0, 256, (480, 832, 3), dtype=np.uint8)
    original = frame.copy()
    actions = np.eye(8, dtype=np.float32)
    renderer = InputHeaderRenderer(initial_frame=frame, prompt="A static mountain scene.", actions=actions, seed=42)
    assert renderer.layout == "split_hud"
    boxes = renderer.key_boxes
    assert boxes["W"][0] == boxes["S"][0] and boxes["W"][1] < boxes["S"][1]
    assert boxes["A"][0] < boxes["S"][0] < boxes["D"][0]
    assert boxes["I"][0] == boxes["K"][0] and boxes["I"][1] < boxes["K"][1]
    assert boxes["J"][0] < boxes["K"][0] < boxes["L"][0]
    outside = np.ones(frame.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in renderer.hud_regions.values():
        assert 0 <= x1 <= x2 < 832 and HEADER_HEIGHT <= y1 <= y2 < 528
        outside[y1 - HEADER_HEIGHT:y2 - HEADER_HEIGHT + 1, x1:x2 + 1] = False
    for index in range(9):
        rendered = renderer.render(frame, index)
        expected = () if index == 0 else (tuple(boxes)[index - 1],)
        assert renderer.input_state(index)["keys"] == expected
        np.testing.assert_array_equal(rendered[HEADER_HEIGHT:][outside], frame[outside])
        for key, (x, y, _, _) in boxes.items():
            # The corner sample is inside the key fill but outside its symbol.
            color = np.array(KEY_ACTIVE if key in expected else KEY_IDLE)
            source = frame[y + 3 - HEADER_HEIGHT, x + 3].astype(int)
            blended = (color * HUD_KEY_ALPHA + source * (255 - HUD_KEY_ALPHA) + 127) // 255
            np.testing.assert_array_equal(rendered[y + 3, x + 3], blended)
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(actions, np.eye(8, dtype=np.float32))
    metadata = renderer.metadata()
    assert metadata["key_display_labels"] == {"W": "W", "A": "A", "S": "S", "D": "D", **ARROW_LABELS}
    assert metadata["action_keys"] == ["W", "A", "S", "D", "I", "J", "K", "L"]
    assert metadata["body_layout"] == "unscaled_only_two_hud_regions_overlaid_before_video_encoding"
    assert metadata["body_pixels_outside_hud_regions_unchanged_before_video_encoding"]
    assert metadata["actions_sha256"] == hashlib.sha256(actions.tobytes()).hexdigest()


def test_split_hud_expands_prompt_into_old_key_area_without_changing_photo_seed_or_text():
    frame, _ = _inputs()
    actions = np.ones((240, 8), dtype=np.float32)
    inputs = dict(initial_frame=frame, prompt="A quiet rocky courtyard. " * 80, actions=actions, seed=42)
    split, old = InputHeaderRenderer(**inputs), InputHeaderRenderer(**inputs, layout="inline")
    assert split.prompt_left == old.prompt_left == 80
    assert old.prompt_right == old.keys_left - 12 == 624
    assert split.prompt_right == split.width - 8 == 824
    assert split.prompt_width == old.prompt_width + 200 == 744
    np.testing.assert_array_equal(np.asarray(split.prompt_strip), np.asarray(old.prompt_strip))
    for renderer in (split, old):
        metadata = renderer.metadata()
        assert metadata["prompt"] == inputs["prompt"] and metadata["seed"] == 42
        assert metadata["prompt_scroll_pixels_per_second"] <= 32
        assert metadata["prompt_region"] == [80, 19, renderer.prompt_right, 45]
        assert metadata["prompt_region_coordinates"] == "output_xyxy_right_bottom_exclusive"
    assert old.scroll_distance - split.scroll_distance == 200
    assert all(box[1:] == (23, old.keys_left + index * 24 + 20, 44)
               for index, box in enumerate(old.key_boxes.values()))
    for index in (0, 1, 48, 120, 240):
        current, legacy = split.render(frame, index), old.render(frame, index)
        np.testing.assert_array_equal(current[:HEADER_HEIGHT, :80], legacy[:HEADER_HEIGHT, :80])
        np.testing.assert_array_equal(current[3:18, 80:old.prompt_right], legacy[3:18, 80:old.prompt_right])
        # The former inline key area contains actual prompt glyphs, not blank padding or keys.
        extension = current[19:45, split.keys_left:split.prompt_right]
        offset = round(split.scroll_travel * index / max(1, split.frames - 1))
        expected = split.prompt_strip.crop((offset + split.keys_left - split.prompt_left, 0,
                                           offset + split.prompt_width, 26))
        np.testing.assert_array_equal(extension, np.asarray(expected))
        assert np.any(extension != np.asarray(split.base)[19:45, split.keys_left:split.prompt_right])
        np.testing.assert_array_equal(current[:HEADER_HEIGHT, split.prompt_right:],
                                      np.asarray(split.base)[:, split.prompt_right:])


@pytest.mark.parametrize("key,tip", [("I", (24, 17)), ("J", (17, 24)), ("K", (24, 31)), ("L", (31, 24))])
def test_camera_directions_use_rotated_vector_arrows_not_font_glyphs(key, tip):
    from unittest.mock import Mock
    draw = Mock()
    InputHeaderRenderer._draw_arrow(draw, key, (10, 10, 39, 39), (255, 255, 255, 255))
    points = draw.polygon.call_args.args[0]
    assert len(points) == 7 and points[0] == tip
    draw.text.assert_not_called()


def test_long_prompt_scrolls_and_complete_original_is_retained():
    frame, actions = _inputs()
    prompt = "\n".join(["Walk toward the trees and turn left at the next street."] * 12)
    renderer = InputHeaderRenderer(initial_frame=frame, prompt=prompt, actions=actions, seed=7)
    assert renderer.scroll_distance > 0
    first = renderer.render(frame, 0)
    last = renderer.render(frame, 3)
    assert not np.array_equal(first[19:45, renderer.prompt_left:renderer.prompt_right],
                              last[19:45, renderer.prompt_left:renderer.prompt_right])
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


def test_decoder_uses_legacy_passthrough_and_logs_error_before_cleanup(tmp_path, capsys):
    frame, actions = _inputs()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"unchanged source evidence")
    decoder = MagicMock(stdin=None, stdout=io.BytesIO())
    decoder.poll.return_value = decoder.wait.return_value = 1
    encoder = MagicMock(stdin=io.BytesIO(), stdout=None)
    encoder.poll.return_value = None

    def spawn(command, **kwargs):
        if "pipe:1" in command:
            assert "-fps_mode" not in command and command[command.index("-vsync") + 1] == "0"
            kwargs["stderr"].write(b"specific software decoder failure\n")
            return decoder
        return encoder

    with patch("training.eval.input_header._probe_video", return_value={"width": 832, "height": 32, "fps": Fraction(16)}), \
         patch("training.eval.input_header._font_candidates", return_value=[]), \
         patch("training.eval.input_header.subprocess.Popen", side_effect=spawn):
        with pytest.raises(RuntimeError, match="source decoding failed"):
            annotate_video(source, tmp_path / "display.mp4", initial_frame=frame, prompt="Walk", actions=actions, seed=0)
    assert "ffmpeg decoder stderr:\nspecific software decoder failure" in capsys.readouterr().err
    assert source.read_bytes() == b"unchanged source evidence"
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
