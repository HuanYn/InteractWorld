"""CPU-only display annotation; the added input strip never conditions the model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER_HEIGHT = 48
ACTION_KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
BACKGROUND = (16, 21, 28)
KEY_ACTIVE = (51, 174, 130)
KEY_IDLE = (43, 53, 66)
ARROW_LABELS = {"I": "↑", "J": "←", "K": "↓", "L": "→"}
HUD_BACKGROUND_ALPHA = 170
HUD_KEY_ALPHA = 220


def _hash_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _font_candidates(cjk: bool) -> list[str]:
    cjk_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/NotoSansSC-VF.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    latin_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    ]
    configured = os.environ.get("ABOT_HEADER_FONT")
    result = ([configured] if configured else []) + (cjk_fonts + latin_fonts if cjk else latin_fonts + cjk_fonts)
    if shutil.which("fc-match"):
        match = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans:lang=zh-cn" if cjk else "sans"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if match.returncode == 0 and match.stdout.strip():
            result.append(match.stdout.strip())
    return list(dict.fromkeys(result))


def _has_glyphs(font: Any, text: str) -> bool:
    # FreeType renders unsupported codepoints using the same .notdef glyph.
    # Check actual coverage, rather than accepting fc-match's possible Latin fallback.
    try:
        absent = font.getmask(chr(0x10FFFF))
        absent_signature = (absent.size, bytes(absent))
        for char in set(text):
            if char.isspace():
                continue
            glyph = font.getmask(char)
            if (glyph.size, bytes(glyph)) == absent_signature:
                return False
        return True
    except (UnicodeError, OSError):
        return False


def _load_font(text: str, size: int = 14):
    cjk = any("\u2e80" <= char <= "\ua4cf" or "\uac00" <= char <= "\ud7af" or "\uf900" <= char <= "\ufaff" for char in text)
    for candidate in _font_candidates(cjk):
        try:
            font = ImageFont.truetype(candidate, size=size)
        except (OSError, ValueError):
            continue
        if _has_glyphs(font, text):
            return font, candidate
    if all(ord(char) < 128 for char in text):
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # Older Pillow still has a readable ASCII fallback.
            font = ImageFont.load_default()
        if _has_glyphs(font, text):
            return font, "Pillow default ASCII"
    raise RuntimeError(
        "No available font covers the complete prompt. Install Noto Sans CJK under the configured data root "
        "or set ABOT_HEADER_FONT to a readable font file; refusing missing-glyph boxes."
    )


class InputHeaderRenderer:
    """Keep the 48px photo/prompt strip; overlay two display-only action HUDs."""

    def __init__(self, *, initial_frame: np.ndarray, prompt: str, actions: np.ndarray, seed: int,
                 fps: int = 16, layout: str = "split_hud"):
        if not isinstance(initial_frame, np.ndarray) or initial_frame.dtype != np.uint8 or initial_frame.ndim != 3 or initial_frame.shape[2] != 3:
            raise ValueError("initial_frame must be uint8 [H,W,3]")
        self.height, self.width = initial_frame.shape[:2]
        if self.width < 400 or self.height <= 0:
            raise ValueError("input-header layout needs width >=400 and positive height")
        if layout not in ("split_hud", "inline"):
            raise ValueError("layout must be split_hud or inline")
        if layout == "split_hud" and self.height < 8:
            raise ValueError("split HUD layout needs source height >=8")
        self.layout = layout
        if not isinstance(actions, np.ndarray) or actions.dtype != np.float32 or actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError("actions must be float32 [future_frames,8]")
        if not np.isfinite(actions).all() or not np.logical_or(actions == 0, actions == 1).all():
            raise ValueError("actions must contain only finite binary 0/1 inputs")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must contain the actual nonempty model prompt")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.prompt, self.seed, self.fps = prompt, seed, fps
        self.actions = np.ascontiguousarray(actions).copy()
        self.frames = len(actions) + 1
        self.initial_frame_sha256 = hashlib.sha256(np.ascontiguousarray(initial_frame).tobytes()).hexdigest()
        display_prompt = " ".join(prompt.split())
        self.font, self.font_name = _load_font(display_prompt + " PROMPT INPUT KEYS INIT seed WASDIJKL0123456789", 14)
        self.label_font, _ = _load_font("PROMPT INPUT KEYS INIT seed0123456789", 10)
        self.base = Image.new("RGB", (self.width, HEADER_HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(self.base)
        thumb_width = 68
        thumb = Image.fromarray(initial_frame)
        factor = min(thumb_width / self.width, 40 / self.height)
        thumb = thumb.resize((max(1, round(self.width * factor)), max(1, round(self.height * factor))), Image.Resampling.LANCZOS)
        self.base.paste(thumb, (4 + (thumb_width - thumb.width) // 2, (HEADER_HEIGHT - thumb.height) // 2))
        self.keys_left = self.width - 8 * 24 - 4
        self.prompt_left = 80
        self.prompt_right = self.width - 8 if self.layout == "split_hud" else self.keys_left - 12
        self.prompt_width = self.prompt_right - self.prompt_left
        self.key_boxes = {
            key: (self.keys_left + index * 24, 23, self.keys_left + index * 24 + 20, 44)
            for index, key in enumerate(ACTION_KEYS)
        }
        self.hud_regions = {}
        if self.layout == "split_hud":
            # Coordinates are inclusive PIL boxes in the final header+body image.
            # Small CPU fixtures scale down; normal 480px videos use 30px keys.
            margin = min(16, max(1, self.height // 16))
            padding = min(8, max(1, self.height // 16))
            gap = min(5, max(1, self.height // 32))
            size = min(30, (self.height - 2 * margin - 2 * padding - gap) // 2)
            panel_width, panel_height = 3 * size + 2 * gap + 2 * padding, 2 * size + gap + 2 * padding
            top = HEADER_HEIGHT + self.height - margin - panel_height
            self.key_boxes = {}
            for group, left, keys in (("wasd", margin, ("W", "A", "S", "D")),
                                      ("arrows", self.width - margin - panel_width, ("I", "J", "K", "L"))):
                self.hud_regions[group] = (left, top, left + panel_width - 1, top + panel_height - 1)
                for key, (column, row) in zip(keys, ((1, 0), (0, 1), (1, 1), (2, 1))):
                    x, y = left + padding + column * (size + gap), top + padding + row * (size + gap)
                    self.key_boxes[key] = (x, y, x + size - 1, y + size - 1)
            self.hud_font, _ = _load_font("WASD", max(1, min(18, size - 4)))
        bbox = self.font.getbbox(display_prompt)
        text_width = max(1, bbox[2] - min(0, bbox[0]))
        self.prompt_strip = Image.new("RGB", (max(text_width + 2, self.prompt_width), 26), BACKGROUND)
        ImageDraw.Draw(self.prompt_strip).text((max(0, -bbox[0]), 0), display_prompt, font=self.font, fill=(236, 242, 249))
        self.scroll_distance = max(0, self.prompt_strip.width - self.prompt_width)
        # A long source prompt cannot be read in full within 15 seconds. Keep
        # motion comfortably slow and explicitly label excerpts; receipt keeps
        # the exact full text instead of silently losing it or racing past it.
        self.scroll_travel = min(self.scroll_distance, 32.0 * max(0, self.frames - 1) / self.fps)
        self.prompt_display = "scrolling_excerpt" if self.scroll_distance else "full"
        label = "PROMPT excerpt (full in receipt)" if self.scroll_distance else "PROMPT"
        label_strip = Image.new("RGB", (self.prompt_width, 15), BACKGROUND)
        ImageDraw.Draw(label_strip).text((0, 0), f"{label}  |  seed {seed}", font=self.label_font, fill=(155, 170, 188))
        self.base.paste(label_strip, (self.prompt_left, 3))

    def input_state(self, frame_index: int) -> dict:
        if not 0 <= frame_index < self.frames:
            raise IndexError("display frame index does not match the action sequence")
        keys = () if frame_index == 0 else tuple(key for key, active in zip(ACTION_KEYS, self.actions[frame_index - 1]) if active)
        return {"label": "INIT" if frame_index == 0 else "KEYS", "keys": keys}

    def render(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError("decoded body frame geometry/dtype changed")
        state = self.input_state(frame_index)
        header = self.base.copy()
        offset = round(self.scroll_travel * frame_index / max(1, self.frames - 1))
        header.paste(self.prompt_strip.crop((offset, 0, offset + self.prompt_width, 26)), (self.prompt_left, 19))
        if self.layout == "inline":
            draw = ImageDraw.Draw(header)
            draw.text((self.keys_left, 3), "INPUT " + state["label"], font=self.label_font, fill=(155, 170, 188))
            for key, box in self.key_boxes.items():
                active = key in state["keys"]
                draw.rounded_rectangle(box, radius=3, fill=KEY_ACTIVE if active else KEY_IDLE)
                draw.text((box[0] + 4, box[1] + 1), key, font=self.font, fill=(255, 255, 255) if active else (155, 170, 188))
            return np.concatenate((np.asarray(header), frame), axis=0)
        canvas = Image.fromarray(np.concatenate((np.asarray(header), frame), axis=0)).convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for region in self.hud_regions.values():
            draw.rounded_rectangle(region, radius=8, fill=(*BACKGROUND, HUD_BACKGROUND_ALPHA))
        for key, box in self.key_boxes.items():
            active = key in state["keys"]
            draw.rounded_rectangle(box, radius=3, fill=(*(KEY_ACTIVE if active else KEY_IDLE), HUD_KEY_ALPHA))
            color = (255, 255, 255, 255) if active else (175, 189, 205, 255)
            if key in ARROW_LABELS:
                self._draw_arrow(draw, key, box, color)
            else:
                bounds = self.hud_font.getbbox(key)
                x = (box[0] + box[2] - (bounds[2] - bounds[0])) / 2 - bounds[0]
                y = (box[1] + box[3] - (bounds[3] - bounds[1])) / 2 - bounds[1]
                draw.text((round(x), round(y)), key, font=self.hud_font, fill=color)
        return np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB"))

    @staticmethod
    def _draw_arrow(draw, key, box, color):
        """Seven-point vector arrow: no Unicode arrow font dependency."""
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        radius = max(1, (box[2] - box[0]) // 4)
        shaft = max(1, radius // 3)
        points = ((0, -radius), (-radius, 0), (-shaft, 0), (-shaft, radius),
                  (shaft, radius), (shaft, 0), (radius, 0))
        rotate = {"I": lambda x, y: (x, y), "J": lambda x, y: (y, -x),
                  "K": lambda x, y: (-x, -y), "L": lambda x, y: (-y, x)}[key]
        draw.polygon([(cx + dx, cy + dy) for dx, dy in (rotate(x, y) for x, y in points)], fill=color)

    def metadata(self) -> dict:
        return {
            "source_width": self.width, "source_height": self.height,
            "width": self.width, "height": self.height + HEADER_HEIGHT, "header_height": HEADER_HEIGHT,
            "frames": self.frames, "fps": self.fps, "prompt": self.prompt, "seed": self.seed,
            "actions_sha256": hashlib.sha256(self.actions.tobytes()).hexdigest(),
            "action_keys": list(ACTION_KEYS), "future_action_frame_offset": 1,
            "initial_frame_sha256": self.initial_frame_sha256, "font": self.font_name,
            "prompt_display": self.prompt_display,
            "prompt_total_overflow_pixels": self.scroll_distance,
            "prompt_scroll_pixels": self.scroll_travel,
            "prompt_scroll_pixels_per_second": self.scroll_travel * self.fps / max(1, self.frames - 1),
            "prompt_full_text_in_metadata": True,
            "prompt_region": [self.prompt_left, 19, self.prompt_right, 45],
            "prompt_region_coordinates": "output_xyxy_right_bottom_exclusive",
            "key_layout": self.layout,
            "key_display_labels": {key: ARROW_LABELS.get(key, key) if self.layout == "split_hud" else key for key in ACTION_KEYS},
            "key_boxes": {key: list(box) for key, box in self.key_boxes.items()},
            "hud_regions": {name: list(box) for name, box in self.hud_regions.items()},
            "display_box_coordinates": "output_xyxy_inclusive_including_header",
            "body_layout": ("unscaled_only_two_hud_regions_overlaid_before_video_encoding"
                            if self.layout == "split_hud" else "unscaled_unmodified_before_video_encoding"),
            "body_pixels_outside_hud_regions_unchanged_before_video_encoding": True,
            "display_only_not_model_conditioning": True,
        }


def _probe_video(path: Path, *, count_frames: bool = False) -> dict:
    if not shutil.which("ffprobe"):
        try:
            import av
        except ImportError as error:
            raise RuntimeError("ffprobe is unavailable; install the PyAV 'av' dependency for CPU video probing") from error
        # PyAV uses software decoding by default. Count decoded frames when
        # requested: container metadata may omit or misreport the frame count.
        with av.open(str(path), mode="r") as container:
            streams = container.streams.video[:1]
            if len(streams) != 1:
                raise ValueError("expected one selected video stream")
            stream = streams[0]
            rate = stream.average_rate or stream.base_rate
            if not rate:
                raise ValueError("selected video stream has no valid frame rate")
            return {"width": int(stream.width), "height": int(stream.height),
                    "fps": Fraction(rate),
                    "frames": sum(1 for _ in container.decode(stream)) if count_frames else None}
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command += ["-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames", "-of", "json", str(path)]
    probe = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("expected one selected video stream")
    stream = streams[0]
    rate = stream.get("avg_frame_rate")
    if not rate or rate == "0/0":
        rate = stream["r_frame_rate"]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": Fraction(rate), "frames": int(stream["nb_read_frames"]) if count_frames else None}


def _stop_process(process) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _read_frame(stream, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            break
        chunks.extend(part)
    if chunks and len(chunks) != length:
        raise RuntimeError("ffmpeg returned a truncated RGB frame")
    return bytes(chunks)


def annotate_video(source: Path, output: Path, *, initial_frame: np.ndarray, prompt: str,
                   actions: np.ndarray, seed: int, fps: int = 16, layout: str = "split_hud") -> dict:
    """Create a separate MP4; never overwrite the source or an existing output.

    The default changes body pixels only in two bottom HUD regions before
    re-encoding; legacy inline changes none. H.264/YUV420 compression can add
    codec differences elsewhere; this is not a lossless remux.
    """
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source, output = source.resolve(strict=True), output.resolve()
    if not source.is_file() or output.suffix.lower() != ".mp4":
        raise ValueError("source must be a video file and output must be a new MP4")
    renderer = InputHeaderRenderer(initial_frame=initial_frame, prompt=prompt, actions=actions, seed=seed, fps=fps, layout=layout)
    if renderer.width % 2 or renderer.height % 2:
        raise ValueError("YUV420 display output requires even source width and height")
    probe = _probe_video(source)
    if (probe["width"], probe["height"], probe["fps"]) != (renderer.width, renderer.height, Fraction(fps)):
        raise ValueError("source video geometry/fps does not match the supplied model inputs")
    source_digest = _hash_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Unique same-directory temporary paths keep concurrent invocations isolated
    # and leave all CPU annotation files on the caller's output filesystem.
    with tempfile.TemporaryDirectory(prefix=".input-header-", dir=output.parent) as temporary:
        work = Path(temporary)
        partial = work / "annotated.mp4"
        decoder = encoder = None
        with (work / "decoder.log").open("w+b") as decoder_log, (work / "encoder.log").open("w+b") as encoder_log:
            try:
                decoder = subprocess.Popen(
                    ["ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(source),
                     # -vsync 0 preserves decoded frames on both the deployed
                     # FFmpeg 4.2 and newer releases (-fps_mode requires newer).
                     "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=decoder_log,
                )
                encoder = subprocess.Popen(
                    ["ffmpeg", "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{renderer.width}x{renderer.height + HEADER_HEIGHT}", "-r", str(fps), "-i", "pipe:0",
                     "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", str(partial)],
                    stdin=subprocess.PIPE, stderr=encoder_log,
                )
                index = 0
                while data := _read_frame(decoder.stdout, renderer.width * renderer.height * 3):
                    if index >= renderer.frames:
                        raise ValueError("source has more frames than initial frame plus supplied actions")
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(renderer.height, renderer.width, 3)
                    encoder.stdin.write(renderer.render(frame, index).tobytes())
                    index += 1
                decoder.stdout.close()
                if decoder.wait(timeout=60) != 0:
                    raise RuntimeError("ffmpeg source decoding failed")
                if index != renderer.frames:
                    raise ValueError("source frame count does not match initial frame plus supplied actions")
                encoder.stdin.close()
                if encoder.wait(timeout=60) != 0:
                    raise RuntimeError("ffmpeg annotated-video encoding failed")
            except BaseException as error:
                for label, log in (("decoder", decoder_log), ("encoder", encoder_log)):
                    log.flush()
                    log.seek(0)
                    tail = log.read().decode("utf-8", errors="replace")[-3000:]
                    if tail:
                        detail = f"ffmpeg {label} stderr:\n{tail}"
                        # The deployed Python 3.10 lacks Exception.add_note.
                        # Always retain diagnostics in worker.log before the
                        # temporary directory is removed, without changing the
                        # exception type or swallowing the original failure.
                        print(detail, file=sys.stderr, flush=True)
                        if hasattr(error, "add_note"):
                            error.add_note(detail)
                raise
            finally:
                _stop_process(decoder)
                _stop_process(encoder)
        result_probe = _probe_video(partial, count_frames=True)
        if (result_probe["width"], result_probe["height"], result_probe["frames"], result_probe["fps"]) != (
            renderer.width, renderer.height + HEADER_HEIGHT, renderer.frames, Fraction(fps)
        ):
            raise RuntimeError("encoded display video failed dimensions/frame-count/fps validation")
        if _hash_file(source) != source_digest:
            raise RuntimeError("source video changed during annotation")
        video_digest = _hash_file(partial)
        # Unlike rename/replace, hard-link publication is atomic and refuses an
        # existing target even if another worker created it after our first check.
        os.link(partial, output)
    return {**renderer.metadata(), "video": output.name, "video_sha256": video_digest,
            "source_video": str(source), "source_video_sha256": source_digest}
