#!/usr/bin/env python3
"""Package a completed generated preview for playback and CPU visual inspection."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.paths import project_root

PROJECT = project_root()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked_video(source: Path, scene: dict) -> Path:
    path = (source / scene['video']).resolve()
    if path.parent != source.resolve() or path.suffix != '.mp4':
        raise ValueError('receipt video must be an MP4 directly inside the source directory')
    if digest(path) != scene['video_sha256']:
        raise ValueError('generated video does not match its receipt')
    return path


def validate_stream(stream: dict, *, header_height: int = 0) -> dict:
    if header_height not in (0, 48):
        raise ValueError('input header must be exactly 48 pixels')
    fps = Fraction(stream['r_frame_rate'])
    frames = int(stream['nb_read_frames'])
    if (int(stream['width']), int(stream['height']), frames, fps) != (832, 480 + header_height, 241, 16):
        raise ValueError(f'encoded video must actually contain 241 frames at 16 fps, 832x{480 + header_height}')
    return {'width': 832, 'height': 480 + header_height, 'frames': frames, 'fps': float(fps),
            'header_height': header_height, 'generated_frame_height': 480,
            'timeline_span_seconds': (frames - 1) / float(fps),
            'container_duration_seconds': stream.get('duration')}


def package(source: Path, output: Path, *, project: Path | None = None) -> dict:
    source, output, project = source.resolve(), output.resolve(), project_root(project).resolve()
    if project not in source.parents or project not in output.parents:
        raise ValueError('source and delivery must remain under the configured project')
    if source in output.parents or output in source.parents or source == output:
        raise ValueError('delivery must be separate from the generated source')
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((source / 'receipt.json').read_text(encoding='utf-8'))
    if receipt.get('outcome') != 'completed' or receipt.get('kind') != 'trained_continuous_preview':
        raise ValueError('only a completed self-trained generated preview can be packaged')
    scenes = receipt.get('scenes', [])
    if not scenes:
        raise ValueError('completed preview has no scenes')
    verified = []
    for scene in scenes:
        path = checked_video(source, scene)
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,duration',
            '-of', 'json', str(path)], text=True, capture_output=True, check=True)
        streams = json.loads(probe.stdout)['streams']
        if len(streams) != 1:
            raise ValueError('preview must have one video stream')
        raw_stream = validate_stream(streams[0])
        display = scene.get('display_video')
        display_path, display_stream = None, None
        if display is not None:
            if display.get('header_height') != 48:
                raise ValueError('display receipt must identify the 48-pixel input header')
            if display.get('source_video_sha256') != scene['video_sha256']:
                raise ValueError('input display must bind the exact raw generated video')
            display_path = checked_video(source, display)
            if display_path == path:
                raise ValueError('input display must not replace the raw generated video')
            probe_display = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,duration',
                '-of', 'json', str(display_path)], text=True, capture_output=True, check=True)
            display_streams = json.loads(probe_display.stdout)['streams']
            if len(display_streams) != 1:
                raise ValueError('input display must have one video stream')
            display_stream = validate_stream(display_streams[0], header_height=48)
        verified.append((path, scene, raw_stream, display_path, display_stream))
    if not (source / 'index.html').is_file():
        raise FileNotFoundError(source / 'index.html')
    output.mkdir(parents=True)
    for name in ('receipt.json',):
        shutil.copy2(source / name, output / name)
    result = {'kind': 'generated_demo_delivery', 'source': str(source),
              'source_receipt_sha256': digest(source / 'receipt.json'),
              'quality_evaluation': 'not_assessed_by_packager',
              'visual_review_required': True, 'scenes': []}
    for path, scene, stream, display_path, display_stream in verified:
        shutil.copy2(path, output / path.name)
        if display_path is not None:
            shutil.copy2(display_path, output / display_path.name)
        frames = output / (path.stem + '-frames')
        frames.mkdir()
        subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
            '-vf', r'select=not(mod(n\,48))', '-fps_mode', 'vfr',
            '-frames:v', '6', '-q:v', '2', str(frames / 'frame-%02d.jpg')], check=True)
        pictures = sorted(frames.glob('frame-*.jpg'))
        if len(pictures) != 6:
            raise RuntimeError('expected six inspectable frames at 0/3/6/9/12/15 seconds')
        playback = scene.get('display_video', scene)
        result['scenes'].append({'scene_id': scene['scene_id'], 'video': playback['video'],
                                 'video_sha256': playback['video_sha256'], 'stream': display_stream or stream,
                                 'raw_video': path.name, 'raw_video_sha256': scene['video_sha256'],
                                 'generated_stream': stream,
                                 'inspection_frames': [str(p.relative_to(output)) for p in pictures]})
    (output / 'delivery.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    cards = ''.join(
        f'<article><h2>{html.escape(item["scene_id"])}</h2>'
        f'<video controls src="{html.escape(item["video"], quote=True)}"></video></article>'
        for item in result['scenes'])
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>ABot-inspired trained preview</title>'
        '<style>body{background:#10151c;color:#eee;font:16px sans-serif;max-width:960px;margin:40px auto}'
        'video{width:100%}</style><h1>ABot-inspired 自训预览</h1>'
        '<p>输入栏仅展示模型收到的输入，不表示动作已准确执行；画质仍需验收。</p>' + cards,
        encoding='utf-8')
    checkpoint = receipt.get('lineage', {})
    (output / 'README.txt').write_text(
        'ABot-inspired 自训练视频 Demo\n\n'
        '播放：用浏览器打开 index.html，或直接播放 MP4。\n'
        '原始生成画面：832×480，16 fps，241帧，首尾时间跨度15秒。\n'
        '若含 .inputs.mp4 展示版：上方增加48像素输入栏，显示首帧、文字提示与当前按键，主体画面不裁剪；总尺寸832×528。\n'
        '输入栏属于导出后添加的信息，不会送回模型，也不参与原始画面评测。\n'
        '来源：Wan2.2 基座上的本项目训练权重；只给首帧、文字与动作输入，未来画面由模型生成。\n'
        '输入动作不等于已经验证动作正确执行；不是实时互动演示。\n'
        '画质仍需观看完整视频：frames目录的六张图片只辅助检查，不替代时序检查。\n'
        f"训练阶段：{receipt.get('stage')}\nCheckpoint：{checkpoint.get('path')}\n"
        f"Checkpoint SHA256：{checkpoint.get('sha256')}\n"
        'receipt.json记录生成来源，delivery.json记录实际视频帧数和打包信息。\n',
        encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--project-root', type=Path, help='Data root (or set INTERACTWORLD_ROOT)')
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output, project=args.project_root), indent=2))


if __name__ == '__main__':
    main()
