import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/package_abot_demo.py'
SPEC = importlib.util.spec_from_file_location('package_abot_demo', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PackageTests(unittest.TestCase):
    def test_actual_encoded_contract(self):
        stream = dict(width=832, height=480, nb_read_frames='241', r_frame_rate='16/1')
        self.assertEqual(MODULE.validate_stream(stream)['timeline_span_seconds'], 15)
        with self.assertRaises(ValueError):
            MODULE.validate_stream(dict(stream, nb_read_frames='240'))
        with self.assertRaises(ValueError):
            MODULE.validate_stream(dict(stream, r_frame_rate='24/1'))

    def test_reject_receipt_video_outside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                MODULE.checked_video(Path(temporary), {'video': '../outside.mp4'})

    def test_display_stream_is_distinct_from_raw_contract(self):
        display = dict(width=832, height=528, nb_read_frames='241', r_frame_rate='16/1')
        self.assertEqual(MODULE.validate_stream(display, header_height=48)['generated_frame_height'], 480)
        with self.assertRaises(ValueError):
            MODULE.validate_stream(display)
        with self.assertRaises(ValueError):
            MODULE.validate_stream(display, header_height=96)

    def test_reject_escaped_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                MODULE.package(root / 'source', root.parent / 'outside', project=root)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_cpu_fixture_exports_six_frames(self):
        # A temporary synthetic test pattern, never a model-quality result.
        with tempfile.TemporaryDirectory(prefix='abot-export-cpu-fixture-') as temporary:
            root = Path(temporary)
            source = root / 'synthetic-source'
            source.mkdir()
            video = source / 'synthetic-test-pattern.mp4'
            subprocess.run([
                'ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=832x480:rate=16', '-frames:v', '241', '-threads', '1',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(video)],
                check=True)
            receipt = {'outcome': 'completed', 'kind': 'trained_continuous_preview',
                       'test_fixture_only': True,
                       'scenes': [{'scene_id': 'synthetic-test-pattern', 'video': video.name,
                                   'video_sha256': MODULE.digest(video)}]}
            (source / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            (source / 'index.html').write_text('<title>CPU test fixture only</title>', encoding='utf-8')
            output = root / 'synthetic-export'
            result = MODULE.package(source, output, project=root)
            self.assertEqual(len(result['scenes'][0]['inspection_frames']), 6)
            self.assertTrue(result['visual_review_required'])
            self.assertTrue((output / 'README.txt').is_file())
            readme = (output / 'README.txt').read_text(encoding='utf-8')
            self.assertIn('左下WASD、右下方向箭头（I↑ J← K↓ L→）', readme)
            self.assertIn('仅两个半透明HUD区域覆盖画面', readme)
            self.assertIn('保留未注释原视频', readme)

            # A CPU-only padded fixture verifies packaging and default playback,
            # not the annotation renderer or model quality.
            display = source / 'synthetic-test-pattern.inputs.mp4'
            subprocess.run([
                'ffmpeg', '-nostdin', '-v', 'error', '-i', str(video),
                '-vf', 'pad=iw:ih+48:0:48:color=black', '-threads', '1',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(display)], check=True)
            receipt['scenes'][0]['display_video'] = {
                'video': display.name, 'video_sha256': MODULE.digest(display), 'header_height': 48,
                'source_video_sha256': MODULE.digest(video)}
            (source / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            result = MODULE.package(source, root / 'synthetic-display-export', project=root)
            self.assertEqual(result['scenes'][0]['video'], display.name)
            self.assertEqual(result['scenes'][0]['raw_video'], video.name)
            self.assertEqual(result['scenes'][0]['stream']['height'], 528)
            self.assertEqual(result['scenes'][0]['generated_stream']['height'], 480)
            self.assertIn(display.name, (root / 'synthetic-display-export/index.html').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
