#!/usr/bin/env python3
"""Add the actual scene inputs above an existing generated video, using CPU only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--initial-frame', type=Path, help='Optional local copy of the scene initial .npy file')
    args = parser.parse_args()
    from training.eval.input_header import annotate_video
    from training.eval.rollout15s import action_script, default_image_loader, load_rollout_config
    from training.runtime import sha256_file

    config = load_rollout_config(args.config)
    scene = next((item for item in config.scenes if item.scene_id == args.scene_id), None)
    if scene is None:
        parser.error('scene-id is not present in the supplied generation config')
    receipt = args.output.with_suffix('.json')
    if args.output.exists() or receipt.exists():
        raise FileExistsError('presentation video and receipt paths must both be new')
    # Label only the actual inputs of this recorded generation, including when
    # its initial-frame file was copied locally from the remote machine.
    original = json.loads((args.source.parent / 'receipt.json').read_text(encoding='utf-8'))
    source_scene = next((item for item in original.get('scenes', [])
                         if item['scene_id'] == scene.scene_id and item['video'] == args.source.name), None)
    initial_path = args.initial_frame or Path(scene.initial_frame_path)
    if (source_scene is None or original.get('config_sha256') != sha256_file(args.config)
            or source_scene['video_sha256'] != sha256_file(args.source)
            or source_scene['initial_sha256'] != sha256_file(initial_path)
            or source_scene['seed'] != scene.seed):
        raise ValueError('source receipt, scene inputs and generation config must match')
    result = annotate_video(
        args.source, args.output,
        initial_frame=default_image_loader(initial_path),
        prompt=scene.prompt, actions=action_script(scene), seed=scene.seed, fps=config.fps,
    )
    result.update(kind='input_header_presentation_only', scene_id=scene.scene_id,
                  generation_config=str(args.config), quality_improvement_claim=False)
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
