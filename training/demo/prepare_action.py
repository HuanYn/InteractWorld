"""Prepare a single Action-teacher scene/config on CPU; does not launch a service or GPU."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import yaml

from training.demo.contracts import ACTION_ADAPTER, ACTION_METHOD, ACTION_STAGE, KEYS, action_array, load_initial, require, sha256


def segments_from_array(array):
    array = np.asarray(array)
    if array.shape == (1, 240, 8):
        array = array[0]
    require(array.shape == (240, 8) and np.isfinite(array).all() and np.isin(array, [0, 1]).all(), 'preset actions must be240x8 finite binary values')
    segments = []
    for row in array:
        keys = [key for key, active in zip(KEYS, row) if active]
        if segments and segments[-1]['keys'] == keys:
            segments[-1]['frames'] += 1
        else:
            segments.append({'frames': 1, 'keys': keys})
    action_array(segments)
    return segments


def prepare(*, training_config, checkpoint, checkpoint_sha256, initial_frame, episode_id,
            action_input, output, seed=42, initial_origin='source_rgb'):
    from training.config import load_config
    from training.data.action_dataset import cache_index_path
    from training.demo.action_backend import load_action_inputs
    import torch
    require(not output.exists(), 'refusing to overwrite an existing Action preset config')
    require(type(seed) is int and 0 <= seed <= 2**32 - 1, 'preset seed must be uint32')
    config = load_config(training_config)
    require(config.data.prompt_cache_path is not None, 'Action UI requires the actual static prompt cache')
    initial = load_initial(initial_frame)
    require(initial.shape == (480, 832, 3), 'Action initial RGB must be832x480')
    require(initial_origin in ('source_rgb', 'decoded_condition_rgb'), 'unrecognized initial-frame provenance')
    index = cache_index_path(config.data.manifest_path)
    cache = Path(config.data.prompt_cache_path)
    receipt = json.loads(cache.with_suffix('.pt.receipt.json').read_text(encoding='utf-8'))
    binding = receipt.get('episodes', {}).get(episode_id)
    require(binding and binding.get('split') == 'dev', 'preset episode must have a real held-out static-text binding')
    if action_input.suffix == '.npy':
        actions = np.load(action_input, allow_pickle=False)
    else:
        # The existing frozen external-input bundle contains initial condition,
        # actions, text and noise, not future ground-truth video. Only actions
        # are copied: this UI reencodes its actual RGB and uses the request seed.
        payload = torch.load(action_input, map_location='cpu', weights_only=True)
        actions = payload['actions'].detach().cpu().numpy().copy()
        del payload
    scene = dict(scene_id='action-dev1', source_episode_id=episode_id, prompt=binding['prompt'],
                 initial_frame_path=str(initial_frame.resolve()), initial_origin=initial_origin,
                 initial_source_sha256=sha256(initial_frame),
                 seed=seed, action_segments=segments_from_array(actions),
                 preset_actions_source_sha256=sha256(action_input))
    raw = dict(version=1, run_id='action-ui-preset', method=ACTION_METHOD,
               adapter_factory=ACTION_ADAPTER,
               geometry=dict(width=832, height=480, fps=16, total_rgb_frames=241,
                             num_chunks=5, chunk_rgb_frames=49, chunk_future_frames=48),
               sampler=dict(solver='flow_euler', steps=40, shift=5.0, cfg='none'),
               lineage=dict(checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha256.lower(),
                            expected_stage=ACTION_STAGE, expected_base_model_path=config.model.base_model_path,
                            artifact_paths=dict(dataset_manifest=config.data.manifest_path,
                                                feature_index=str(index), feature_receipt=str(index.with_suffix('.jsonl.receipt.json')),
                                                training_config=str(training_config.resolve()), prompt_cache=str(cache),
                                                prompt_cache_receipt=str(cache.with_suffix('.pt.receipt.json')))),
               scenes=[scene])
    state, _, prompt, lineage = load_action_inputs(raw, scene)
    del state, prompt
    gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return dict(status='cpu_action_preset_prepared', config=str(output), config_sha256=sha256(output),
                checkpoint_step=lineage['step'], checkpoint_sha256=checkpoint_sha256,
                method=ACTION_METHOD, static_prompt=scene['prompt'], initial_origin=initial_origin,
                fixed_noise_regression=False, t5_loaded=False, gpu_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--initial-frame', type=Path, required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--action-input', type=Path, required=True, help='actual240x8.npy or existing frozen inputs.pt actions field')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial-origin', choices=('source_rgb', 'decoded_condition_rgb'), default='source_rgb')
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
