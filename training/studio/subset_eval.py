"""Prepare a fixed-input quality trial for a completed weights-only subset branch.

This is CPU-only preparation, not generation or evidence of quality improvement.
The parent inputs are copied byte-for-byte. Only checkpoint/training lineage is
rebound; the existing Action validator and guarded quality worker remain in use.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from training.config import load_config
from training.demo.action_backend import load_action_inputs, validate_action_config
from training.demo.contracts import (
    ACTION_STAGE, ACTION_WINDOW6_METHOD, action_array, load_initial, require, sha256, write_json,
)
from training.demo.relocation import ArtifactHashes, artifact_paths
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, validate_cases


CASE_NAME = 'condition_robustness_full15'
CONTROLS = [dict(frames=120, keys=['W', 'J']), dict(frames=120, keys=['W', 'L'])]
FIXED_INPUTS = [('rollout.yaml', 'config_sha256'), ('initial.npy', 'initial_sha256'),
                ('actions.npy', 'actions_sha256')]


def prepare(parent_source_job, checkpoint, training_config, output):
    parent, checkpoint, config_path, directory = [Path(value).resolve() for value in
                                                (parent_source_job, checkpoint, training_config, output)]
    require(parent.is_dir(), 'parent source directory does not exist')
    require(checkpoint.is_file() and config_path.is_file(), 'completed checkpoint and training config must exist')
    require(not directory.exists(), 'output must be a new, nonexistent job directory')
    require(not directory.is_relative_to(parent), 'evaluation job must not be inside the preserved parent source')
    require(checkpoint.name not in ('best.pt', 'last.pt'), 'pin an immutable step checkpoint, not a mutable alias')
    original = json.loads((parent / 'request.json').read_text(encoding='utf-8'))
    for filename, field in FIXED_INPUTS:
        require(sha256(parent / filename) == original[field], f'frozen source changed: {filename}')
    raw = yaml.safe_load((parent / 'rollout.yaml').read_text(encoding='utf-8'))
    validate_action_config(raw)
    require(raw['method'] == ACTION_WINDOW6_METHOD and len(raw['scenes']) == 1,
            'parent source must contain one original40-step window6 scene')
    scene = raw['scenes'][0]
    require(scene['prompt'] == original['prompt'] and scene['source_episode_id'] == original['source_episode_id'],
            'source scene binding differs')
    require(scene['seed'] == original['seed'] == 42, 'fixed evaluation requires seed42')
    require(scene['action_segments'] == original['action_segments'] == CONTROLS,
            'fixed evaluation requires the original WJ120 then WL120 controls')
    initial, actions = load_initial(parent / 'initial.npy'), np.load(parent / 'actions.npy', allow_pickle=False)
    require(initial.shape == (480, 832, 3), 'fixed evaluation requires original832x480 RGB')
    require(actions.dtype == np.float32 and np.array_equal(actions, action_array(CONTROLS)),
            'source action tensor differs from declared controls')

    config = load_config(config_path)
    require(config.data.num_frames == 49, 'this subset evaluation expects49-frame training, not native6s training')
    require(config.model.action_scale == .03, 'paired quality comparison must preserve the parent action scale .03')
    require(config.data.prompt_cache_path is not None, 'subset evaluation needs its bound static prompt cache')
    paths = artifact_paths(config, config_path, {})
    require(all(path.is_file() for path in paths.values()), 'new training lineage has missing manifest/cache/receipt assets')
    old_spec = raw['lineage']
    require(checkpoint != Path(old_spec['checkpoint_path']).resolve(), 'new evaluation must not relabel the parent checkpoint')
    cache_path = old_spec.get('verification_cache')
    require(cache_path and Path(cache_path).is_absolute() and Path(cache_path).is_file(),
            'parent verified-artifact cache is required; no repeated full artifact audit')
    digest = ArtifactHashes(cache_path, protected_paths=[checkpoint, *paths.values(),
        Path(config.model.base_model_path) / 'Wan2.2_VAE.pth'])
    checkpoint_sha = digest(checkpoint)
    require(checkpoint_sha != old_spec['checkpoint_sha256'], 'new checkpoint bytes are identical to the parent')
    source = directory / 'source'
    frozen = copy.deepcopy(raw)
    selected = copy.deepcopy(scene)
    selected['initial_frame_path'] = str(source / 'initial.npy')
    frozen.update(run_id=source.name, scenes=[selected], output_root=str(source))
    frozen['lineage'] = dict(checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        expected_stage=ACTION_STAGE, expected_base_model_path=config.model.base_model_path,
        artifact_paths={key: str(path) for key, path in paths.items()}, verification_cache=str(cache_path))
    # Real CPU-only checkpoint/config/receipt/text validation. Never relax the
    # original loader to make a new checkpoint appear inference-compatible.
    validated = load_action_inputs(frozen, selected)
    lineage = copy.deepcopy(validated[3])
    del validated
    initialization = lineage.get('initialization')
    require(isinstance(initialization, dict) and initialization.get('mode') == 'subset_weights_only',
            'checkpoint is not a completed explicit subset weights-only branch')
    require(initialization.get('sha256') == old_spec['checkpoint_sha256']
            and initialization.get('source_stage') == ACTION_STAGE,
            'subset branch was not initialized from this paired parent checkpoint')
    require(initialization.get('optimizer_restored') is False and initialization.get('rng_restored') is False,
            'subset initialization must distinguish fresh optimizer/RNG from resume')
    case = validate_cases(dict(cases=[dict(name=CASE_NAME, mode='full15', solver='flow_euler',
        steps=40, cfg_scale=5, lora_strength=1, action_scale=.03)]))[0]
    metadata = dict(schema_version=1, status='cpu_evaluation_prepared', gpu_launched=False,
        video_generated=False, quality_improvement_verified=False, parent_source_job=str(parent),
        parent_checkpoint=dict(path=old_spec['checkpoint_path'], sha256=old_spec['checkpoint_sha256']),
        checkpoint=dict(path=str(checkpoint), sha256=checkpoint_sha, stage=lineage['stage'], step=lineage['step']),
        initialization=initialization, training_config=str(config_path), training_rgb_frames=49,
        inference_window_rgb_frames=[97, 97, 49], generated_seconds=15, native6s_training=False,
        fixed_inputs=dict(seed=42, prompt=selected['prompt'], source_episode_id=selected['source_episode_id'],
            initial_sha256=original['initial_sha256'], actions_sha256=original['actions_sha256'],
            action_segments=copy.deepcopy(CONTROLS), copied_bytes=True),
        prompt_condition=lineage['prompt_condition'],
        validation='original CPU load_action_inputs; new lineage, unchanged fixed inputs',
        limitation='49-frame subset adaptation evaluated with existing6+6+3 inference; '
                   'not native6s training, real-time generation, or verified quality improvement')

    # Publish only after CPU lineage validation succeeds. Old source/media/model
    # files are never overwritten; the shared stat-keyed verification cache is
    # the sole reused mutable index and contains no model or evaluation result.
    directory.mkdir(parents=True, exist_ok=False)
    source.mkdir(exist_ok=False)
    for filename in ('initial.npy', 'actions.npy'):
        shutil.copyfile(parent / filename, source / filename)
    for filename, field in FIXED_INPUTS[1:]:
        require(sha256(source / filename) == original[field], f'copied source differs: {filename}')
    rollout_path = source / 'rollout.yaml'
    rollout_path.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding='utf-8')
    source_request = copy.deepcopy(original)
    source_request.update(job_id=source.name, config_sha256=sha256(rollout_path),
        checkpoint_sha256=checkpoint_sha, checkpoint_stage=ACTION_STAGE, method=ACTION_WINDOW6_METHOD,
        source_rollout_sha256=original['config_sha256'], parent_source_job=str(parent),
        cpu_input_preparation_only=True, ground_truth_future_used=False)
    write_json(source / 'request.json', source_request)
    request = dict(job_id=directory.name, mode='quality_probe', max_seconds=1000,
        source_job=str(source), residency='co_resident', fastpath=False,
        negative_cache=DEFAULT_NEGATIVE_CACHE, cases=[case],
        evaluation_preparation='preparation.json', training_rgb_frames=49, native6s_training=False,
        checkpoint_sha256=checkpoint_sha, parent_checkpoint_sha256=old_spec['checkpoint_sha256'])
    write_json(directory / 'preparation.json', metadata)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_evaluation_prepared', request=str(directory / 'request.json'),
        source_job=str(source), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        gpu_launched=False, video_generated=False, quality_improvement_verified=False,
        native6s_training=False, max_seconds=1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare', help='CPU-only fixed-input evaluation preparation')
    command.add_argument('--parent-source-job', type=Path, required=True)
    command.add_argument('--checkpoint', type=Path, required=True)
    command.add_argument('--training-config', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    options = vars(parser.parse_args())
    options.pop('command')
    print(json.dumps(prepare(**options), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
