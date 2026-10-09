"""Prepare one pinned reverse-control15s history probe; CPU only, no new model.

Only WJ120/WL120 becomes WL120/WJ120. The complete verified forward request,
source request and rollout are bound by their small-file digests. No weights,
feature caches, generated histories or old videos are copied or overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from training.demo.contracts import ACTION_WINDOW6_METHOD, action_array, contained, require, sha256, write_json

FORWARD_CONTROLS = [dict(frames=120, keys=['W', 'J']), dict(frames=120, keys=['W', 'L'])]
REVERSE_CONTROLS = [dict(frames=120, keys=['W', 'L']), dict(frames=120, keys=['W', 'J'])]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _forward_request(path):
    from training.studio import long_history_probe as probe
    path = contained(path, probe.DATA_ROOT)
    request = _read_json(path)
    require(request.get('mode') == 'long_history_probe' and request.get('max_seconds') == 1200
            and request.get('residency', 'phase') == 'phase' and request.get('fastpath', False) is False
            and request.get('reverse_timeline', False) is False and 'reverse_binding' not in request
            and 'cases' not in request and 'action_adapter_donor_source' not in request,
            'reverse preparation requires a forward-only fixed1200s phase long-history request')
    for field, value in dict(width=probe.WIDTH, height=probe.HEIGHT, fps=probe.FPS,
                             seed=probe.SEED, **probe.SETTINGS).items():
        require(field not in request or request[field] == value, 'forward request changes fixed ' + field)
    probe.validate_source_binding(request)
    source = contained(request['source_job'], probe.DATA_ROOT)
    record = _read_json(source / 'request.json')
    for name, field in (('rollout.yaml', 'config_sha256'), ('initial.npy', 'initial_sha256'), ('actions.npy', 'actions_sha256')):
        require(sha256(source / name) == record[field], 'forward source changed: ' + name)
    raw = yaml.safe_load((source / 'rollout.yaml').read_text(encoding='utf-8'))
    require(raw['method'] == ACTION_WINDOW6_METHOD and len(raw['scenes']) == 1, 'one original window6 scene required')
    scene = raw['scenes'][0]
    require(scene['seed'] == record['seed'] == probe.SEED
            and scene['action_segments'] == record['action_segments'] == FORWARD_CONTROLS,
            'forward source must retain original seed42 and WJ120/WL120 controls')
    for field in ('prompt', 'source_episode_id'):
        require(scene[field] == record[field], 'forward scene differs: ' + field)
    require(raw['lineage']['checkpoint_sha256'] == record['checkpoint_sha256'] == request['checkpoint_sha256'],
            'forward checkpoint request/config binding differs')
    actions = np.load(source / 'actions.npy', allow_pickle=False)
    require(actions.dtype == np.float32 and np.array_equal(actions, action_array(FORWARD_CONTROLS)),
            'forward action tensor is not the fixed timeline')
    return path, request, source, record, raw


def _reverse_rollout(raw, source):
    result = copy.deepcopy(raw)
    result.update(run_id=source.name, output_root=str(source))
    result['scenes'][0].update(initial_frame_path=str(source / 'initial.npy'), action_segments=copy.deepcopy(REVERSE_CONTROLS))
    return result


def _reverse_source_record(original, source, forward, config_sha, actions_sha):
    return dict(copy.deepcopy(original), job_id=source.name, config_sha256=config_sha,
        actions_sha256=actions_sha, action_segments=copy.deepcopy(REVERSE_CONTROLS),
        reverse_timeline=True, forward_source_job=str(forward),
        forward_source_request_sha256=sha256(forward / 'request.json'),
        forward_rollout_sha256=original['config_sha256'], cpu_input_preparation_only=True,
        ground_truth_future_used=False)


def validate_reverse_source(request):
    """CPU source binding only; worker still validates actual weights and inputs."""
    from training.studio import long_history_probe as probe
    require(request.get('reverse_timeline') is True, 'reverse source needs an explicit true opt-in')
    binding = request.get('reverse_binding')
    require(isinstance(binding, dict) and binding.get('version') == 1
            and binding.get('kind') == 'long_history_reverse_timeline_v1', 'missing prepared reverse source binding')
    path = contained(binding['forward_request_path'], probe.DATA_ROOT)
    require(sha256(path) == binding['forward_request_sha256'], 'bound forward request changed')
    _, forward_request, forward, original, raw = _forward_request(path)
    source = contained(request['source_job'], probe.DATA_ROOT)
    require(source != forward and not source.is_relative_to(forward)
            and str(forward) == binding['forward_source_job'], 'reverse requires a separate source directory')
    require(sha256(forward / 'request.json') == binding['forward_source_request_sha256']
            and original['config_sha256'] == binding['forward_rollout_sha256'], 'bound forward source request/config changed')
    mutable = {'job_id', 'source_job', 'source_revision', 'experiment_revision', 'reverse_timeline', 'reverse_binding'}
    require({key: value for key, value in request.items() if key not in mutable}
            == {key: value for key, value in forward_request.items() if key not in mutable},
            'reverse request may not relabel checkpoint, candidate contract or original source')
    require(sha256(source / 'request.json') == binding['reverse_source_request_sha256']
            and sha256(source / 'rollout.yaml') == binding['reverse_rollout_sha256'],
            'prepared reverse source request/config changed')
    require(binding['initial_sha256'] == original['initial_sha256'] == sha256(source / 'initial.npy')
            and binding['forward_actions_sha256'] == original['actions_sha256'],
            'reverse source initial or original action binding differs')
    require(sha256(source / 'actions.npy') == binding['reverse_actions_sha256'], 'reverse action file changed')
    copied = _read_json(source / 'request.json')
    expected = _reverse_source_record(original, source, forward,
        binding['reverse_rollout_sha256'], binding['reverse_actions_sha256'])
    require(copied == expected, 'reverse source request changed more than the allowed action timeline')
    frozen = yaml.safe_load((source / 'rollout.yaml').read_text(encoding='utf-8'))
    require(frozen == _reverse_rollout(raw, source), 'reverse rollout changes model, prompt, seed or exact controls')
    actions = np.load(source / 'actions.npy', allow_pickle=False)
    require(actions.dtype == np.float32 and np.array_equal(actions, action_array(REVERSE_CONTROLS)),
            'reverse tensor must be exact WL120 then WJ120 controls')
    return dict(binding, reverse_timeline=True, original_r005_source_job=str(probe.R005_SOURCE_JOB.resolve()),
        parent_checkpoint_sha256=probe.R005_SHA, original_initial_copy_verified=True,
        prompt_seed_episode_checkpoint_preserved=True, action_timeline_only_change=True,
        action_segments=copy.deepcopy(REVERSE_CONTROLS), actions_sha256=binding['reverse_actions_sha256'])


def prepare(*, forward_request, output, source_revision):
    from training.studio import long_history_probe as probe
    from training.studio.worker import read_source
    directory = contained(output, probe.DATA_ROOT)
    require(not directory.exists(), 'reverse job requires a new nonexistent output directory')
    path, forward, parent, original, raw = _forward_request(forward_request)
    require(not directory.is_relative_to(parent), 'reverse job cannot overwrite or nest inside forward source')
    loaded = read_source(forward)  # Original CPU-only asset/config/text verification remains intact.
    _, _, _, lineage, initial, actions, scene = loaded
    candidate_contract = probe.validate_loaded_source(forward, lineage)
    probe.validate_action_timeline(actions, scene, reverse_timeline=False)
    require(initial.dtype == np.uint8 and initial.shape == (probe.HEIGHT, probe.WIDTH, 3),
            'reverse trial requires the exact original initial-frame geometry')
    del loaded
    source = directory / 'source'
    frozen = _reverse_rollout(raw, source)
    directory.mkdir(parents=True, exist_ok=False)
    source.mkdir(exist_ok=False)
    shutil.copyfile(parent / 'initial.npy', source / 'initial.npy')
    np.save(source / 'actions.npy', action_array(REVERSE_CONTROLS), allow_pickle=False)
    (source / 'rollout.yaml').write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding='utf-8')
    copied = _reverse_source_record(original, source, parent, sha256(source / 'rollout.yaml'), sha256(source / 'actions.npy'))
    write_json(source / 'request.json', copied)
    binding = dict(version=1, kind='long_history_reverse_timeline_v1',
        forward_request_path=str(path), forward_request_sha256=sha256(path), forward_source_job=str(parent),
        forward_source_request_sha256=sha256(parent / 'request.json'), forward_rollout_sha256=original['config_sha256'],
        reverse_source_request_sha256=sha256(source / 'request.json'), reverse_rollout_sha256=copied['config_sha256'],
        initial_sha256=original['initial_sha256'], forward_actions_sha256=original['actions_sha256'],
        reverse_actions_sha256=copied['actions_sha256'])
    request = dict(copy.deepcopy(forward), job_id=directory.name, source_job=str(source),
        source_revision=source_revision, experiment_revision='long-history15-reverse-controls-only-v1',
        reverse_timeline=True, reverse_binding=binding)
    validated = validate_reverse_source(request)
    write_json(directory / 'preparation.json', dict(status='cpu_reverse_source_prepared',
        fixed_input_binding=validated, checkpoint_sha256=lineage['sha256'], checkpoint_step=lineage['step'],
        candidate_contract=candidate_contract, inference_settings=probe.SETTINGS, residency='phase', max_seconds=1200,
        weights_or_cache_copied=False, gpu_launched=False, video_generated=False, quality_improvement_verified=False))
    write_json(directory / 'request.json', request)
    return dict(status='cpu_reverse_source_prepared', request=str(directory / 'request.json'),
        source_job=str(source), checkpoint_sha256=lineage['sha256'], checkpoint_step=lineage['step'],
        gpu_launched=False, video_generated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
