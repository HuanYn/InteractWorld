"""CPU-only checks at the model-output / deterministic-planner boundary."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from training.creator.model_worker import _render_prompt, validate_plan
from training.creator.planner import expand_segments, plan_request


def _proposal(segments, scope='all'):
    return dict(status='ready', explanation='按输入提议动作。', action_segments=segments,
                goals=['人物持续向前移动', '后半段可见镜头向上变化'], edit_scope=scope)


def test_model_proposals_keep_movement_and_original_camera_start_on_second_turn():
    first = _proposal([{'frames': 120, 'keys': ['W']}, {'frames': 120, 'keys': ['W', 'I']}])
    previous = plan_request('一直前进，后半段抬头', proposal=validate_plan(first))
    assert previous['status'] == 'ready'
    second = _proposal([{'frames': 120, 'keys': ['W']}, {'frames': 60, 'keys': ['W', 'I']},
                        {'frames': 60, 'keys': ['W']}], scope='camera')
    edited = plan_request('保留前进，只缩短抬头', previous, validate_plan(second))
    assert edited['status'] == 'ready' and edited['preserved'] is True
    rows = expand_segments(edited['action_segments'])
    assert all('W' in row for row in rows)
    assert [index for index, row in enumerate(rows) if 'I' in row] == list(range(120, 180))


@pytest.mark.parametrize('field,value', [('explanation', 'x' * 2001), ('goals', ['x' * 201]),
                                       ('goals', ['goal'] * 17)])
def test_model_output_bounds_match_downstream_planner(field, value):
    proposal = _proposal([{'frames': 240, 'keys': ['W']}])
    proposal[field] = value
    with pytest.raises(ValueError):
        validate_plan(proposal)


def test_downstream_still_rejects_proposal_that_drops_protected_movement():
    previous = plan_request('一直前进，后半段抬头')
    proposal = _proposal([{'frames': 120, 'keys': []}, {'frames': 60, 'keys': ['I']},
                          {'frames': 60, 'keys': []}], scope='camera')
    original = copy.deepcopy(previous)
    result = plan_request('保留前进，只缩短抬头', previous, validate_plan(proposal))
    assert result['status'] == 'clarify' and previous == original


def _patch(start=8, end=10):
    return dict(status='ready', explanation='将抬头输入限定在第8到10秒。',
                edit_scope='camera', goals=['镜头在指定时段向上俯仰'],
                edits=[dict(op='replace_intervals', key='I',
                            intervals=[dict(start_seconds=start, end_seconds=end)])])


def test_sparse_model_patch_is_compiled_without_rewriting_protected_tracks():
    previous = plan_request('一直前进，后半段抬头')
    proposal = _patch()
    assert validate_plan(proposal) is proposal
    result = plan_request('保持前进不变，只在第8到10秒抬头', previous, proposal)
    assert result['status'] == 'ready' and result['preserved'] is True
    rows = expand_segments(result['action_segments'])
    assert all('W' in row for row in rows)
    assert [i for i, row in enumerate(rows) if 'I' in row] == list(range(128, 160))


@pytest.mark.parametrize('start,end', [(8, 16), (8, 8), (0.1, 10), (True, 10), ('NaN', 10)])
def test_sparse_worker_rejects_invalid_exact_times(start, end):
    with pytest.raises(ValueError):
        validate_plan(_patch(start, end))


def test_sparse_worker_does_not_allow_both_output_representations():
    proposal = _patch()
    proposal['action_segments'] = []
    with pytest.raises(ValueError):
        validate_plan(proposal)


def test_valid_patch_structure_is_not_evidence_of_user_intent_match():
    previous = plan_request('一直前进，后半段抬头')
    proposal = validate_plan(_patch(7, 9))
    assert plan_request('保持前进不变，只在第8到10秒抬头', previous, proposal)['status'] == 'clarify'


def _stub_planner_session(monkeypatch, worker, callback):
    class Session:
        load_count = 0
        def __init__(self, model_path):
            self.path = model_path
        def generate(self, system, content):
            self.load_count = 1  # Explicit test double, not a GPU observation.
            return callback(self.path, system, content)
    monkeypatch.setattr(worker, '_ModelSession', Session)


def test_invalid_user_time_is_clarified_before_model_loading(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    (tmp_path / 'config.json').write_text('{}')
    _stub_planner_session(monkeypatch, worker, lambda *args: pytest.fail('invalid user time must not call model'))
    result = worker.execute_request(tmp_path, dict(kind='plan', text='第0.1到1秒抬头',
                                                  previous_plan=None, capabilities={}), tmp_path)
    assert result['status'] == 'clarify' and result['edits'] == []
    assert result['planning_trace']['attempts'] == [] and result['planning_trace']['model_load_count'] == 0
    assert not (tmp_path / 'model-proposal.json').exists()
    assert json.loads((tmp_path / 'planning-preflight.json').read_text(encoding='utf-8'))['repairable'] is False


def test_numeric_new_plan_uses_patch_only_prompt_and_keeps_all_scope(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    (tmp_path / 'config.json').write_text('{}')
    seen = []
    proposal = dict(status='ready', explanation='先右移，再镜头低头。', goals=['输入右移和镜头低头'], edit_scope='all',
                    edits=[dict(op='replace_intervals', key='D', intervals=[dict(start_seconds=0, end_seconds=4)]),
                           dict(op='replace_intervals', key='K', intervals=[dict(start_seconds=4, end_seconds=6)])])
    _stub_planner_session(monkeypatch, worker, lambda p, s, c: seen.append((s,c)) or proposal)
    result = worker.execute_request(tmp_path, dict(kind='plan', text='前4秒向右移动，然后低头2秒',
                                                  previous_plan=None, capabilities={}), tmp_path)
    assert {k:v for k,v in result.items() if k != 'planning_trace'} == proposal and 'EVERY requested action' in seen[0][0]
    assert result['planning_trace']['outcome'] == 'first_pass' and len(seen) == 1
    assert worker._read_json(seen[0][1])['requested_edit_scope'] == 'all'


def test_malformed_model_json_is_not_repaired_or_promoted_to_success(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    (tmp_path / 'config.json').write_text('{}')
    raw = ' {"status":"ready","edits":[  '
    def malformed(*args):
        return worker.decode_model_response(raw)
    _stub_planner_session(monkeypatch, worker, malformed)
    result = worker.execute_request(tmp_path, dict(kind='plan', text='前4秒向右移动，然后低头2秒',
                                                  previous_plan=None, capabilities={}), tmp_path)
    assert result['status'] == 'clarify' and result['edits'] == [] and result['edit_scope'] == 'all'
    assert (tmp_path / 'model-response.txt').read_text(encoding='utf-8') == raw
    validation = json.loads((tmp_path / 'model-validation.json').read_text(encoding='utf-8'))
    assert validation['error_type'] == 'invalid_model_json' and validation['json_repaired'] is False
    assert (tmp_path / 'model-response-repair.txt').read_text(encoding='utf-8') == raw
    assert result['planning_trace']['outcome'] == 'failed' and result['planning_trace']['revision_count'] == 1


@pytest.mark.parametrize('template,expected', [
    ('{{ messages }}', {}),
    ('{% if enable_thinking %}<think>{% endif %}', {'enable_thinking': False}),
    ({'default': '{% if enable_thinking %}<think>{% endif %}'}, {'enable_thinking': False}),
])
def test_thinking_switch_only_reaches_templates_that_support_it(template, expected):
    seen = {}
    def render(messages, **kwargs):
        seen.update(kwargs)
        return 'rendered'
    processor = SimpleNamespace(chat_template=template, apply_chat_template=render)
    assert _render_prompt(processor, [{'role': 'user', 'content': 'plan'}]) == 'rendered'
    assert seen == dict(tokenize=False, add_generation_prompt=True, **expected)


def test_dynamic_camera_scope_is_prompted_without_repairing_model_output(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    (tmp_path / 'config.json').write_text('{}')
    previous = plan_request('一直前进，后半段抬头')
    seen = []
    _stub_planner_session(monkeypatch, worker, lambda p, s, c: seen.append((s, c)) or _proposal(previous['action_segments']))
    result = worker.execute_request(tmp_path, dict(kind='plan', text='保留前进，只缩短抬头', previous_plan=previous, capabilities={}), tmp_path)
    assert worker._read_json(seen[0][1])['requested_edit_scope'] == 'camera' and 'MUST be "camera"' in seen[0][0]
    assert len(seen) == 2 and result['status'] == 'clarify' and result['planning_trace']['outcome'] == 'failed'
    assert json.loads((tmp_path/'model-proposal.json').read_text(encoding='utf-8'))['edit_scope'] == 'all'


def _judgment(observations, *, verdict='satisfied'):
    return dict(verdict=verdict, evidence=observations,
                decision='accept' if verdict == 'satisfied' else 'revise',
                revision_text='' if verdict == 'satisfied' else '调整镜头输入。',
                confidence_note='Only sparse sampled frames were inspected; not calibrated.')


def _inspection_files(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    current, reference = tmp_path / 'current.mp4', tmp_path / 'reference.mp4'
    # Explicit unit fixtures, not real videos or claims of visual model output.
    current.write_bytes(b'current test fixture')
    reference.write_bytes(b'reference test fixture')
    return current, reference


def test_goal_semantics_separate_camera_movement_relative_and_dense_time():
    from training.creator.model_worker import goal_semantics
    goals = ['镜头向上俯仰', '人物全程向前移动', '镜头比之前更高', '缩短镜头抬头时长', '停止移动']
    semantics = goal_semantics(goals)
    assert [item['categories'] for item in semantics] == [
        ['camera'], ['movement'], ['camera', 'relative'], ['camera', 'relative'], ['movement']]
    assert [item['requires_dense_temporal_evidence'] for item in semantics] == [False, True, False, True, True]


def test_relative_goal_without_reference_skips_model_and_frame_extraction(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    current, _ = _inspection_files(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('missing-reference rule must run before sampling or model inference')
    monkeypatch.setattr(worker, '_generate', forbidden)
    monkeypatch.setattr(worker, 'sample_frames', forbidden)
    result = worker.execute_request(tmp_path, dict(kind='inspect', video_path=str(current),
                                                  goals=['缩短镜头抬头时长']), tmp_path)
    assert (result['verdict'], result['decision'], result['reference_used']) == ('uncertain', 'ask_user', False)
    assert 'reference_required' in result['rule_reasons']
    assert result['original_model_judgment'] is None and not result['rule_downgraded']


def test_reference_inputs_are_labelled_bounded_and_never_include_actions(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    current, reference = _inspection_files(tmp_path)
    def sample(path, directory):
        offset = 0.25 if str(path) == str(reference) else 0.0
        return [(directory / f'{index}.png', index + offset) for index in range(8)], ''
    def generate(model_path, system, text, samples, reference_samples=()):
        content = worker._read_json(text)
        assert len(samples) == len(reference_samples) == 8
        assert content['sample_times_seconds'] == dict(current=list(range(8)), reference=[x + 0.25 for x in range(8)])
        assert 'action_segments' not in text and 'FORBIDDEN_ACTION_EVIDENCE' not in text
        assert 'CURRENT' in system and 'REFERENCE' in system and 'never character head pose' in system
        return _judgment([dict(video='current', time_seconds=0, observation='Horizon is lower within this image.'),
                          dict(video='reference', time_seconds=0.25, observation='Horizon is higher within this image.')])
    monkeypatch.setattr(worker, 'sample_frames', sample)
    monkeypatch.setattr(worker, '_generate', generate)
    result = worker.execute_request(tmp_path, dict(kind='inspect', video_path=str(current), reference_video_path=str(reference),
        goals=['镜头比之前更高'], action_segments='FORBIDDEN_ACTION_EVIDENCE'), tmp_path)
    assert result['reference_used'] is True and result['verdict'] == 'satisfied'
    assert not result['rule_downgraded'] and result['original_model_judgment'] is None
    assert '未校准' in result['confidence_note']


@pytest.mark.parametrize('evidence,reference_times', [
    ([dict(time_seconds=0, observation='Missing source label.')], [0.25]),
    ([dict(video='current', time_seconds=0.25, observation='Wrong video timestamp.')], [0.25]),
    ([dict(video='reference', time_seconds=0, observation='Wrong video timestamp.')], [0.25]),
    ([dict(video='reference', time_seconds=0, observation='No reference was supplied.')], None),
])
def test_comparison_evidence_validates_each_videos_actual_timestamps(evidence, reference_times):
    from training.creator.model_worker import validate_inspection
    with pytest.raises(ValueError, match='video|unsampled'):
        validate_inspection(_judgment(evidence), [0.0], reference_times)


def test_legacy_single_video_evidence_still_validates():
    from training.creator.model_worker import validate_inspection
    judgment = _judgment([dict(time_seconds=0, observation='The horizon is near the image centre.')])
    assert validate_inspection(judgment, [0.0]) == judgment


@pytest.mark.parametrize('goal,observation,reason', [
    ('镜头向上俯仰', '人物头部没有抬起。', 'character_pose_is_not_camera_evidence'),
    ('人物向前移动', '人物持续前进，随后停止移动。', 'sparse_frames_do_not_establish_continuity_or_stopping'),
    ('人物全程向前移动', '人物在图像中央。', 'dense_temporal_evidence_required'),
    ('缩短镜头抬头时长', '天空在当前图像中占比更大。', 'dense_temporal_evidence_required'),
])
def test_sparse_or_character_pose_claims_are_downgraded_with_original_preserved(goal, observation, reason):
    from training.creator.model_worker import apply_observation_rules, goal_semantics
    original = _judgment([dict(video='current', time_seconds=0, observation=observation),
                          dict(video='reference', time_seconds=0, observation='山脊位于图像中部。')], verdict='unsatisfied')
    before = copy.deepcopy(original)
    result = apply_observation_rules(original, goal_semantics([goal]), reference_used=True)
    assert (result['verdict'], result['decision'], result['revision_text']) == ('uncertain', 'ask_user', '')
    assert reason in result['rule_reasons'] and result['rule_downgraded']
    assert result['original_model_judgment'] == before and original == before
    if reason != 'dense_temporal_evidence_required':
        assert all(item['observation'] != observation for item in result['evidence'])


def test_relative_comparison_needs_evidence_from_both_supplied_videos():
    from training.creator.model_worker import apply_observation_rules, goal_semantics
    original = _judgment([dict(video='current', time_seconds=0, observation='山脊位于图像下方。')])
    result = apply_observation_rules(original, goal_semantics(['镜头比之前更高']), reference_used=True)
    assert result['verdict'] == 'uncertain'
    assert 'relative_comparison_needs_both_video_sources' in result['rule_reasons']
    assert result['original_model_judgment'] == original


def test_more_than_eight_frames_per_video_rejected_before_model_import():
    from training.creator.model_worker import _generate
    with pytest.raises(ValueError, match='eight frames per video'):
        _generate(None, '', '', [(None, 0)] * 8, reference_samples=[(None, 0)] * 9)


def test_camera_half_duration_synonyms_remain_relative_not_character_movement():
    from training.creator.model_worker import goal_semantics
    goals = ['镜头俯仰减半', '减少镜头转向', '增加镜头俯仰时间一半']
    for semantics in goal_semantics(goals):
        assert semantics['categories'] == ['camera', 'relative']
        assert semantics['requires_dense_temporal_evidence'] is True


def test_explicit_half_and_half_more_proposals_follow_extension_policy():
    previous = plan_request('一直前进，后半段抬头')
    for instruction, start, duration in [('保留前进，只把抬头减半', 120, 60),
                                         ('保留前进，抬头时间增加一半', 120, 90),
                                         ('保留前进，抬头时间增加一半', 105, 135)]:
        segments = [dict(frames=start, keys=['W']), dict(frames=duration, keys=['W', 'I'])]
        if start + duration < 240:
            segments.append(dict(frames=240-start-duration, keys=['W']))
        proposal = _proposal(segments, scope='camera')
        result = plan_request(instruction, previous, validate_plan(proposal))
        assert result['status'] == 'ready' and result['preserved'] is True
        assert [i for i, row in enumerate(expand_segments(result['action_segments'])) if 'I' in row] == list(range(start, start+duration))
        previous = result


def _install_cpu_model_fixtures(monkeypatch, worker, runtime, *, fail_at=None, responses=None):
    """Instrumented CPU test doubles, never evidence of real model inference."""
    seen = []
    responses = iter(responses or [_proposal([dict(frames=240, keys=['W'])])])
    def observe(stage):
        snapshot = json.loads((runtime / 'model-progress.json').read_text(encoding='utf-8'))
        assert snapshot['status'] == 'running' and snapshot['active_stage'] == stage
        seen.append(stage)
        if fail_at == stage:
            raise RuntimeError('CPU fixture failure during ' + stage)

    class Inputs(dict):
        def to(self, device):
            return self
    class Generated:
        def __getitem__(self, key):
            return self
    class Processor:
        chat_template = '{{ messages }}'
        def apply_chat_template(self, *args, **kwargs):
            return 'test prompt'
        def __call__(self, *args, **kwargs):
            observe('input_prepare')
            return Inputs(input_ids=SimpleNamespace(shape=(1, 1)))
        def batch_decode(self, *args, **kwargs):
            observe('response_decode')
            return [json.dumps(next(responses))]
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            observe('processor_load')
            return cls()
    class Model:
        device = 'cpu-test-fixture'
        def eval(self):
            return self
        def generate(self, **kwargs):
            observe('inference')
            return Generated()
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            observe('model_load')
            return cls()

    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
        is_initialized=lambda: True, max_memory_allocated=lambda: 123, max_memory_reserved=lambda: 456)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, bfloat16='fixture', inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        __version__='5.10.0', AutoProcessor=Processor, Qwen3VLForConditionalGeneration=Model))
    monkeypatch.setattr(worker, '_offline_environment', lambda directory: None)
    return seen


def _worker_cli(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}', encoding='utf-8')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(kind='plan', text='一直前进', capabilities={})), encoding='utf-8')
    output = tmp_path / 'response.json'
    return ['--model', str(model), '--request', str(request), '--output', str(output)]


def test_worker_records_stage_before_work_and_retains_legacy_success_metrics(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    seen = _install_cpu_model_fixtures(monkeypatch, worker, tmp_path)
    assert worker.main(_worker_cli(tmp_path)) == 0
    progress = json.loads((tmp_path / 'model-progress.json').read_text(encoding='utf-8'))
    metrics = json.loads((tmp_path / 'model-metrics.json').read_text(encoding='utf-8'))
    result = json.loads((tmp_path / 'response.json').read_text(encoding='utf-8'))
    assert seen == ['processor_load', 'model_load', 'input_prepare', 'inference', 'response_decode']
    assert result['status'] == 'ready' and progress['status'] == 'completed'
    assert progress['active_stage'] is None and progress['last_stage'] == 'output_write'
    assert all(stage['status'] == 'completed' for stage in progress['stages'])
    assert metrics['elapsed_seconds'] >= 0 and metrics['cuda_initialized'] is True
    assert metrics['peak_allocated_bytes'] == 123 and metrics['peak_reserved_bytes'] == 456
    assert metrics['execution_status'] == 'completed'
    assert {'model_load', 'processor_load', 'inference'} <= metrics['stage_seconds'].keys()
    assert all(seconds >= 0 for seconds in metrics['stage_seconds'].values())
    assert metrics['stage_timing_kind'] == 'host_wall_clock_no_cuda_synchronization'
    assert worker._PROGRESS.get() is None


def test_two_planning_inferences_reuse_exactly_one_model_load(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    bad = _proposal([dict(frames=240, keys=['D'])])
    good = _proposal([dict(frames=240, keys=['W'])])
    seen = _install_cpu_model_fixtures(monkeypatch, worker, tmp_path, responses=[bad, good])
    assert worker.main(_worker_cli(tmp_path)) == 0
    result = json.loads((tmp_path/'response.json').read_text(encoding='utf-8'))
    assert result['planning_trace']['outcome'] == 'repaired'
    assert seen.count('processor_load') == seen.count('model_load') == 1
    assert seen.count('inference') == seen.count('response_decode') == 2
    assert result['planning_trace']['model_load_count'] == 1


@pytest.mark.parametrize('failure_stage', ['model_load', 'inference'])
def test_worker_stage_failure_is_diagnostic_not_success(tmp_path, monkeypatch, failure_stage):
    from training.creator import model_worker as worker
    _install_cpu_model_fixtures(monkeypatch, worker, tmp_path, fail_at=failure_stage)
    assert worker.main(_worker_cli(tmp_path)) == 1
    progress = json.loads((tmp_path / 'model-progress.json').read_text(encoding='utf-8'))
    metrics = json.loads((tmp_path / 'model-metrics.json').read_text(encoding='utf-8'))
    response = json.loads((tmp_path / 'response.json').read_text(encoding='utf-8'))
    assert progress['status'] == 'failed' and progress['last_stage'] == failure_stage
    assert progress['stages'][-1]['status'] == 'failed'
    assert progress['error']['type'] == response['error'] == 'RuntimeError'
    assert metrics['execution_status'] == 'failed' and metrics['last_stage'] == failure_stage
    assert metrics['stage_seconds'][failure_stage] >= 0 and worker._PROGRESS.get() is None


def test_force_killed_cpu_worker_retains_last_stage_without_fabricated_final_metrics(tmp_path):
    # Exercise actual subprocess termination, not a mocked finally block. The
    # child uses no torch, model weights, network, CUDA or execution authority.
    script = '''
import sys, time
from pathlib import Path
from training.creator.model_worker import _WorkerProgress
progress = _WorkerProgress(Path(sys.argv[1]))
progress.mark("model_load")
time.sleep(60)
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        snapshot = None
        while time.monotonic() < deadline:
            progress_path = tmp_path / 'model-progress.json'
            if progress_path.is_file():
                snapshot = json.loads(progress_path.read_text(encoding='utf-8'))
                if snapshot['active_stage'] == 'model_load':
                    break
            if process.poll() is not None:
                pytest.fail('CPU diagnostic child exited before its stage snapshot')
            time.sleep(0.02)
        assert snapshot and snapshot['active_stage'] == 'model_load'
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
    retained = json.loads((tmp_path / 'model-progress.json').read_text(encoding='utf-8'))
    assert retained['active_stage'] == 'model_load' and retained['status'] == 'running'
    assert retained['status_is_last_observation'] is True and retained['pid'] == process.pid
    assert retained['stages'][-1]['started_at'] and retained['observed_at']
    assert not (tmp_path / 'model-metrics.json').exists()
    assert not (tmp_path / 'response.json').exists()


def test_retry_requires_new_directory_to_preserve_killed_worker_diagnostics(tmp_path):
    from training.creator.model_worker import _WorkerProgress
    progress = _WorkerProgress(tmp_path)
    progress.mark('model_load')
    before = progress.path.read_bytes()
    with pytest.raises(ValueError, match='existing worker progress'):
        _WorkerProgress(tmp_path)
    assert progress.path.read_bytes() == before


def test_offline_setup_creates_kernel_cache_directories_without_importing_models(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    environment = dict(worker.os.environ)
    environment['PYTORCH_KERNEL_CACHE_PATH'] = '/unused/external/cache'
    monkeypatch.setattr(worker.os, 'environ', environment)
    monkeypatch.setattr(worker.tempfile, 'tempdir', None)
    worker._offline_environment(tmp_path)
    for name, leaf in [('CUDA_CACHE_PATH', 'cuda'), ('TRITON_CACHE_DIR', 'triton'),
                       ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('PYTORCH_KERNEL_CACHE_PATH', 'kernels')]:
        assert Path(environment[name]) == tmp_path / 'cache' / leaf
        assert Path(environment[name]).is_dir()
    assert environment['HF_HUB_OFFLINE'] == '1' and environment['WANDB_DISABLED'] == 'true'
