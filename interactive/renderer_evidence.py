"""Read-only evidence projection; never certifies running weights or edits sessions."""
import json
from pathlib import Path

from .renderer_capabilities import RendererCapabilities


def renderer_evidence(profile_path, acceptance_path):
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
    gate = RendererCapabilities(profile, profile.get('renderer_id'))
    pack = json.loads(Path(acceptance_path).read_text(encoding='utf-8'))
    if pack.get('schema') != 'renderer-acceptance/v1':
        raise ValueError('Unsupported acceptance package')
    return {
        'source': 'current_read_only_evidence_not_historical_session_result',
        'profile_version': profile['version'],
        'recorded_renderer_configuration': profile['renderer_id'],
        'running_weights_verified': False,
        'execution_gate_enabled_in_this_ui': False,
        'visual_success_verified': False,
        'scope': profile['scope'],
        'assessment_kind': profile['assessment_kind'],
        'stages': {name: gate.check(name) for name in profile['stages']},
        'fixed_acceptance': {
            'reference': str(acceptance_path),
            'source_counts': pack['source_counts'], 'episodes': pack['episodes'],
            'overlaps': pack['overlaps'], 'source_records': pack['source_records'],
            'matched_inputs': {name: {key: row.get(key) for key in
                ('initial', 'prompt_cache', 'prompt_key', 'seed', 'seconds', 'steps',
                 'cfg', 'lora_strength', 'checkpoint')}
                for name, row in pack['matched_inputs'].items()},
            'execution_status': 'reference_only_no_job_requested',
            'limits': pack['limits'],
        },
        'notice': '只读能力证据，不是运行权重认证；不启用模型或新技能，不改变剧情。历史会话仍保留原始记录。',
    }
