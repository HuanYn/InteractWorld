"""Versioned, conservative renderer admission. Evidence labels are not truth.

Only explicitly selected sessions use this gate; the preserved demo is unchanged.
The caller binds the profile to a renderer configuration identity, not its display
name. Admitting a stage never bypasses per-clip observation or commits story state.
"""
from copy import deepcopy
import json
from pathlib import Path

DIMENSIONS = ('direction', 'structure', 'target_visibility')
EVIDENCE_STATES = {'supported', 'partial', 'failed', 'unverified'}


class RendererCapabilities:
    def __init__(self, profile, renderer_id):
        self._profile = deepcopy(profile)
        if profile.get('schema') != 'renderer-capabilities/v1':
            raise ValueError('Unsupported capability schema')
        if not renderer_id or profile.get('renderer_id') != renderer_id:
            raise ValueError('Capability profile does not match renderer configuration')
        if not profile.get('version') or not isinstance(profile.get('stages'), dict):
            raise ValueError('Capability version and stages are required')
        for row in profile['stages'].values():
            if row.get('interface') not in ('supported', 'diagnostic_only', 'unsupported'):
                raise ValueError('Invalid interface capability')
            if any(row.get(k) not in EVIDENCE_STATES for k in DIMENSIONS):
                raise ValueError('Missing or invalid visual capability dimension')
            if not isinstance(row.get('evidence'), list) or not row['evidence']:
                raise ValueError('Capability evidence references are required')

    @classmethod
    def load(cls, path, renderer_id):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), renderer_id)

    def snapshot(self):
        return deepcopy(self._profile)

    def check(self, stage):
        row = self._profile['stages'].get(stage)
        status, reason = 'unsupported', 'This stage has no registered renderer evidence.'
        if row is not None:
            if row['interface'] != 'supported':
                reason = 'Interface is unavailable or restricted to isolated diagnostics.'
            elif row['direction'] != 'supported' or row['structure'] != 'supported':
                reason = 'Joint direction and structure qualification is incomplete.'
            elif row['target_visibility'] != 'supported':
                status, reason = 'needs_observation', 'Target visibility is not qualified.'
            else:
                status, reason = 'admitted', 'Per-clip prerequisites and observation are still required.'
        return dict(status=status, admitted=status == 'admitted', reason=reason,
                    stage=stage, profile_version=self._profile['version'],
                    renderer_id=self._profile['renderer_id'], capability=deepcopy(row),
                    visual_success_verified=False, rule_effects_committed=False)
