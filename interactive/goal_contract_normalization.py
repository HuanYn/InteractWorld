"""Opt-in typed-tool compilation; never repair actions or mutate world state.

The model can retain a valid approach-then-look sequence but put it in the
observation-only completion mode. This narrowly compiles the approach's existing
rule effect into its completion check without changing a single tool step.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tasks import ToolGoalDraft


def normalize_npc_observation_contract(draft: 'ToolGoalDraft'):
    """Return (draft, receipt or None), leaving every unmatched input untouched."""
    # Local import avoids a module cycle: tasks owns the typed schema and invokes
    # this helper only after validating the original language-model draft.
    from .tasks import TOOL_IDS, ToolGoalDraft

    if (not isinstance(draft, ToolGoalDraft) or draft.status != 'ready'
            or draft.completion_mode != 'action_execution' or draft.success_checks):
        return draft, None

    approaches = []
    for index, step in enumerate(draft.steps):
        call = TOOL_IDS.get(getattr(step, 'tool_id', None))
        if call in (('approach', 'anna', None), ('approach', 'leo', None)):
            approaches.append((index, call[1]))
        elif (call is None or call[0] not in ('observe', 'inspect', 'wait')
              or call[1] not in ('left', 'right', 'room', 'desk', 'door_a')
              or call[2] is not None):
            return draft, None

    if len(approaches) != 1:
        return draft, None
    later_observation = any(
        TOOL_IDS[step.tool_id][0] in ('observe', 'inspect')
        for step in draft.steps[approaches[0][0] + 1:])
    if not later_observation:
        return draft, None

    original = draft.model_dump(mode='json')
    check = 'met_' + approaches[0][1]
    normalized = draft.model_copy(deep=True, update={
        'completion_mode': 'rule_state', 'success_checks': [check]})
    receipt = {
        'version': 'npc_observation_contract_v1',
        'source': 'typed_tool_contract_compilation_not_model_repair',
        'applied': True,
        'reason': 'Exactly one existing NPC approach followed by supported observation '
                  'tools requires the existing contact rule check, not observation-only completion.',
        'original_draft': original,
        'normalized_draft': normalized.model_dump(mode='json'),
        'steps_unchanged': original['steps'] == normalized.model_dump(mode='json')['steps'],
        'added_or_removed_actions': False,
        'model_autonomous_repair': False,
        'world_state_changed': False,
        'visual_success_verified': False,
    }
    return normalized, receipt
