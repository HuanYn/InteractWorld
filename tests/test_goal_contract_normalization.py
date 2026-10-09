"""CPU contract fixtures, not model outputs or visual-success evidence."""
import json

import pytest

from interactive.agent import LLMAgent
from interactive.engine import InteractionEngine
from interactive.goal_contract_normalization import normalize_npc_observation_contract
from interactive.tasks import ToolGoalDraft, ToolGoalStep, propose_goal


def draft_for(*tools, **updates):
    return ToolGoalDraft.model_validate({
        'status': 'ready', 'completion_mode': 'action_execution', 'success_checks': [],
        'steps': [{'instruction': f'Keep exact step {i}: {tool}', 'tool_id': tool}
                  for i, tool in enumerate(tools)], **updates})


def test_anna_sequence_is_preserved_exactly():
    draft = draft_for('approach:anna', 'inspect:left', 'wait:room', 'inspect:desk')
    before = draft.model_dump(mode='json')
    normalized, receipt = normalize_npc_observation_contract(draft)
    assert draft.model_dump(mode='json') == before
    assert normalized.steps == draft.steps
    assert normalized.steps is not draft.steps
    assert normalized.completion_mode == 'rule_state'
    assert normalized.success_checks == ['met_anna']
    assert normalized.compile().steps[0].plan.semantic_action == 'approach'
    assert receipt['original_draft'] == before and receipt['steps_unchanged']
    assert not receipt['model_autonomous_repair'] and not receipt['world_state_changed']


def test_leo_with_supported_prefix_keeps_all_steps():
    draft = draft_for('observe:room', 'approach:leo', 'inspect:right')
    normalized, receipt = normalize_npc_observation_contract(draft)
    assert normalized.steps == draft.steps and normalized.success_checks == ['met_leo']
    assert receipt['normalized_draft'] == normalized.model_dump(mode='json')


def test_requires_exactly_one_approach_and_a_later_observation():
    for tools in [('approach:anna',), ('observe:room', 'approach:anna'),
                  ('approach:anna', 'wait:room'),
                  ('approach:anna', 'approach:leo', 'inspect:left')]:
        draft = draft_for(*tools)
        normalized, receipt = normalize_npc_observation_contract(draft)
        assert normalized is draft and receipt is None


def test_complete_route_or_mutating_tool_is_never_normalized():
    for tools in [('approach:anna', 'talk:anna:distrust', 'search:desk',
                   'pick_up:red_key', 'open:door_a', 'move_to:storage_room'),
                  ('approach:anna', 'inspect:terminal'),
                  ('approach:leo', 'ask:leo:logs'),
                  ('approach:anna', 'explore:laboratory')]:
        draft = draft_for(*tools)
        normalized, receipt = normalize_npc_observation_contract(draft)
        assert normalized is draft and receipt is None


def test_illegal_tool_or_untyped_input_is_not_repaired():
    illegal = ToolGoalStep.model_construct(instruction='unsupported', tool_id='inspect:mountain')
    draft = draft_for('approach:anna', 'inspect:left').model_copy(update={'steps': [
        ToolGoalStep(instruction='approach Anna', tool_id='approach:anna'), illegal]})
    normalized, receipt = normalize_npc_observation_contract(draft)
    assert normalized is draft and receipt is None
    raw = draft.model_dump()
    normalized, receipt = normalize_npc_observation_contract(raw)
    assert normalized is raw and receipt is None


def test_clarification_existing_checks_or_rule_mode_is_unchanged():
    candidates = [draft_for(clarification='Unavailable goal', status='needs_clarification'),
                  draft_for('approach:anna', 'inspect:left', success_checks=['met_anna']),
                  draft_for('approach:anna', 'inspect:left', completion_mode='rule_state')]
    for draft in candidates:
        normalized, receipt = normalize_npc_observation_contract(draft)
        assert normalized is draft and receipt is None


def test_observation_only_goal_keeps_original_execution_mode():
    draft = draft_for('inspect:left', 'observe:room', 'wait:room')
    normalized, receipt = normalize_npc_observation_contract(draft)
    assert normalized is draft and receipt is None
    assert normalized.compile().completion_mode == 'action_execution'


class FixedProvider:
    def __init__(self, draft):
        self.raw = draft.model_dump_json()
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return self.raw


class UnusedRenderer:
    def generate(self, *args, **kwargs):
        raise AssertionError('Planning must not call the renderer')


def test_existing_engine_default_keeps_original_schema_repair_limit(tmp_path):
    provider = FixedProvider(draft_for('approach:anna', 'inspect:left'))
    engine = InteractionEngine(tmp_path, agent=LLMAgent(provider), renderer=UnusedRenderer())
    before = engine.get_state()
    with pytest.raises(ValueError, match='action_execution'):
        propose_goal(engine, 'Approach Anna, then look left')
    assert provider.calls == 2  # Same single JSON-repair allowance as before.
    assert engine.get_state() == before and not engine.history
    assert all('goal_contract_normalization' not in call for call in engine.goal_planner_calls)


def test_explicit_opt_in_compiles_once_and_records_original_and_normalized(tmp_path):
    original = draft_for('approach:anna', 'inspect:left')
    provider = FixedProvider(original)
    engine = InteractionEngine(tmp_path, agent=LLMAgent(provider), renderer=UnusedRenderer())
    engine.goal_contract_normalization = True
    before = engine.get_state()
    proposal = propose_goal(engine, 'Approach Anna, then look left')
    assert provider.calls == 1
    assert [step.plan.semantic_action for step in proposal.steps] == ['approach', 'inspect']
    assert proposal.completion_mode == 'rule_state' and proposal.success_checks == ['met_anna']
    call = engine.goal_planner_calls[0]
    assert call['raw_response'] == provider.raw
    receipt = call['goal_contract_normalization']
    assert receipt['original_draft'] == json.loads(provider.raw)
    assert receipt['normalized_draft']['steps'] == receipt['original_draft']['steps']
    assert receipt['reason'] and receipt['source'].endswith('not_model_repair')
    assert engine.get_state() == before and not engine.history and engine.task is None
