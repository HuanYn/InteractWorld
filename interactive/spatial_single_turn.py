"""Opt-in one-action contract; no legacy training/evaluation prompt mutation."""
import json
from .models import Plan
from .spatial_interactions import SpatialStoryGraph,SPATIAL_PROFILE_ID
from .spatial_skills import spatial_tool_context


def spatial_planner_messages(text,state,observation,memory,*,story_observation_context=None):
    prompt=(
        'Plan ONE next action in this authored spatial laboratory profile. Return ONLY a Plan JSON. '
        'You propose, never modify world state. Follow CURRENT_PLAYER_REQUEST, not quoted history. '
        'Physical proximity is an AUTHORED RULE recorded by player.interaction_target, not visual truth. '
        'approach:terminal is required before inspect/use/interact terminal; use:terminal on A enters '
        'the known password and authorizes the original door but does not open or enter it. '
        'approach:door_a is required before unlocking/opening it or moving through it. '
        'B needs red_key in inventory, then use:door_a to unlock, open:door_a, move_to:storage_room. '
        'A needs help Anna and her password, then approach terminal/use terminal, approach door/open door, '
        'move_to:core_area. C needs approach terminal, inspect terminal twice to retrieve logs and an '
        'explicitly AUTHORED log-passage permission, approach door/open door, move_to storage_room, '
        'then approach Leo and ask Leo logs. Never meet Leo remotely or before entering storage_room. '
        'Do not switch established routes; null is no route selected. Anna is outside and Leo inside. '
        'If a specifically requested action is blocked, preserve that action/target and expected_valid=false; '
        'do not silently perform a different prerequisite. For an authorized broader goal, propose its '
        'next needed action only. Observe/inspect directions do not invent proximity or inventory. '
        'Do not claim rule position, a fallible VLM report or a prepared cinematic cut proves camera movement. '
        'Use a short reason_summary, no chain of thought. No invented targets, voice or capabilities. '
        'Schema: '+json.dumps(Plan.model_json_schema()))
    context={'profile_id':SPATIAL_PROFILE_ID,'world_state':state.model_dump(mode='json'),
        'observation':observation.model_dump(),'memory':memory,
        'available_story_transitions':SpatialStoryGraph().available_transitions(state),
        'current_tool_preconditions':spatial_tool_context(state)}
    # Only an Engine-issued, bounded display capability may describe the new
    # story-only post-ending inspection. It is not a proposed plan, a VLM fact,
    # or permission to mutate the ended world. Legacy/default prompts stay exact.
    scoped=story_observation_context
    if (isinstance(scoped,dict) and state.story.terminal
            and state.player.location=='storage_room'
            and scoped.get('version')=='declared_storage_observation_v1'
            and scoped.get('source')=='engine_opt_in_capability_not_visual_truth'
            and scoped.get('permitted_calls')==[['inspect','right',None]]
            and scoped.get('state_effects_allowed') is False
            and type(scoped.get('remaining_attempts')) is int
            and 1<=scoped['remaining_attempts']<=2):
        context['nonstory_viewing_capability']=dict(scoped)
        for row in context['current_tool_preconditions']:
            if row['call']==['inspect','right',None]:
                row.update(allowed_now=True,story_event=None,
                    rule_reason='Engine opt-in: inspect the current room to the right only; no story/state effects.',
                    capability_source='engine_opt_in_not_visual_truth')
        prompt+=(' The engine supplied nonstory_viewing_capability is a separate display-only '
                 'exception after the narrative ending. Only inspect:right is available within '
                 'its remaining_attempts. If requested, propose that action using the ordinary '
                 'Plan schema; do not treat terminal=true as blocking that one display action. '
                 'Terminal, location, inventory and story remain unchanged. Do not invent '
                 'other movement, objects, left inspection or navigation success.')
    return [{'role':'system','content':prompt},{'role':'user','content':
        json.dumps(context,ensure_ascii=False)+'\nCURRENT_PLAYER_REQUEST:\n'+json.dumps(text,ensure_ascii=False)}]
