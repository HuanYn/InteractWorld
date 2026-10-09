"""Expose the actual engine affordances; never infer actions from player keywords."""
from .models import Plan
from .rules import RuleEngine
from .story import StoryGraph

TOOL_CALLS=[
    ('observe','room',None),('wait','room',None),('explore','laboratory',None),
    ('inspect','left',None),('inspect','right',None),('inspect','desk',None),('inspect','door_a',None),
    ('approach','anna',None),('talk','anna',None),('interact','anna','help'),
    ('talk','anna','trust'),('talk','anna','distrust'),('ask','anna','password'),
    ('avoid','anna',None),('search','desk',None),('pick_up','red_key',None),
    ('open','door_a',None),('inspect','terminal',None),('use','terminal',None),
    ('move_to','core_area',None),('move_to','storage_room',None),('approach','leo',None),('ask','leo','logs')]

def available_tools(state):
    rules=RuleEngine(); graph=StoryGraph(); tools=[]
    for action,target,dialogue in TOOL_CALLS:
        plan=Plan(player_intent='tool_contract',semantic_action=action,target=target,dialogue_action=dialogue)
        result=rules.validate(state,plan)
        if not result.valid: continue
        try: graph.transition(state.apply_delta(result.state_delta),result.story_event)
        except ValueError: continue
        tools.append({'semantic_action':action,'target':target,'dialogue_action':dialogue,
            'effect':result.reason,'story_event':result.story_event})
    return tools
