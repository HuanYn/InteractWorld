from __future__ import annotations
from enum import Enum
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class CameraSkill(str, Enum):
    forward = 'forward'
    backward = 'backward'
    truck_left = 'truck_left'
    truck_right = 'truck_right'
    pan_left = 'pan_left'
    pan_right = 'pan_right'
    static = 'static'


Action = Literal['explore','approach','move_to','inspect','observe','talk','ask','use',
                 'pick_up','open','search','wait','follow','avoid','interact']
Dialogue = Literal['help','trust','distrust','password','logs','greet']


class Plan(StrictModel):
    player_intent: str = Field(min_length=1, max_length=200)
    target: str = Field(default='room', max_length=60)
    semantic_action: Action = 'observe'
    camera_skills: list[CameraSkill] = Field(default_factory=lambda:[CameraSkill.static], max_length=3)
    dialogue_action: Dialogue | None = Field(default=None,description='Only talk/ask/interact may use dialogue actions; approach may only greet. Other movement/inspection uses null.')
    story_event: str | None = Field(default=None, max_length=80)
    reason_summary: str = Field(default='', max_length=400)
    confidence: float = Field(default=0.5, ge=0, le=1)
    needs_clarification: bool = False
    clarification: str = Field(default='', max_length=300)
    expected_valid: bool | None = None

    @model_validator(mode='after')
    def compatible_dialogue(self):
        if self.dialogue_action is not None and self.semantic_action not in ('talk','ask','interact'):
            if not (self.semantic_action=='approach' and self.dialogue_action=='greet'):
                raise ValueError('Movement/inspection must have dialogue_action=null; helping is a separate interact/anna/help call.')
        return self


class PlayerState(StrictModel):
    location: str = 'entrance'
    goal: str = 'discover_what_happened'
    # Rule proximity only; this is not a measured camera/player pose.
    interaction_target: str | None = Field(default=None, max_length=60)


class ObjectState(StrictModel):
    state: str
    visible: bool = True
    location: str = 'laboratory'


class CharacterState(StrictModel):
    location: str
    trust: float = Field(default=0.0, ge=0, le=1)
    fear: float = Field(default=0.2, ge=0, le=1)
    visible: bool = False


class StoryState(StrictModel):
    chapter: int = Field(default=1, ge=1)
    node: str = 'enter_lab'
    route: Literal['A','B','C'] | None = None
    flags: dict[str, bool] = Field(default_factory=dict)
    terminal: bool = False


class EnvironmentState(StrictModel):
    visual_scene: str = 'unknown'
    visual_evidence: dict = Field(default_factory=dict)
    observation_source: str = 'none'


class Inventory(StrictModel):
    items: list[Literal['red_key']] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique(self):
        if len(self.items) != len(set(self.items)):
            raise ValueError('duplicate inventory item')
        return self


class WorldState(StrictModel):
    scene: Literal['abandoned_laboratory'] = 'abandoned_laboratory'
    player: PlayerState = Field(default_factory=PlayerState)
    inventory: Inventory = Field(default_factory=Inventory)
    objects: dict[str, ObjectState] = Field(default_factory=lambda: {
        'door_a':ObjectState(state='locked'), 'red_key':ObjectState(state='hidden',visible=False,location='desk'),
        'terminal':ObjectState(state='locked'), 'desk':ObjectState(state='normal')})
    characters: dict[str, CharacterState] = Field(default_factory=lambda: {
        'anna':CharacterState(location='laboratory',trust=0.2,fear=0.7),
        'leo':CharacterState(location='storage_room')})
    story: StoryState = Field(default_factory=StoryState)
    environment: EnvironmentState = Field(default_factory=EnvironmentState)

    def serialize(self):
        return self.model_dump_json(indent=2)

    @classmethod
    def deserialize(cls, text):
        return cls.model_validate_json(text)

    def apply_delta(self, delta):
        """Called only with rule/reconciler-owned deltas, never provider JSON."""
        data=self.model_dump()
        def merge(dst, src):
            for key,value in src.items():
                if key not in dst and dst is data:
                    raise ValueError('unknown state field')
                if isinstance(value,dict) and isinstance(dst.get(key),dict):
                    merge(dst[key],value)
                else:
                    dst[key]=value
        merge(data,delta)
        return WorldState.model_validate(data)


class Observation(StrictModel):
    scene: str = 'unknown'
    visible_objects: list[str] = Field(default_factory=list)
    visible_characters: list[str] = Field(default_factory=list)
    major_event: str | None = None
    confidence: float = Field(default=0,ge=0,le=1)
    source: Literal['mock','vlm','unavailable'] = 'unavailable'
    evidence: str = ''


class RuleResult(StrictModel):
    valid: bool
    reason: str
    state_delta: dict = Field(default_factory=dict)
    story_event: str | None = None
    npc_response: str = ''
    fallback_suggestion: str = ''


def parse_plan(text):
    value=text.strip()
    if value.startswith('```') and value.endswith('```'):
        value='\n'.join(value.splitlines()[1:-1])
    return Plan.model_validate(json.loads(value))
