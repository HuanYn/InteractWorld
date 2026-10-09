"""Private fixed-pulse experiment, not a new public Agent camera skill."""
import json
from pathlib import Path
import numpy as np
from .lab_video import LaboratoryVideoModel


def pitch_pulse(key):
    if key not in ('I','K'): raise ValueError('Only the two preregistered pitch keys')
    actions=np.zeros((96,8),dtype=np.float32)
    actions[:24,['W','A','S','D','I','J','K','L'].index(key)]=1
    return actions


def validate_pulse(actions):
    value=np.asarray(actions)
    if value.shape!=(96,8) or not np.isfinite(value).all() or not np.isin(value,[0,1]).all():
        raise ValueError('Expected finite binary 96x8 actions')
    for key in ('I','K'):
        if np.array_equal(value,pitch_pulse(key)): return key
    raise ValueError('Only 24-frame isolated I or K followed by 72 static frames')


class PitchDiagnosticModel(LaboratoryVideoModel):
    diagnostic_actions=None

    def _action_schedule(self,skills,frames):
        if frames!=96 or list(skills)!=['static']: raise ValueError('Fixed six-second diagnostic only')
        key=validate_pulse(self.diagnostic_actions)
        return np.asarray(self.diagnostic_actions,dtype=np.float32).copy(),[
            {'frames':24,'keys':[key]},{'frames':72,'keys':[]}]

    def generate(self,*args,**kwargs):
        key=validate_pulse(self.diagnostic_actions)
        result=super().generate(*args,**kwargs)
        result['metadata'].update(action_source='isolated_preregistered_pitch_diagnostic',
            diagnostic_key=key,public_agent_skills_changed=False,visual_goal_verified=False)
        Path(result['video_path']).with_name('render.json').write_text(
            json.dumps(result['metadata'],ensure_ascii=False,indent=2),encoding='utf-8')
        return result
