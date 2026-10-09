"""Goal evaluation by held-out rule execution, not visual or real-world success."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from interactive.tasks import ToolGoalDraft,GoalTask,check_goal
from interactive.models import WorldState
from interactive.rules import RuleEngine
from interactive.story import StoryGraph


def score(rows,predictions):
    counts=dict(json_valid=0,status_correct=0,goal_contract_correct=0,task_correct=0,multistep_correct=0)
    details=[]; multi=0
    for row in rows:
        expected=ToolGoalDraft.model_validate(row['expected_output']).compile()
        is_multi=expected.status=='ready' and len(expected.steps)>=2; multi+=is_multi
        detail={'id':row['id'],'correct':False}
        try:
            raw=predictions[row['id']].strip()
            if raw.startswith('```') and raw.endswith('```'): raw='\n'.join(raw.splitlines()[1:-1])
            actual=ToolGoalDraft.model_validate_json(raw).compile(); counts['json_valid']+=1
            matching_status=actual.status==expected.status; counts['status_correct']+=matching_status
            matching_contract=(matching_status and actual.completion_mode==expected.completion_mode and
                set(actual.success_checks)==set(expected.success_checks))
            counts['goal_contract_correct']+=matching_contract
            correct=matching_contract
            if correct and actual.status=='ready':
                state=WorldState.model_validate(row['world_state']); rules=RuleEngine(); graph=StoryGraph()
                GoalTask(row['player_input'],actual).preflight(SimpleNamespace(state=state,rules=rules,graph=graph))
                # Don't count an unwanted detour through another route as a valid equivalent plan.
                desired_route={'key':'B','door':'B','password':'A'}.get(row['scenario'])
                for step in actual.steps:
                    result=rules.validate(state,step.plan)
                    state=graph.transition(state.apply_delta(result.state_delta),result.story_event)
                    if desired_route and state.story.route not in (None,desired_route): correct=False
                if actual.completion_mode=='action_execution':
                    correct=all(s.plan.semantic_action=='inspect' and s.plan.target=='left' for s in actual.steps)
                else: correct=correct and check_goal(state,expected.success_checks)
            detail['correct']=bool(correct)
            counts['task_correct']+=bool(correct); counts['multistep_correct']+=bool(correct and is_multi)
        except (ValueError,KeyError,TypeError) as exc: detail['error']=str(exc)[:300]
        details.append(detail)
    metrics={name:{'correct':count,'total':multi if name=='multistep_correct' else len(rows)} for name,count in counts.items()}
    for value in metrics.values(): value['rate']=value['correct']/value['total'] if value['total'] else None
    return {'metrics':metrics,'details':details,'protocol':'goal_v1_rule_execution','visual_success_measured':False}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--eval',required=True); p.add_argument('--base',required=True)
    p.add_argument('--sft',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    rows=[json.loads(l) for l in Path(a.eval).read_text(encoding='utf-8').splitlines() if l]
    def load(path):
        records=[json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l]
        if len({r['id'] for r in records})!=len(records): raise ValueError('Duplicate IDs')
        if {r['id'] for r in records}!={r['id'] for r in rows}: raise ValueError('Incomplete evaluation')
        return {r['id']:r['text'] for r in records}
    result={'base':score(rows,load(a.base)),'sft':score(rows,load(a.sft)),
        'limitations':'Single authored world, small state-family dev split. Not live execution or visual success.'}
    with Path(a.output).open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
