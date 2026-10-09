"""Score actual model outputs; a parser baseline is never called Base/SFT."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from interactive.models import WorldState,Plan,StrictModel
from pydantic import create_model
from copy import deepcopy
from interactive.rules import RuleEngine
from interactive.story import StoryGraph

# Keep the measured v1 schema stable: later runtime cross-field guards must not
# retroactively alter historical JSON-validity numbers.
EvaluationPlanV1=create_model('EvaluationPlanV1',__base__=StrictModel,
    **{name:(field.annotation,deepcopy(field)) for name,field in Plan.model_fields.items()})

def parse_plan(text):
    clean=text.strip()
    if clean.startswith('```') and clean.endswith('```'): clean='\n'.join(clean.splitlines()[1:-1])
    return EvaluationPlanV1.model_validate(json.loads(clean))


def score(rows,predictions):
    counts={key:0 for key in ['json_valid','intent_correct','action_correct','target_correct','invalid_action_detection',
        'state_consistency','memory_consistency']}
    invalid=memory=0; details=[]
    for row in rows:
        expected=row['expected_output']; needs_memory=bool(row['memory'].get('recent') or row['memory'].get('important'))
        invalid+=not expected['expected_valid']; memory+=needs_memory
        try:
            plan=parse_plan(predictions.get(row['id'],'')); counts['json_valid']+=1
            counts['intent_correct']+=plan.player_intent==expected['player_intent']
            counts['action_correct']+=plan.semantic_action==expected['semantic_action']
            counts['target_correct']+=plan.target==expected['target']
            if not expected['expected_valid']:
                counts['invalid_action_detection']+=plan.expected_valid is False
            state=WorldState.model_validate(row['world_state']); result=RuleEngine().validate(state,plan)
            def transition(result):
                if not result.valid: return False,state
                try: return True,StoryGraph().transition(state.apply_delta(result.state_delta),result.story_event)
                except ValueError: return False,state
            valid,next_state=transition(result)
            gold_result=RuleEngine().validate(state,Plan.model_validate(expected))
            _,gold_next=transition(gold_result)
            # Also require the actual narrative transition, not just a superficially valid action.
            matched=(plan.semantic_action==expected['semantic_action'] and plan.target==expected['target'])
            counts['state_consistency']+=matched and valid==expected['expected_valid'] and plan.expected_valid==expected['expected_valid'] and next_state==gold_next
            if needs_memory: counts['memory_consistency']+=matched
            details.append({'id':row['id'],'parsed':True,'target_correct':plan.target==expected['target']})
        except (ValueError,TypeError): details.append({'id':row['id'],'parsed':False})
    n=len(rows)
    report={k:{'correct':v,'total':invalid if k=='invalid_action_detection' else memory if k=='memory_consistency' else n}
        for k,v in counts.items()}
    for value in report.values(): value['rate']=value['correct']/value['total'] if value['total'] else None
    return {'metrics':report,'details':details,'protocol':'measured_v1_schema_and_metrics',
        'note':'State consistency requires intended action/target, correct precondition prediction AND identical rule/story next state. Memory metric is action/target accuracy on memory-tagged cases, not causal proof of memory benefit. Intent is exact taxonomy match. Later runtime cross-field guards are not retroactively added to v1 JSON validity.'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--eval',required=True); p.add_argument('--base-predictions',required=True)
    p.add_argument('--sft-predictions',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); rows=[json.loads(l) for l in Path(args.eval).read_text(encoding='utf-8').splitlines() if l.strip()]
    def read(path):
        records=[json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
        if len({r['id'] for r in records})!=len(records): raise ValueError('Duplicate prediction IDs')
        return {r['id']:r['text'] for r in records}
    report={'base':score(rows,read(args.base_predictions)),'sft':score(rows,read(args.sft_predictions))}
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in report.items()}))


if __name__=='__main__': main()
