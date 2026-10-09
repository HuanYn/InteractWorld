"""Curated seed export or bounded provider-driven Self-Instruct expansion."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from interactive.dataset import curated_seeds,validate_row,normalized
from interactive.agent import OpenAIProvider


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--provider',choices=['seeds','openai'],default='seeds')
    p.add_argument('--target',type=int,default=1000); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=19); p.add_argument('--max-seconds',type=int,default=900)
    args=p.parse_args(); out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new output directory; existing datasets are preserved')
    out.mkdir(parents=True,exist_ok=True)
    seeds=curated_seeds(); groups=[r['seed_group'] for r in seeds]; random.Random(args.seed).shuffle(groups)
    heldout=set(groups[:8]); rows=list(seeds); seen={normalized(r['player_input']) for r in rows}; rejected=0
    provider=OpenAIProvider() if args.provider=='openai' else None
    began=time.monotonic(); calls=0
    raw=(out/'provider-outputs.jsonl').open('x',encoding='utf-8') if provider else None
    def export():
        for split in ('train','eval'):
            selected=[r for r in rows if (r['seed_group'] in heldout)==(split=='eval')]
            with (out/f'{split}.jsonl').open('w',encoding='utf-8') as stream:
                for row in selected: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    if provider:
        for i in range(120):
            if len(rows)>=args.target or time.monotonic()-began>=args.max_seconds: break
            source=seeds[i%len(seeds)]
            count=min(args.batch_size,args.target-len(rows))
            prompt='Return ONLY JSON {"paraphrases":["..."]} with '+str(count)+' DISTINCT short natural Chinese paraphrases of this player instruction. Preserve exactly its action, target, negation, order and constraints. Do not answer or add new actions. Use different concise conversational wording; no numbered variants. If the instruction uses an ambiguous pronoun, keep it ambiguous. Variation batch '+str(i)+'\nInstruction: '+source['player_input']
            try:
                answer=provider.complete([{'role':'user','content':prompt}],max_tokens=1800); calls+=1
                raw.write(json.dumps({'batch':i,'seed_group':source['seed_group'],'raw':answer},ensure_ascii=False)+'\n'); raw.flush()
                clean=answer.strip()
                if clean.startswith('```'): clean=clean.split('\n',1)[1].rsplit('```',1)[0].strip()
                paraphrases=json.loads(clean)['paraphrases']
                if not isinstance(paraphrases,list): raise ValueError('Not a list')
                for j,text in enumerate(paraphrases[:count]):
                    if not isinstance(text,str) or not 2<=len(text)<=150: rejected+=1; continue
                    key=normalized(text)
                    if key in seen: rejected+=1; continue
                    row=dict(source,id=f'variation-{i:03d}-{j:02d}',player_input=text,source='local_llm_paraphrase_rule_checked')
                    validate_row(row); rows.append(row); seen.add(key)
            except Exception as exc:
                rejected+=1
                print(json.dumps({'batch':i,'error':str(exc)}),flush=True)
                if rejected>=20 and len(rows)==len(seeds):
                    break
            export()
            print(json.dumps({'batch':i,'accepted':len(rows),'rejected':rejected,'elapsed_seconds':time.monotonic()-began}),flush=True)
        raw.close()
    export()
    report={'provider':args.provider,'curated_seeds':len(seeds),'accepted':len(rows),'rejected':rejected,
        'train':sum(r['seed_group'] not in heldout for r in rows),'eval':sum(r['seed_group'] in heldout for r in rows),
        'split_policy':'seed_group_disjoint_before_expansion','seed':args.seed,
        'target_met':len(rows)>=args.target,'provider_calls':calls,'elapsed_seconds':time.monotonic()-began,
        'limitations':'Rule consistency is not independent semantic quality verification. Paraphrase-only expansion does not add new world states. Seeds-only export is not 1000-example Self-Instruct.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))


if __name__=='__main__': main()
