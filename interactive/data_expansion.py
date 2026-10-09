"""State-grouped authored corpus. No generated text is treated as executed evidence."""
from collections import Counter
from copy import deepcopy
import json
from .goal_data import (build_goal_completion_references, build_goal_references,
                        _state_family, score_goal_predictions)
from .models import WorldState
from .dataset import normalized, validate_row

# Independently authored phrasings. Evaluation wording is never expanded into train.
PHRASES = {
 'key': ['我不信任 Anna，帮我找到钥匙，但不要开门，也不要进入储藏室。',
         '我要独立调查，把红钥匙拿到手，门先别打开，也别进去。',
         '选择不信任 Anna 的路线，拿到红钥匙就停；禁止开门和进入。',
         '不走终端路线，也不帮助 Anna：独自调查拿钥匙，不开门、不进入。',
         '沿独自调查路线取得钥匙即可，开门和进去都暂时不做。',
         '继续独自找线索，红钥匙进背包后就结束，门保持原状，不进入。',
         '本次只以独立调查取得红钥匙为目标，允许必要前置步骤，但不许开门或进入。',
         '走不信任 Anna 的分支，最终拿到钥匙，别额外开门进去。'],
 'door': ['独自调查，找到钥匙后打开储藏室门；不要进去。',
          '走独立调查路线，把门打开就停，必要时先拿钥匙，不进入。',
          '不相信 Anna，想办法拿钥匙打开门，但留在门外。',
          '完成独立调查分支的取钥匙和开门，暂不进入储藏室。'],
 'ending_key': ['选择独自调查，找到钥匙开门，再进入储藏室。',
                '不信任 Anna，沿钥匙路线调查到底，最后进入储藏室。',
                '帮我独立调查拿钥匙、开门并进去，补齐必要前置动作。',
                '走钥匙分支直到进入储藏室，不要改走其他路线。'],
 'password': ['帮助 Anna 并取得密码；不要操作终端，也不要进入核心区域。',
              '走帮助 Anna 的路线，问到密码就停，不操作终端也不进入核心区域。',
              '我愿意协助 Anna，先取得终端密码，终端和核心区域暂时别动。',
              '与 Anna 合作取得密码即可，不操作终端、不进入核心区域。'],
 'ending_anna': ['选择相信并帮助 Anna，取得密码、解锁终端并进入核心区域。',
                 '走帮助 Anna 的路线一直到核心区域，完成必要前置动作。',
                 '相信 Anna，协助她拿到密码，再通过终端进入核心区域。',
                 '完成 Anna 合作分支，最终抵达核心区域。'],
 'public_logs': ['选择终端调查路线，读取公开日志；不要去找 Leo。',
                 '走终端分支读出公开日志就停，暂时别接近 Leo。',
                 '只沿终端路线恢复公开日志，不去见 Leo。',
                 '完成终端路线的公开日志调查，先不找 Leo。'],
 'ending_terminal': ['选择终端路线，恢复日志、找到 Leo 并向他核实记录。',
                     '走终端分支调查到底，读出公开日志后找 Leo 核实。',
                     '通过终端的公开日志追查，再向 Leo 确认记录。',
                     '完成终端调查路线，最终与 Leo 核实安全日志。'],
 'look': ['别继续之前的计划，只看看左边。','取消剩余任务，镜头转向左侧观察。',
          '刚才的目标先作废，现在只向左看。','停下旧计划，仅检查左边，不推进剧情。'],
 'right': ['别继续之前的计划，只看看右边。','取消剩余任务，镜头转向右侧观察。',
           '刚才的目标先作废，现在只向右看。','停下旧计划，仅检查右边，不推进剧情。'],
 'wait': ['暂停推进剧情，只在这里等待。','先别继续任务，原地等一下。',
          '取消后续动作，现在待在原地等待。','暂时不行动，只等待。'],
 'unsupported': ['走到桥边，再回头看山。','离开实验室，开车去海滩。',
                 '飞到月球上找钥匙。','传送到城堡，再骑马过河。'],
 'direct_open': ['只打开门，不允许搜索、拿钥匙、对话或做其他动作。',
                 '现在仅执行开门，禁止任何前置动作。',
                 '不要找钥匙，不要交谈，也不要做其他事，只开门。',
                 '只能开门，条件不满足就说明原因，不做其他动作。'],
 'direct_password': ['只向 Anna 问密码，不允许接近她、帮助她或做其他动作。',
                     '只询问 Anna 的密码，禁止接近或帮助她以及任何其他步骤。',
                     '不要补前置动作，仅向 Anna 问密码。',
                     '只能问 Anna 密码，条件不足就说明，不做别的。'],
}


def build_expanded():
    seeds, _ = build_goal_completion_references()
    old, _ = build_goal_references()
    # Protect the previous held-out states, including semantic-equivalent variants.
    test = {_state_family(WorldState.model_validate(r['world_state'])) for r in old if r['split']=='test'}
    dev = {_state_family(WorldState.model_validate(r['world_state'])) for r in old if r['split']=='dev'} - test
    families = sorted({_state_family(WorldState.model_validate(r['world_state'])) for r in seeds})
    groups = {k: ('test' if k in test else 'dev' if k in dev else 'train') for k in families}
    for split in ('dev','test'):
        if split not in groups.values():
            groups[next(k for k in reversed(families) if groups[k]=='train')] = split
    originals = list(seeds)
    for seed in originals:
        if seed['scenario']=='look':
            right=deepcopy(seed); right['scenario']='right'; right['allowed_tools']=['inspect:right']
            right['expected_output']['steps'][0]['tool_id']='inspect:right'; seeds.append(right)
    rows=[]; seen=set()
    for seed in seeds:
        family=_state_family(WorldState.model_validate(seed['world_state']))
        split=groups[family]; name=seed['scenario'].removesuffix('-repair')
        # Goal/action semantics unchanged; held-out uses reserved last two phrasings.
        phrases=PHRASES[name][:-2] if split=='train' else PHRASES[name][-2:]
        for variant,text in enumerate(phrases):
            for context in ('clean','history','feedback'):
                row=deepcopy(seed)
                row.update(player_input=text,split=split,seed_group='state-'+str(families.index(family)),
                           scenario=name,context_kind=context,annotation_source='authored_rule_verified',
                           human_review='not_performed',ai_review='reference_contract_checked',
                           visual_ground_truth=False,model_success=False)
                # Existing repair rows would otherwise duplicate the feedback augmentation.
                if seed['scenario'].endswith('-repair') and context!='feedback': continue
                if context=='clean': row['feedback']=None
                if context=='history':
                    row['feedback']=None
                    row['world_state']['player']['goal']='之前的目标：检查终端并查看公开日志。'
                    row['memory']={'recent':[{'player_text':'先检查终端。','status':'cancelled_unexecuted',
                                            'note':'历史请求已取消，未执行，不是当前指令。'}], 'important':[]}
                if context=='feedback':
                    if row['expected_output']['status']!='ready': continue
                    if name.startswith('direct_'): continue
                    row['feedback']={'failure':'先前计划没有满足当前目标，请结合最新状态重新规划剩余步骤。',
                        'latest_turn':None,'committed_steps':[],
                        'required_success_checks':row['expected_output']['success_checks'],
                        'required_completion_mode':row['expected_output']['completion_mode'],
                        'remaining_only':True,'source':'authored_failure_context_not_actual_rollout'}
                for step in row['expected_output']['steps']: step['instruction']='执行 '+step['tool_id']
                # Exact context-aware dedup: not just id or paraphrase count.
                signature=json.dumps([family,normalized(text),row.get('memory'),row.get('feedback')],sort_keys=True,ensure_ascii=False)
                if signature in seen: continue
                seen.add(signature); row['id']=f'expanded-{len(rows):05d}'
                validate_row(row); rows.append(row)
    # All references must pass the same strict rule/constraint scorer before use.
    scored=score_goal_predictions(rows,{r['id']:json.dumps(r['expected_output']) for r in rows})
    bad=[d for d in scored['details'] if not d['joint_success']]
    if bad: raise ValueError('Reference scorer rejected: '+str(bad[:3]))
    summary={'records':len(rows),'state_families':len(families),
             'splits':dict(Counter(r['split'] for r in rows)),
             'family_splits':dict(Counter(groups.values())),
             'scenario_counts':dict(Counter(r['scenario'] for r in rows)),
             'context_counts':dict(Counter(r['context_kind'] for r in rows)),
             'multistep':dict(Counter(r['split'] for r in rows if len(r['expected_output']['steps'])>1)),
             'unique_wordings':len({r['player_input'] for r in rows}),
             'source':'Authored paraphrases + reachable rule states + authored history/repair contexts',
             'llm_generated':False,'real_rollout':False,'human_review':False,'visual_ground_truth':False,
             'limitations':['One authored world, not independent real-world tasks.',
                            'No automatically generated reasoning or visual truth.',
                            'State families and held-out wording disjoint; old test families excluded from training.']}
    return rows,summary


def select_eval(rows,split,limit=48):
    pool=[r for r in rows if r['split']==split]; selected=[]; counts=Counter()
    # Round-robin distinct scenario/status/context strata, then balanced families.
    while pool and len(selected)<limit:
        row=min(pool,key=lambda r:(counts[(r['scenario'],r['expected_output']['status'])],
                   counts[('context',r['context_kind'])],counts[('family',r['seed_group'])],r['id']))
        pool.remove(row); selected.append(row)
        counts[(row['scenario'],row['expected_output']['status'])]+=1
        counts[('context',row['context_kind'])]+=1; counts[('family',row['seed_group'])]+=1
    return selected
