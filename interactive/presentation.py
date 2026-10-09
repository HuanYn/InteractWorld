"""Small, read-only presentation helpers. No model loading or state changes."""
import os
from datetime import datetime,timezone
from html import escape
import requests
from .tools import available_tools


def service_status(mode,get=None):
    from .closed_handoff import closed_reason
    if closed_reason():
        return False,closed_reason()
    get=get or requests.get
    checked=datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    if mode=='mock':
        return True,'离线规则 / Mock 模式（不是 LLM 或真实生成）。'
    base=os.environ.get('IW_LLM_BASE_URL','').rstrip('/')
    model=os.environ.get('IW_LLM_MODEL','')
    headers={}
    if os.environ.get('IW_LLM_API_KEY'): headers['Authorization']='Bearer '+os.environ['IW_LLM_API_KEY']
    try:
        if not base or not model: raise ValueError('not configured')
        response=get(base+'/models',headers=headers,timeout=(2,3))
        if getattr(response,'status_code',200)==404 and base.endswith('/v1'):
            # Compatibility with the existing local adapter's read-only health API.
            response=get(base[:-3]+'/health',headers=headers,timeout=(2,3)); response.raise_for_status()
            if response.json().get('status')!='ready': raise ValueError('provider not ready')
        else:
            response.raise_for_status()
            models=response.json().get('data',[])
            if not any(item.get('id')==model for item in models): raise ValueError('model not served')
        expected_adapter=os.environ.get('IW_GOAL_ADAPTER')
        if expected_adapter:
            health=get(base.removesuffix('/v1')+'/health',headers=headers,timeout=(2,3))
            health.raise_for_status()
            if health.json().get('adapter')!=expected_adapter:
                return False,f'目标适配器版本不匹配；新交互暂停，未借用旧服务。检查：{checked}'
        if mode=='wan':
            response=get(os.environ.get('IW_WAN_BASE_URL','http://127.0.0.1:8878').rstrip('/')+'/health',timeout=(2,3))
            response.raise_for_status()
            if response.json().get('busy'):
                return False,f'视频服务正忙，请稍后刷新；未提交新任务。检查：{checked}'
    except (requests.RequestException,ValueError,TypeError,AttributeError):
        return False,f'模型服务未就绪：新交互暂停，已保存回放仍可观看。服务恢复后点“刷新服务状态”。检查：{checked}'
    visual='预生成分镜播放，不是即时视频生成' if mode=='cinematic' else '异步视频生成' if mode=='wan' else 'Mock 视觉'
    return True,f'模型接口可达 · {visual}。仅连通性检查，不代表推理或画质已通过。检查：{checked}'


LABELS={('approach','anna',None):'走近 Anna',('interact','anna','help'):'帮助 Anna',
    ('talk','anna','distrust'):'表达怀疑并独自调查',('ask','anna','password'):'询问密码',
    ('search','desk',None):'搜索桌子',('pick_up','red_key',None):'拿起红钥匙',
    ('open','door_a',None):'打开储藏室门',('inspect','terminal',None):'检查终端 / 读取日志',
    ('move_to','core_area',None):'进入核心区域',('move_to','storage_room',None):'进入储藏室',
    ('approach','leo',None):'找到 Leo',('ask','leo','logs'):'询问 Leo 安全日志'}


def next_actions(state,story_profile='legacy'):
    if state.story.terminal: return '本路线已结束，可重新开始体验其他分支。'
    if story_profile=='spatial_v1':
        from .spatial_skills import spatial_tool_context
        spatial_labels={**LABELS,('approach','terminal',None):'走近终端',
            ('approach','door_a',None):'走到原门前',
            ('use','door_a',None):'用红钥匙解锁原门',
            ('open','door_a',None):'打开原门',
            ('use','terminal',None):'输入密码 / 授权原门'}
        labels=[spatial_labels[key] for item in spatial_tool_context(state)
            if item['allowed_now'] and (key:=tuple(item['call'])) in spatial_labels]
    else:
        calls=available_tools(state)
        labels=[LABELS[key] for t in calls
            if (key:=(t['semantic_action'],t['target'],t['dialogue_action'])) in LABELS]
    return '当前规则允许（不是模型建议，也不保证画面完成）：'+('；'.join(labels) or '观察当前场景')+'。可用自己的话表达。'


def turn_summary(row):
    if not row: return '开始后输入一句行动，查看理解 → 校验 → 呈现 → 观察的结果。'
    plan=row['plan']; renderer=row.get('renderer',{})
    visual=row.get('visual_assessment',{})
    verdict={'supported':'VLM 认为可能符合（未验证）','contradicted':'VLM 认为不符合',
        'uncertain':'画面证据不足','not_visually_testable':'无法仅凭画面判断'}.get(visual.get('status'),'未验证')
    if row.get('planner_fallback'): outcome='模型失败，本次未执行；保持原画面'
    elif row.get('renderer_status')=='failed': outcome='呈现失败，剧情未提交'
    elif not row['valid']: outcome='请求未执行（规则拦截或待澄清）'
    elif row.get('renderer_status')=='dialogue_only': outcome='剧情对话已更新，保留当前画面；本轮未生成新视频'
    elif renderer.get('pre_generated'): outcome='播放预生成分镜，非本次生成'
    else: outcome='已完成本轮呈现（不等于动作画面成功）'
    safe=lambda value: escape(str(value)).replace('\n',' ')
    return (f'**本轮理解：** {safe(plan["semantic_action"])} → {safe(plan["target"])}  \n'
        f'**执行结果：** {outcome}  \n**画面观察：** {verdict}  \n'
        f'**处理耗时：** {row.get("latency_seconds",0):.1f} 秒（非单独生成耗时）')


def task_summary(task):
    if task is None: return ''
    labels={'ready':'待执行','running':'执行中','paused':'已暂停','needs_clarification':'需要澄清',
        'completed_rule_goal':'规则目标完成（视觉未认证）','completed_action_goal':'观察工具已执行（视角未认证）'}
    text=f'\n\n### 目标任务 · {labels.get(task.status,task.status)}\n\n{escape(task.goal)}\n\n'
    if getattr(task,'planning_source',None)=='authored_skill_fallback':
        text+='计划来源：**规则技能兜底（不是模型自主规划成功）**。原模型失败记录保留。\n\n'
    for i,step in enumerate(task.proposal.steps):
        marker='✓' if i<task.index else '→' if i==task.index else '·'
        call=f'{step.plan.semantic_action} → {step.plan.target}'
        if step.plan.dialogue_action: call+=' / '+step.plan.dialogue_action
        text+=f'{marker} {i+1}. {escape(step.instruction)}（{escape(call)}）  \n'
    return (text+'\n完成条件：'+(', '.join(task.proposal.success_checks) or '观察工具执行，不代表画面认证')+
            f'  \n自动重规划：{task.replans}/2 · 执行尝试：{task.attempts}/12  \n'+escape(task.message))
