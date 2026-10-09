"""Narrative truth is rule-owned, not inferred from generated pixels."""
from pathlib import Path
import yaml
from .models import Plan, WorldState, RuleResult


class RuleEngine:
    def __init__(self):
        self.profiles=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/characters.yaml').read_text(encoding='utf-8'))

    def validate(self, state:WorldState, plan:Plan):
        def ok(reason,delta=None,event=None,npc=''):
            return RuleResult(valid=True,reason=reason,state_delta=delta or {},story_event=event,npc_response=npc)
        def no(reason,suggestion='观察房间，或询问下一步可做什么。'):
            return RuleResult(valid=False,reason=reason,fallback_suggestion=suggestion)
        a,t,d=plan.semantic_action,plan.target,plan.dialogue_action
        flags=state.story.flags
        if plan.needs_clarification:
            return no(plan.clarification or '目标不明确，请说明对象。')
        if state.story.terminal:
            return no('本路线已结束，可以重置体验另一条路线。')
        if t not in {'room','laboratory','entrance','storage_room','core_area','left','right',*state.objects,*state.characters}:
            return no('当前实验室没有这个可交互目标。')
        if a in ('observe','wait'):
            return ok('观察当前环境；没有修改关键剧情状态。')
        if a in ('explore','move_to') and t in ('room','laboratory','entrance'):
            event='hear_noise' if state.story.node=='enter_lab' else None
            return ok('进入实验室，听见一阵异响。',{'player':{'location':'laboratory'},'story':{'flags':{'heard_noise':True}}},event)
        if a=='avoid' and t=='anna':
            if state.story.route not in (None,'C'):
                return no('当前已经选择其他路线；重置后可忽略 Anna。')
            return ok('暂时忽略 Anna，准备检查终端。',{'story':{'route':'C','flags':{'ignored_anna':True}}})
        if t=='anna' and a in ('approach','talk','ask','interact','follow'):
            if not flags.get('met_anna'):
                if d not in (None,'greet'):
                    return no('尚未见到 Anna，请先接近她。','先输入：走近 Anna。')
                return ok('你见到了紧张的研究员 Anna。',{'characters':{'anna':{'visible':True}},
                    'story':{'flags':{'met_anna':True}}},'meet_anna','Anna：你是谁？这里发生了事故。')
            if d in ('help','trust'):
                if state.story.route not in (None,'A'):
                    return no('已经选择其他路线，不能直接改写剧情分支。')
                if flags.get('helped_anna'):
                    return ok('Anna 记得你的帮助，重复承诺不会无限增加信任。',npc='Anna：谢谢，我记得。')
                trust=round(min(1,state.characters['anna'].trust+0.4),2)
                return ok('你帮助 Anna 稳定情绪，她开始信任你。',{'characters':{'anna':{'trust':trust,'fear':0.3}},
                    'story':{'route':'A','flags':{'helped_anna':True}}},'trust_anna','Anna：谢谢你愿意帮我。')
            if d=='distrust':
                if state.story.route not in (None,'B'):
                    return no('当前路线已确定；重置后可以独自调查。')
                return ok('你决定独自寻找线索。',{'story':{'route':'B','flags':{'distrusted_anna':True}}},'distrust_anna','Anna：好吧，你可以检查那张桌子。')
            if d=='password':
                if state.characters['anna'].trust < self.profiles['anna']['password_trust_threshold']:
                    return no('Anna 信任不足，不会透露密码。','先帮助 Anna。')
                if state.story.route!='A':
                    return no('当前不在 Anna 路线。')
                event='get_password' if not flags.get('password_known') else None
                return ok('Anna 告诉了你终端密码。',{'story':{'flags':{'password_known':True}}},event,'Anna：终端密码是 AURORA。请小心使用。')
            return ok('Anna 根据你们的关系回应。',npc=('Anna：我们一起查清事故。' if flags.get('helped_anna') else 'Anna：我还不能完全相信你。'))
        if a=='search' and t in ('room','desk','red_key'):
            if state.story.route!='B':
                return no('先选择独自调查路线。','先遇见 Anna，再说：我不相信你。')
            if flags.get('found_key'):
                return ok('你已经发现红钥匙。')
            return ok('在桌子的抽屉里发现了红钥匙。',{'objects':{'red_key':{'visible':True,'state':'available'}},
                'story':{'flags':{'found_key':True}}},'find_red_key')
        if a=='pick_up' and t=='red_key':
            if 'red_key' in state.inventory.items:
                return ok('钥匙已经在背包中。')
            if not state.objects['red_key'].visible or state.objects['red_key'].state!='available':
                return no('还没有发现可拾取的钥匙。','搜索桌子。')
            return ok('红钥匙放入背包。',{'inventory':{'items':['red_key']},'objects':{'red_key':{'state':'picked_up','visible':False}},'story':{'flags':{'key_picked_up':True}}})
        if a in ('open','use') and t=='door_a':
            if 'red_key' not in state.inventory.items:
                return no('门已上锁，你没有红钥匙。','找到并拾取红钥匙。')
            if state.story.route!='B':
                return no('当前不在钥匙路线。')
            if state.objects['door_a'].state=='open':
                return ok('门已经打开。')
            return ok('使用红钥匙打开门。',{'objects':{'door_a':{'state':'open'}},'story':{'flags':{'door_opened':True}}},'open_door')
        if a in ('inspect','use','interact') and t=='terminal':
            if state.story.route=='A':
                if not flags.get('password_known'):
                    return no('终端需要密码。','询问 Anna 密码。')
                return ok('密码验证通过，核心区域已解锁。',{'objects':{'terminal':{'state':'unlocked'}},
                    'story':{'flags':{'terminal_accessed':True}}},'core_area')
            if state.story.route=='B':
                return no('钥匙路线请继续调查桌子与门。')
            if state.story.node=='inspect_terminal':
                return ok('恢复了一份公开事故日志，得知 Leo 在储藏室。',{'story':{'flags':{'log_retrieved':True}}},'retrieve_log')
            if flags.get('log_retrieved'):
                return ok('公开日志已经恢复；核心终端仍未解锁。')
            return ok('终端锁定，但可恢复公开日志；再次检查可读取。',{'story':{'route':'C'}},'inspect_terminal')
        if a in ('move_to','approach','follow') and t=='core_area':
            if state.story.route!='A' or not flags.get('terminal_accessed'):
                return no('核心区域尚未解锁。')
            return ok('与 Anna 一起进入核心区域。',{'player':{'location':'core_area'}},'ending_anna')
        if a in ('move_to','approach','open') and t=='storage_room':
            if state.story.route=='B' and flags.get('door_opened'):
                return ok('你穿过打开的门，找到事故留下的线索。',{'player':{'location':'storage_room'}},'ending_key')
            return no('无法进入储藏室。','钥匙路线先开门；日志路线可呼叫 Leo。')
        if a in ('approach','talk','ask','interact') and t=='leo':
            if not flags.get('log_retrieved'):
                return no('还不知道 Leo 的位置。','先恢复终端公开日志。')
            if not flags.get('met_leo'):
                return ok('依据日志找到 Leo。',{'characters':{'leo':{'visible':True}},'story':{'flags':{'met_leo':True}}},'meet_leo','Leo：你看过事故日志了？')
            return ok('Leo 帮助你核对了事故记录。',{'characters':{'leo':{'trust':0.4}}},'ending_terminal','Leo：我们现在有足够证据了。')
        if a=='inspect' and t in ('left','right','desk','room','door_a'):
            return ok('检查目标，关键剧情状态未改变。')
        return no('当前动作无法执行，或缺少前置条件。')
