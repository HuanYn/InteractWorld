from collections import deque


class Memory:
    def __init__(self):
        self.short=deque(maxlen=5)
        self.long=[]

    def update(self, turn, state):
        self.short.append({k:turn.get(k) for k in ('player_text','plan','npc_response','story_event','valid','visual_assessment')})
        if turn['valid']:
            tags=[k for k,v in state.story.flags.items() if v]
            seen={row['tag'] for row in self.long}
            for tag in tags:
                if tag not in seen:
                    self.long.append({'tag':tag,'importance':1,'turn':turn['turn_id']})
        return self.retrieve()

    def retrieve(self):
        # Include only committed facts; no vector database required.
        return {'recent':list(self.short),'important':self.long[-30:]}

    def last_target(self):
        for row in reversed(self.short):
            target=(row.get('plan') or {}).get('target')
            if row.get('valid') and target not in (None,'room','left','right'):
                return target
        return None
