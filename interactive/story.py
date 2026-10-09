from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


class StoryGraph:
    def __init__(self, path=None):
        self.config=yaml.safe_load(Path(path or ROOT/'configs/story.yaml').read_text(encoding='utf-8'))

    def available_transitions(self, state):
        return self.config['nodes'][state.story.node]['next']

    def transition(self, state, event):
        if not event or event == state.story.node:
            return state
        if event not in self.available_transitions(state):
            raise ValueError(f'当前剧情不能从 {state.story.node} 跳到 {event}')
        return state.apply_delta({'story':{'node':event,'terminal':event in self.config['terminal']}})

    def terminal_check(self, state):
        return state.story.node in self.config['terminal']

    def title(self, state):
        return self.config['nodes'][state.story.node]['title']
