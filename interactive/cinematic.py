"""Authored branching-film playback. Never label prepared footage as live generation."""
import json
from pathlib import Path
from .renderers import WorldRenderer,RenderResult


class CinematicRenderer(WorldRenderer):
    kind='cinematic_prebuilt'

    def __init__(self,manifest):
        self.path=Path(manifest).resolve()
        self.library=json.loads(self.path.read_text(encoding='utf-8'))
        if self.library.get('ready') is not True: raise RuntimeError('影游素材尚未完成，未启用。')
        self.last_asset='scene'
        self.played_events=set()

    def snapshot(self):
        return self.last_asset,set(self.played_events)

    def restore(self,snapshot):
        self.last_asset,events=snapshot
        self.played_events=set(events)

    def select(self,state,plan):
        if plan is None: return 'scene'
        action,target=plan.semantic_action,plan.target
        if action in ('observe','wait'):
            return {'door_open':'door_open_hold','key_pickup':'key_held'}.get(self.last_asset,self.last_asset)
        if state.player.location=='core_area': return 'core'
        if state.player.location=='storage_room': return 'storage'
        if target=='door_a':
            if state.objects['door_a'].state!='open': return 'door_closed'
            return 'door_open_hold' if 'door_open' in self.played_events else 'door_open'
        if target in ('red_key','desk') and action in ('search','pick_up','inspect'):
            if 'red_key' in state.inventory.items:
                return 'key_pickup' if action=='pick_up' and 'key_pickup' not in self.played_events else 'key_held'
            if state.story.flags.get('found_key'): return 'key_reveal'
            return 'desk'
        if target=='terminal':
            if state.objects['terminal'].state=='unlocked': return 'terminal_unlocked'
            return 'terminal_logs' if state.story.flags.get('log_retrieved') else 'terminal_locked'
        if target=='leo' and state.story.flags.get('met_leo'): return 'leo'
        if target=='anna': return 'anna_talk' if action in ('talk','ask','interact') else 'anna'
        # Preserve the visual location for observations, rather than replaying a closed door/lost key.
        return self.last_asset

    def generate(self,init_frame,camera_skills,world_prompt,world_state,output_dir):
        return self.generate_plan(init_frame,camera_skills,world_prompt,world_state,output_dir,None)

    def generate_plan(self,init_frame,camera_skills,world_prompt,world_state,output_dir,plan):
        if plan is None: self.played_events.clear()
        key=self.select(world_state,plan)
        entry=self.library['clips'][key]
        video=Path(entry['video']); frame=Path(entry['last_frame'])
        if not video.is_file() or not frame.is_file(): raise RuntimeError(f'缺少影游事件素材：{key}')
        metadata={'kind':self.kind,'asset_id':key,'pre_generated':True,'generated_for_current_request':False,
            'transition':'authored_cinematic_cut','camera_skills_requested':[str(getattr(s,'value',s)) for s in camera_skills],
            'camera_skills_executed_live':False,'asset_provenance':entry.get('provenance'),
            'authored_state_insert':entry.get('authored_state_insert',False),
            'visual_goal_verified':False,'requested_semantic_action':plan.model_dump(mode='json') if plan else None}
        if isinstance(entry.get('asset_review'),dict):
            metadata['asset_review']=entry['asset_review']
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
        (out/'playback.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        self.last_asset=key
        self.played_events.add(key)
        return RenderResult(str(video),str(frame),metadata)
