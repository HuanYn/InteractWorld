from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import requests


@dataclass
class RenderResult:
    video_path: str | None
    last_frame: str | None
    metadata: dict = field(default_factory=dict)


class WorldRenderer:
    def generate(self, init_frame, camera_skills, world_prompt, world_state, output_dir):
        raise NotImplementedError


class LaboratoryWanRenderer(WorldRenderer):
    """Real new lab conditioning, short-window endpoint continuation, no cache replay."""
    kind='wan_laboratory'
    def __init__(self,base_url,initial_image,seed=42,seconds=3,cfg=3.0,lora_strength=1.0,npc_framing=False,live_current_frame=False,
                 live_prompt_key=None,live_full_window_pan=False,story_lineage_guard=False):
        if seconds not in (3,6): raise ValueError('片段时长只支持3秒或6秒。')
        if type(npc_framing) is not bool or (npc_framing and seconds!=6):
            raise ValueError('Optional NPC framing requires an explicit bool and six-second rendering')
        self.base=base_url.rstrip('/'); self.initial_image=initial_image; self.seed=seed
        self.seconds=seconds; self.cfg=cfg; self.lora_strength=lora_strength; self.counter=0
        self.npc_framing=npc_framing
        if type(live_current_frame) is not bool or (live_current_frame and npc_framing):
            raise ValueError('Live current-frame mode requires explicit bool and forbids NPC framing')
        if live_current_frame and (not isinstance(live_prompt_key,str) or not live_prompt_key.strip()
                                   or live_prompt_key!=live_prompt_key.strip()):
            raise ValueError('Live mode requires an explicit live_prompt_key; scene cache fallback is forbidden')
        if type(live_full_window_pan) is not bool or (live_full_window_pan and (not live_current_frame or seconds!=3)):
            raise ValueError('Full-window live pan requires explicit bool, live mode and seconds=3')
        self.live_current_frame=live_current_frame
        if type(story_lineage_guard) is not bool or (story_lineage_guard and (live_current_frame or npc_framing)):
            raise ValueError('Story lineage is an independent opt-in and forbids live-camera/goal-framing mixing')
        self.story_lineage_guard=story_lineage_guard
        self._story_binding=None; self._story_session_id=None
        self._story_input_initialized=False; self._story_has_generated=False; self._story_request_active=False
        self._declared_room_context=None
        self._validated_rule_effect=None
        self.live_prompt_key=live_prompt_key
        self.live_full_window_pan=live_full_window_pan
        self._live_binding=None; self._live_session_id=None; self._live_has_generated=False
        self._live_input_initialized=False
        self._live_request_active=False

    def reset_live_session(self,session_id):
        from .live_continuation import reset_live_session
        return reset_live_session(self,session_id)

    def bind_current_frame(self,session_id,frame,*,initial=False,session_dir=None):
        from .live_continuation import bind_current_frame
        return bind_current_frame(self,session_id,frame,initial=initial,session_dir=session_dir)

    def establish_live_input(self,session_id,owned_frame,session_dir):
        from .live_continuation import establish_live_input
        return establish_live_input(self,session_id,owned_frame,session_dir)

    def reset_story_session(self,session_id):
        from .story_lineage import reset_story_session
        return reset_story_session(self,session_id)

    def establish_story_input(self,session_id,owned_frame,session_dir):
        from .story_lineage import establish_story_input
        return establish_story_input(self,session_id,owned_frame,session_dir)

    def bind_story_frame(self,session_id,frame,*,session_dir=None):
        from .story_lineage import bind_story_frame
        return bind_story_frame(self,session_id,frame,session_dir=session_dir)

    def snapshot(self):
        """Keep seed position for one explicit retry; never cache generated media."""
        return {'counter':self.counter}

    def restore(self,snapshot):
        counter=snapshot.get('counter') if isinstance(snapshot,dict) else None
        if type(counter) is not int or counter<0:
            raise ValueError('Invalid laboratory renderer seed snapshot')
        self.counter=counter

    @staticmethod
    def prompt_for(plan):
        if plan is None: return 'scene'
        if plan.target=='anna' and plan.semantic_action in ('talk','ask','interact'): return 'anna_talk'
        if plan.target=='red_key': return 'key_pickup' if plan.semantic_action=='pick_up' else 'key_reveal'
        if plan.target=='desk': return 'key_reveal' if plan.semantic_action=='search' else 'desk'
        if plan.target=='door_a': return 'door_open' if plan.semantic_action=='open' else 'scene'
        return {'anna':'anna','leo':'leo','terminal':'terminal','storage_room':'storage','core_area':'core'}.get(plan.target,'scene')

    def effective_prompt_for(self,plan):
        context=self._declared_room_context
        if context is not None:
            if (not self.story_lineage_guard or self.live_current_frame or self.npc_framing
                    or context.get('session_id')!=self._story_session_id
                    or context.get('room')!='storage_room'):
                raise ValueError('Invalid declared room prompt context')
            if plan is not None and plan.semantic_action in ('inspect','observe','wait') and plan.target in ('left','right','room','laboratory'):
                if plan.semantic_action=='inspect' and plan.target=='right':return 'interior_right'
                raise ValueError('Declared storage currently supports only exact right observation; no old lab-text fallback')
        if self._validated_rule_effect=='unlock_key_door':
            return 'door_unlock'
        return self.prompt_for(plan)

    def generate_rule_plan(self,init_frame,camera_skills,world_prompt,world_state,output_dir,plan,rule_result):
        """Select key-unlock text only from Engine's accepted rule result.

        Never infer it from Plan.story_event or door use alone. Engine holds its
        interaction lock; clear the temporary binding on success and failure.
        """
        from .models import RuleResult
        if not isinstance(rule_result,RuleResult) or not rule_result.valid:
            raise ValueError('Rendering requires an accepted engine rule result')
        effect=rule_result.story_event
        if effect=='unlock_key_door':
            delta=rule_result.state_delta
            if (self.live_current_frame or self._declared_room_context is not None
                    or plan.semantic_action!='use' or plan.target!='door_a'
                    or delta.get('objects',{}).get('door_a',{}).get('state')!='unlocked'
                    or delta.get('story',{}).get('flags',{}).get('door_unlocked_by_key') is not True):
                raise ValueError('Key-unlock rendering requires the exact validated unlock rule effect')
        if self._validated_rule_effect is not None:
            raise ValueError('A validated render effect is already bound')
        self._validated_rule_effect=effect if effect=='unlock_key_door' else None
        try:
            result=self.generate_plan(init_frame,camera_skills,world_prompt,world_state,output_dir,plan)
            if self._validated_rule_effect is not None:
                result.metadata['validated_rule_rendering']={
                    'source':'engine_validated_rule_result','rule_effect':effect,
                    'prompt_key':'door_unlock','model_story_event_used':False,
                    'visual_success_verified':False}
            return result
        finally:
            self._validated_rule_effect=None

    def generate(self,init_frame,camera_skills,world_prompt,world_state,output_dir):
        return self.generate_plan(init_frame,camera_skills,world_prompt,world_state,output_dir,None)

    def generate_plan(self,init_frame,camera_skills,world_prompt,world_state,output_dir,plan):
        requested_skills=[getattr(s,'value',s) for s in camera_skills]
        live_evidence=None
        if self.live_current_frame:
            from .live_continuation import compile_live_skills,request_evidence
            requested_skills=compile_live_skills(plan,requested_skills,full_window_pan=self.live_full_window_pan)
            live_evidence=request_evidence(self,init_frame,output_dir,plan,requested_skills)
        elif self.story_lineage_guard:
            from .story_lineage import request_evidence
            live_evidence=request_evidence(self,init_frame,output_dir,plan,requested_skills)
        payload={'initial':str(init_frame or self.initial_image),
                 'prompt_key':self.live_prompt_key if self.live_current_frame else self.effective_prompt_for(plan),
                 'skills':requested_skills,'output_dir':str(output_dir),
                 'seed':self.seed+self.counter,'seconds':self.seconds,'steps':40,'cfg':self.cfg,'lora_strength':self.lora_strength}
        framing=bool(self.npc_framing and plan is not None and plan.semantic_action=='approach'
                     and plan.target in ('anna','leo'))
        if framing:
            payload.update(skills=['static'],framing_condition='current_frame_zoom_1p08')
        self.counter+=1
        if self.live_current_frame:
            # Every submission needs a fresh engine-lock binding. Do not auto
            # retry a timed-out POST or silently reuse another request's frame.
            self._live_binding=None
            self._live_request_active=True
        elif self.story_lineage_guard:
            self._story_binding=None
            self._story_request_active=True
        try:
            return self._submit(payload,init_frame,plan,requested_skills,framing,live_evidence)
        finally:
            if self.live_current_frame:self._live_request_active=False
            if self.story_lineage_guard:self._story_request_active=False

    def _submit(self,payload,init_frame,plan,requested_skills,framing,live_evidence):
        # Wait only before submission; never blindly repeat a timed-out generation.
        deadline=time.monotonic()+360
        while time.monotonic()<deadline:
            health=requests.get(self.base+'/health',timeout=5); health.raise_for_status()
            status=health.json()
            if framing and payload['framing_condition'] not in status.get('conditioning_extensions',[]):
                raise RuntimeError('当前视频服务未加载候选取景接口；未提交本次生成。')
            if not status.get('busy'): break
            time.sleep(2)
        else: raise TimeoutError('视频生成队列繁忙，尚未提交此次生成。')
        response=requests.post(self.base+'/generate',json=payload,timeout=(10,900))
        if not response.ok: raise RuntimeError(response.text[:800])
        result=response.json()
        if self.live_current_frame:
            from .live_continuation import verify_result
            verify_result(result,live_evidence,plan)
        elif self.story_lineage_guard:
            from .story_lineage import verify_result
            verify_result(result,live_evidence,plan)
        if framing:
            result['metadata'].update(control_compilation={
                'requested_semantic_action':plan.semantic_action,'requested_target':plan.target,
                'requested_skills':requested_skills,'effective_skills':['static'],
                'effective_effect':'2D closer framing of the current generated image',
                'framing_condition':payload['framing_condition'],
                'controlled_entity':'image_framing','3D_navigation_verified':False,
                'semantic_approach_verified':False,'candidate_path':True},
                **{'3D_navigation_verified':False})
        finished=self.finish_render(result,payload,init_frame,plan)
        if self.live_current_frame:self._live_has_generated=True
        if self.story_lineage_guard:self._story_has_generated=True
        return finished

    def recover_initial(self,folder):
        """Attach only an orphaned, completed initial render; never replay a turn."""
        if self.live_current_frame:raise ValueError('Live mode forbids recovered/foreign initial media')
        if self.story_lineage_guard:raise ValueError('Story lineage forbids recovered/foreign initial media')
        folder=Path(folder).resolve(); out=folder/'initial'
        if (folder/'session-export.json').exists() or any(folder.glob('turn*')):
            raise ValueError('Recovery supports only an uncommitted initial render')
        metadata=json.loads((out/'render.json').read_text())
        if (metadata.get('checkpoint')!='R005-step1040' or metadata.get('prompt_key')!='scene' or
            metadata.get('seed')!=self.seed or metadata.get('frames')!=97 or
            metadata.get('sampler',{}).get('cfg')!=self.cfg or
            metadata.get('lora_strength')!=self.lora_strength or self.seconds!=6):
            raise ValueError('Initial render does not match this runtime')
        for name in ('video.mp4','input.png','last.png','actions.npy'):
            if not (out/name).is_file() or (out/name).resolve().parent!=out:
                raise ValueError('Missing or out-of-scope initial artifact')
        import numpy as np
        actions=np.load(out/'actions.npy',allow_pickle=False)
        if actions.shape!=(96,8) or np.any(actions):
            raise ValueError('Initial recovery must be static, not an action turn')
        payload={'initial':metadata['conditioning_image'],'prompt_key':'scene','skills':['static'],
                 'output_dir':str(out),'seed':self.seed,'seconds':6,'steps':40,
                 'cfg':self.cfg,'lora_strength':self.lora_strength}
        metadata['recovered_existing_initial']=True
        result={'metadata':metadata,'video_path':str(out/'video.mp4'),'last_frame':str(out/'last.png')}
        self.counter=1  # Preserve next seed; this completed call must not repeat.
        return self.finish_render(result,payload,None,None)

    def finish_render(self,result,payload,init_frame,plan):
        metadata=result['metadata']; metadata.update(request=payload,
            source_condition='previous_generated_frame' if init_frame else 'external_imagegen_initial_asset',
            requested_semantic_action=plan.model_dump(mode='json') if plan else None)
        if self.story_lineage_guard:
            metadata.update(raw_video_path=result['video_path'],display_video_path=result['video_path'],
                display_policy='raw_model_video_with_UI_prompt_overlay')
            return RenderResult(result['video_path'],result['last_frame'],metadata)
        # Presentation only; the observer always receives the unannotated generated frame.
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        from training.eval.input_header import InputHeaderRenderer
        out=Path(result['video_path']).parent
        actions=np.load(out/'actions.npy',allow_pickle=False)
        if payload.get('framing_condition') and np.any(actions):
            raise ValueError('NPC framing must use the actual all-zero action schedule')
        header=InputHeaderRenderer(initial_frame=np.asarray(Image.open(out/'input.png').convert('RGB')),
            prompt=metadata['prompt'],actions=actions,seed=payload['seed'],fps=16,layout='split_hud')
        display=out/'inputs.mp4'
        with imageio.get_reader(result['video_path']) as reader, imageio.get_writer(display,fps=16,codec='libx264',quality=8,macro_block_size=16) as writer:
            for index,frame in enumerate(reader): writer.append_data(header.render(frame,index))
        metadata['raw_video_path']=result['video_path']; metadata['display_video_path']=str(display)
        metadata['display_policy']='annotated_model_video_with_input_header'
        return RenderResult(str(display),result['last_frame'],metadata)


class MockRenderer(WorldRenderer):
    """A visibly labelled procedural storyboard; not generated Wan footage."""
    kind='mock'
    def generate(self,init_frame,camera_skills,world_prompt,world_state,output_dir):
        from PIL import Image,ImageDraw,ImageFont
        import imageio.v2 as imageio
        import numpy as np
        output=Path(output_dir); output.mkdir(parents=True,exist_ok=False)
        im=Image.new('RGB',(768,432),'#111d2a'); d=ImageDraw.Draw(im)
        d.polygon([(0,0),(235,125),(533,125),(768,0)],fill='#25354a')
        d.polygon([(0,432),(235,300),(533,300),(768,432)],fill='#263d48')
        d.rectangle((235,125,533,300),fill='#344853')
        door_open=world_state.objects['door_a'].state=='open'
        d.rectangle((425,150,480,295),fill='#0c1721' if door_open else '#66717d',outline='#79b7bd',width=3)
        d.rectangle((270,235,363,249),fill='#897453'); d.line((277,249,277,295),fill='#897453',width=5)
        d.rectangle((292,194,338,227),fill='#061c27',outline='#5ee0d0',width=2)
        d.rectangle((268,138,327,151),fill='#9bc3c4')
        if world_state.objects['red_key'].visible:
            d.ellipse((319,266,330,277),outline='#ff5c61',width=3); d.line((330,271,348,271),fill='#ff5c61',width=3)
        for name,x,color in [('anna',380,'#a9bed4'),('leo',230,'#c7b28b')]:
            if world_state.characters[name].visible:
                d.ellipse((x-8,190,x+8,207),fill=color); d.rectangle((x-10,209,x+10,258),fill=color)
                d.text((x-14,175),name.upper(),fill='white')
        d.rectangle((0,0,768,55),fill='#101820')
        d.text((18,12),'MOCK STORYBOARD / NOT WAN / RULE STATE, NOT VISUAL EVIDENCE',fill='#ffd16b')
        d.text((18,34),'ABANDONED LABORATORY | '+world_state.story.node,fill='white')
        d.text((18,385),'SKILLS: '+', '.join(str(getattr(x,'value',x)) for x in camera_skills),fill='#95cde5')
        d.text((18,408),'Inventory: '+str(world_state.inventory.items)+' | Anna trust: '+str(world_state.characters['anna'].trust),fill='white')
        last=output/'last.png'; im.save(last)
        video=output/'storyboard.mp4'
        with imageio.get_writer(video,fps=8,codec='libx264',quality=7,macro_block_size=16) as writer:
            for _ in range(16): writer.append_data(np.asarray(im))
        return RenderResult(str(video),str(last),{'kind':'mock','synthetic_storyboard':True,
            'visual_goal_verified':False,'camera_motion_verified':False})


class WanRenderer(WorldRenderer):
    """Use the existing guarded async HTTP service; never bypass GPU leases.

    Current server supports frozen presets only, not the laboratory world.
    Therefore this adapter refuses unsupported world/image substitution.
    """
    kind='wan_http'
    def __init__(self,base_url,scene_id='action-dev1',seed=42,timeout=900):
        self.base=base_url.rstrip('/'); self.scene_id=scene_id; self.seed=seed; self.timeout=timeout
        self.session=requests.Session()

    def generate(self,init_frame,camera_skills,world_prompt,world_state,output_dir):
        raise RuntimeError('REAL LAB WAN BLOCKED: existing renderer has mountain/grassland presets, not laboratory/NPC conditioning. Use generate_preset for a separately labelled camera-control test.')

    def generate_preset(self,camera_skills,output_dir):
        from .agent import SkillSelector
        config=self.session.get(self.base+'/api/config',timeout=10); config.raise_for_status(); config=config.json()
        if not config.get('generation_enabled'):
            raise RuntimeError('Real generation is disabled by the server GPU authorization guard.')
        if self.scene_id not in [s['scene_id'] for s in config['scenes']]:
            raise ValueError('Unknown frozen scene')
        payload={'scene_id':self.scene_id,'seed':self.seed,'action_segments':SkillSelector().action_segments(camera_skills)}
        response=self.session.post(self.base+'/api/jobs',json=payload,headers={
            'Origin':self.base,'X-InterActWorld-CSRF':config['csrf_token']},timeout=20)
        response.raise_for_status(); job=response.json(); job_id=job['job_id']
        started=time.monotonic()
        while time.monotonic()-started < self.timeout:
            response=self.session.get(self.base+'/api/jobs/'+job_id,timeout=15); response.raise_for_status(); job=response.json()
            if job['status']=='failed': raise RuntimeError(job.get('error','Wan job failed'))
            if job['status']=='completed': break
            time.sleep(2)
        else:
            raise TimeoutError('Client wait expired; server-owned job remains bounded by its own guard. Do not resubmit automatically.')
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
        response=self.session.get(self.base+f'/api/jobs/{job_id}/files/raw.mp4',timeout=90)
        response.raise_for_status(); video=out/'raw.mp4'; video.write_bytes(response.content)
        import imageio.v2 as imageio
        from PIL import Image
        reader=imageio.get_reader(video); last=None
        try:
            for frame in reader: last=frame
        finally: reader.close()
        if last is None: raise ValueError('Generated video is empty')
        last_path=out/'last.png'; Image.fromarray(last).save(last_path)
        metadata={'kind':'wan','job_id':job_id,'scene_id':self.scene_id,'laboratory_supported':False}
        (out/'render.json').write_text(json.dumps(metadata),encoding='utf-8')
        return RenderResult(str(video),str(last_path),metadata)
