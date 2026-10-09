"""Runtime-only extension of the frozen R005 sampler to exact lab text and RGB.

Not a new trained laboratory model. First-frame continuation and short windows
limit drift but do not guarantee identity, object interaction, or geometry.
"""
import gc
import json
import os
from pathlib import Path
import time

class LaboratoryVideoModel:
    def __init__(self, snapshot, prompt_cache):
        rollout_setting=os.environ.get('IW_ACTION_ROLLOUT')
        if not rollout_setting:
            raise ValueError('Set IW_ACTION_ROLLOUT to your actual rollout YAML, absolute or relative to snapshot')
        snapshot=Path(snapshot).resolve()
        rollout=Path(rollout_setting)
        rollout=(rollout if rollout.is_absolute() else snapshot/rollout).resolve(strict=True)
        if not rollout.is_relative_to(snapshot):
            raise ValueError('Rollout configuration must be inside the operator-selected snapshot')
        import torch
        from .wan_assets import load_portable_action_inputs
        from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
        from training.models.lora import configure_action_teacher,load_trainable_state_dict,LoRALinear
        state,self.config,_,self.lineage=load_portable_action_inputs(rollout,snapshot)
        cache=torch.load(prompt_cache,map_location='cpu',weights_only=True)
        self.prompts=cache['prompts']; self.embeddings=cache['embeddings']
        self.model=WanDiffusionWrapper(model_name=self.config.base_model_path,model_type='ci2v',
            is_causal=False,timestep_shift=5.0,downscale_factor_control_adapter=16)
        configure_action_teacher(self.model.model,rank=self.config.lora_rank,alpha=self.config.lora_alpha,dropout=0)
        load_trainable_state_dict(self.model,state); del state
        self.model.requires_grad_(False).eval().to(dtype=torch.bfloat16)
        self.vae=WanVAEWrapper(pretrained_path=str(Path(self.config.base_model_path)/'Wan2.2_VAE.pth'))
        self.vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
        self.loras=[m for m in self.model.modules() if isinstance(m,LoRALinear)]
        gc.collect()

    def _action_schedule(self,skills,frames):
        """Default seven-skill compiler. Isolated diagnostics may override this hook."""
        import numpy as np
        from .agent import SkillSelector
        actions=np.zeros((frames,8),dtype=np.float32); start=0
        keys_order=['W','A','S','D','I','J','K','L']
        segments=SkillSelector().action_segments(skills,frames=frames)
        for segment in segments:
            stop=start+segment['frames']
            for key in segment['keys']: actions[start:stop,keys_order.index(key)]=1
            start=stop
        return actions,segments

    def require_prompt(self,prompt_key):
        if prompt_key=='door_unlock' and (self.embeddings.get(prompt_key) is None
                or not isinstance(self.prompts.get(prompt_key),str) or not self.prompts[prompt_key].strip()):
            raise ValueError("Missing encoded lab prompt 'door_unlock': encode its exact unlock text; scene/door_open fallback is forbidden")
        if prompt_key not in self.embeddings or prompt_key=='negative':
            raise ValueError('Unknown lab prompt')

    def generate(self,initial,prompt_key,skills,output_dir,seed=42,seconds=3,steps=40,cfg=3.0,lora_strength=1.0,framing_condition=None):
        self.require_prompt(prompt_key)  # Reject missing encoding before generation work.
        import numpy as np
        import torch
        from PIL import Image,ImageOps
        import imageio.v2 as imageio
        from .agent import SkillSelector
        from .models import CameraSkill
        from training.models.action_adapter import build_action_context
        from training.longforcing_lite import euler_sigmas
        from training.demo.action_backend import euler_rollout
        if seconds not in (3,6) or steps not in (20,30,40) or not 1<=cfg<=5 or not 0<=lora_strength<=1: raise ValueError('Invalid bounded rendering settings')
        skills=[CameraSkill(s) for s in skills]
        if not skills or len(skills)>7: raise ValueError('Invalid camera skills')
        if framing_condition is not None:
            from .camera_framing import MODE
            if framing_condition!=MODE or seconds!=6 or skills!=[CameraSkill.static]:
                raise ValueError('Framing requires six seconds and explicit static player keys')
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
        initial=ImageOps.fit(Image.open(initial).convert('RGB'),(832,480),method=Image.Resampling.LANCZOS)
        initial.save(out/'input.png'); framing=None; target=None
        if framing_condition is not None:
            from .camera_framing import current_frame_goal
            target,framing=current_frame_goal(initial,framing_condition)
            target.save(out/'supplied-framing-goal.png')
        initial=np.asarray(initial)
        actions,segments=self._action_schedule(skills,seconds*16)
        np.save(out/'actions.npy',actions)
        for m in self.loras: m.scaling=m.alpha/m.rank*lora_strength
        self.model.to('cpu'); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        began=time.monotonic(); self.vae.to('cuda')
        pixel=torch.from_numpy(initial.copy()).permute(2,0,1)[None,:,None].float().div(127.5).sub(1)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            first=self.vae.encode_to_latent(pixel.to('cuda',dtype=torch.bfloat16)).detach().cpu()
            if target is not None:
                target_pixel=torch.from_numpy(np.asarray(target).copy()).permute(2,0,1)[None,:,None].float().div(127.5).sub(1)
                goal=self.vae.encode_to_latent(target_pixel.repeat(1,1,9,1,1).to('cuda',dtype=torch.bfloat16))[:,-1:].detach().cpu()
                del target_pixel
        self.vae.to('cpu'); gc.collect(); torch.cuda.empty_cache(); self.model.to('cuda')
        latent_frames=seconds*4+1
        noise=torch.randn((1,latent_frames,48,30,52),generator=torch.Generator().manual_seed(seed)).to('cuda',dtype=torch.bfloat16)
        common={'act_context_scale':self.config.action_scale*lora_strength,
                'act_context':build_action_context(torch.from_numpy(actions)[None],height=480,width=832,device='cuda',dtype=torch.bfloat16)}
        positive=dict(common,prompt_embeds=[self.embeddings[prompt_key].to('cuda')])
        negative=dict(common,prompt_embeds=[self.embeddings['negative'].to('cuda')])
        def guided(current,conditional_dict,timestep,**kwargs):
            v,_=self.model(current,conditional_dict=conditional_dict,timestep=timestep,**kwargs)
            if cfg>1:
                uncond,_=self.model(current,conditional_dict=negative,timestep=timestep,**kwargs)
                v=uncond+cfg*(v-uncond)
            return v,None
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            sigmas=euler_sigmas(steps,shift=5.0,device=torch.device('cuda'),dtype=torch.bfloat16)
            if framing is None:
                latent=euler_rollout(guided,first=first.to('cuda'),noise=noise,conditions=positive,sigmas=sigmas).cpu()
            else:
                from .endpoint_conditioning import endpoint_rollout
                latent,trace=endpoint_rollout(guided,first=first.to(device=noise.device,dtype=noise.dtype),
                                             goal=goal.to(device=noise.device,dtype=noise.dtype),noise=noise,
                                             conditions=positive,sigmas=sigmas,mode='clean')
                latent=latent.cpu(); del goal
                (out/'endpoint-trace.json').write_text(json.dumps(trace,indent=2))
        self.model.to('cpu'); del noise,positive,negative,common; gc.collect(); torch.cuda.empty_cache(); self.vae.to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pixels=self.vae.decode_to_pixel(latent.to('cuda'),use_cache=False,return_in_cpu=True)
        self.vae.to('cpu'); torch.cuda.empty_cache()
        if tuple(pixels.shape)!=(1,seconds*16+1,3,480,832) or not torch.isfinite(pixels).all(): raise ValueError('Invalid decoded video')
        rgb=pixels[0].permute(0,2,3,1).add(1).mul(127.5).round().clamp(0,255).byte().numpy(); rgb[0]=initial
        with imageio.get_writer(out/'video.mp4',fps=16,codec='libx264',quality=8,macro_block_size=16) as writer:
            for frame in rgb: writer.append_data(frame)
        Image.fromarray(rgb[-1]).save(out/'last.png')
        # Contact sheet is diagnostic extracted frames, not synthesized visual evidence.
        sheet=Image.new('RGB',(832*3,480))
        for col,index in enumerate((0,len(rgb)//2,len(rgb)-1)): sheet.paste(Image.fromarray(rgb[index]),(832*col,0))
        sheet.save(out/'contact.jpg')
        report=dict(kind='wan_laboratory',base='Wan2.2-TI2V-5B',checkpoint='R005-step1040',
            prompt_key=prompt_key,prompt=self.prompts[prompt_key],negative_prompt=self.prompts['negative'],
            conditioning_image=str(out/'input.png'),seed=seed,seconds=seconds,frames=len(rgb),width=832,height=480,fps=16,
            sampler={'solver':'flow_euler','steps':steps,'shift':5,'cfg':cfg},lora_strength=lora_strength,
            action_scale=self.config.action_scale*lora_strength,action_segments=segments,
            elapsed_seconds=time.monotonic()-began,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            visual_goal_verified=False,laboratory_finetuning=False,continuation='previous_generated_RGB_reencoded',
            asset_provenance='Initial lab asset is imagegen concept art; subsequent frames are Wan outputs')
        if framing is not None:
            report['framing_constraint']=framing
            report['controlled_effect']='2D closer framing, NOT walking or 3D camera navigation'
        (out/'render.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        return {'video_path':str(out/'video.mp4'),'last_frame':str(out/'last.png'),'metadata':report}
