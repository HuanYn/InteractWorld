"""Native Wan adapters for genuine self-rollout DMD; no production changes.

Only one frozen Wan backbone and one live FP32 LoRA bank occupy the GPU.
Generator/fake parameters and their full native optimizer states are parked on
CPU between phases. Real uses the frozen expanded80 Adapter/gate and zero-B
LoRA, with the existing explicit full-bidirectional attention mode. Native
parity, causal replay and peak memory are profile gates, not CPU claims.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from training.completion_dmd_core_v2 import require, finite


def native_amp(device):
    """All native forwards need BF16 AMP with the intentionally FP32 LoRA.

    CPU fixtures retain their original precision and cannot certify CUDA AMP.
    Nested AMP is safe for the explicit fake/replay training phases.
    """
    return torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else nullcontext()


class PositiveHistory:
    """Actual positive-only full-history KV adapter, not CFG2 with scale1."""
    branch_count=1
    external_cfg_scale=1.

    def __init__(self,model,config,*,capacity=61,check=lambda:None,initializers=None,cpu_layer_offload=True):
        self.model=model;self.config=config;self.capacity=capacity;self.check=check
        self.forward_count=0;self.position=0;self.kv=None;self.cross=None;self.pending=None
        self.initializers=initializers;self.action=None
        self.cpu_layer_offload=cpu_layer_offload;self.hooks=[];self.host_transfer_bytes=0
        self.host_transfer_seconds=0.;self.device=None

    @torch.no_grad()
    def initialize(self,first,conditions):
        backbone=self.model.model
        require(first.dtype==torch.bfloat16 and first.shape[1]==1 and self.position==0,
            'fresh nativeBF16 positive-history cache required')
        require(backbone.local_attn_size==-1 and not backbone.use_relative_rope
            and backbone.independent_first_frame and backbone.num_frame_per_block==3,
            'same absolute full-history causal model required')
        self.tokens=first.shape[-2]*first.shape[-1]//(backbone.patch_size[1]*backbone.patch_size[2])
        self.prompt=conditions.to(first.device)
        require(self.prompt.ndim==2,'microbatch1 positive text required')
        if self.initializers is None:
            from pipeline.causal_inference import CausalInferencePipeline
            self.initializers=(CausalInferencePipeline._initialize_kv_cache,
                CausalInferencePipeline._initialize_crossattn_cache)
        holder=SimpleNamespace(generator=self.model,local_attn_size=-1,frame_seq_length=self.tokens,
            num_transformer_blocks=len(backbone.blocks),args=SimpleNamespace(image_or_video_shape=[1,self.capacity]))
        cache_device='cpu' if self.cpu_layer_offload else first.device
        self.device=first.device
        self.initializers[0](holder,1,first.dtype,cache_device)
        self.initializers[1](holder,1,first.dtype,cache_device)
        self.kv,self.cross=holder.kv_cache1,holder.crossattn_cache
        if self.cpu_layer_offload:
            self._install_layer_transfers(backbone)
        self._call(first,0.,0,None);self.position=1

    def _install_layer_transfers(self,backbone):
        """Full61 history remains on CPU; one layer's full-capacity scratch.

        Upload only the written clean prefix and download only this call's
        overwritten block. There is no eviction or shortening of context.
        This is explicitly a slow memory fallback, never an acceleration.
        """
        import time
        for index,module in enumerate(backbone.blocks):
            host=self.kv[index];cross=self.cross[index]
            def before(_module,args,kwargs,host=host,cross=cross):
                tick=time.monotonic();end=int(host['local_end_index'])
                resident={}
                for key,value in host.items():
                    if torch.is_tensor(value) and key in ('k','v'):
                        resident[key]=torch.empty(value.shape,dtype=value.dtype,device=self.device)
                        resident[key][:,:end].copy_(value[:,:end])
                        self.host_transfer_bytes+=value[:,:end].numel()*value.element_size()
                    else:
                        resident[key]=value.to(self.device) if torch.is_tensor(value) else value
                        if torch.is_tensor(value):self.host_transfer_bytes+=value.numel()*value.element_size()
                text={key:(value.to(self.device) if torch.is_tensor(value) else value) for key,value in cross.items()}
                self.host_transfer_bytes+=sum(v.numel()*v.element_size() for v in cross.values() if torch.is_tensor(v))
                self.host_transfer_seconds+=time.monotonic()-tick
                return args,{**kwargs,'kv_cache':resident,'crossattn_cache':text}
            def after(_module,args,kwargs,output,host=host,cross=cross):
                tick=time.monotonic();resident=kwargs['kv_cache'];text=kwargs['crossattn_cache']
                stop=int(resident['local_end_index']);start=int(kwargs['current_start'])
                for key,value in resident.items():
                    if torch.is_tensor(value) and key in ('k','v'):
                        host[key][:,start:stop].copy_(value[:,start:stop])
                        self.host_transfer_bytes+=value[:,start:stop].numel()*value.element_size()
                    else:
                        host[key]=value.cpu() if torch.is_tensor(value) else value
                        if torch.is_tensor(value):self.host_transfer_bytes+=value.numel()*value.element_size()
                for key,value in text.items():
                    cross[key]=value.cpu() if torch.is_tensor(value) else value
                    if torch.is_tensor(value):self.host_transfer_bytes+=value.numel()*value.element_size()
                self.host_transfer_seconds+=time.monotonic()-tick
                return output
            self.hooks.append(module.register_forward_pre_hook(before,with_kwargs=True))
            self.hooks.append(module.register_forward_hook(after,with_kwargs=True))

    def _call(self,value,timestep,start,action):
        require(not torch.is_grad_enabled(),'own-history KV rollout must be detached')
        self.check()
        time=torch.as_tensor(timestep,device=value.device,dtype=torch.bfloat16).float().reshape(-1,1).expand(value.shape[0],value.shape[1])
        with native_amp(value.device):
            output=self.model.model(value.permute(0,2,1,3,4),t=time,context=[self.prompt],seq_len=None,
                act_context=action,act_context_scale=self.config.model.action_scale,kv_cache=self.kv,
                crossattn_cache=self.cross,current_start=start*self.tokens)
        output=output[0] if isinstance(output,tuple) else output
        output=output.permute(0,2,1,3,4)
        expected=(start+value.shape[1])*self.tokens
        require(all(int(x['global_end_index'])==expected and int(x['local_end_index'])==expected for x in self.kv),
            'full own-history cache evicted or advanced incorrectly')
        self.forward_count+=1
        finite(output,'native own-history velocity')
        return output

    def velocity(self,*,noisy_block,history,action_block,timestep,block_index,**unused):
        require(self.position==1+3*block_index==history.shape[1] and noisy_block.shape[1]==3,
            'generator may only read its completed own past')
        if self.pending!=block_index:
            require(self.pending is None,'interleaved generator block')
            from training.models.action_adapter import build_action_context
            self.action=build_action_context(action_block,height=self.config.data.height,width=self.config.data.width,
                device=noisy_block.device,dtype=torch.bfloat16)
            self.pending=block_index
        return self._call(noisy_block,timestep,self.position,self.action)

    def commit(self,endpoint,block_index):
        require(self.pending==block_index and self.position==1+3*block_index,'commit only current solved block')
        self._call(endpoint,0.,self.position,self.action)
        self.position+=3;self.pending=None;self.action=None

    def close(self):
        for hook in self.hooks:hook.remove()
        self.hooks=[]
        self.kv=None;self.cross=None;self.action=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


class WanCallbacks:
    """One model with explicit role activation delegated to StateBanks.

    The replay path uses the existing packed TF mask and self-generated clean
    past; history_x is deliberately not used because the native causal model
    currently has no implementation for that wrapper keyword.
    """
    def __init__(self,model,config,banks,*,check=lambda:None,checkpoint_replay=True):
        self.model=model;self.config=config;self.banks=banks;self.check=check
        self.checkpoint_replay=checkpoint_replay
        self.counts={'real_positive':0,'real_negative':0,'fake_positive':0,'replay':0}

    def score(self,*,role,noisy_video,actions,sigma,timestep,branch,conditions,
              attention_mode,gradient_enabled,absolute_start_latent):
        require(role in ('real','fake') and attention_mode=='bidirectional'
            and (branch=='positive' or (role=='real' and branch=='negative')),
            'realCFG5 and fakepositive-only explicit bidirectional score required')
        require(torch.is_grad_enabled()==gradient_enabled and (not gradient_enabled or role=='fake'),
            'real score frozen; only fake training owns score gradients')
        if self.banks.current!=role:self.banks.activate(role)
        self.check()
        from training.models.action_adapter import build_action_context
        times=torch.full(noisy_video.shape[:2],float(timestep),device=noisy_video.device,dtype=torch.float32)
        times[:,0]=0
        with native_amp(noisy_video.device):
            output=self.model(noisy_image_or_video=noisy_video,
                conditional_dict=dict(prompt_embeds=[conditions.to(noisy_video.device)],
                    act_context=build_action_context(actions,height=self.config.data.height,width=self.config.data.width,
                        device=noisy_video.device,dtype=torch.bfloat16),act_context_scale=self.config.model.action_scale),
                timestep=times,training_attention_mode='bidirectional',replace_first_timestep_and_noise_latents=False)
        self.counts[role+'_'+branch]+=1
        return output[0] if isinstance(output,tuple) else output

    def replay(self,*,noisy_block,history,actions,action_block,sigma,timestep,block_index,branch,cfg_scale,conditions):
        require(self.banks.current=='generator' and self.banks.live_graph=='generator'
            and block_index==19 and history.shape[1]==58 and branch=='positive' and cfg_scale==1.,
            'last-block replay only under stable generator bank')
        self.check()
        from training.models.action_adapter import build_action_context
        # Only own completed prefix is clean; current/future clean is zero.
        clean=torch.cat((history.detach(),torch.zeros_like(noisy_block)),dim=1)
        times=torch.full(clean.shape[:2],float(timestep.flatten()[0]),device=clean.device,dtype=torch.float32)
        times[:,0]=0
        act=build_action_context(actions,height=self.config.data.height,width=self.config.data.width,
            device=clean.device,dtype=torch.bfloat16)
        def forward(current):
            self.check()
            noisy=torch.cat((torch.zeros_like(history),current),dim=1)
            with native_amp(current.device):
                output=self.model(noisy_image_or_video=noisy,
                    conditional_dict=dict(prompt_embeds=[conditions.to(clean.device)],
                        act_context=act,act_context_scale=self.config.model.action_scale),
                    timestep=times,clean_x=clean,aug_t=torch.zeros_like(times),
                    replace_first_timestep_and_noise_latents=True)
            output=output[0] if isinstance(output,tuple) else output
            self.counts['replay']+=1
            return output[:,-3:]
        return checkpoint(forward,noisy_block,use_reentrant=False,preserve_rng_state=True) if self.checkpoint_replay else forward(noisy_block)


def gradient_statistics(model,names):
    gradients=[dict(model.named_parameters())[n].grad for n in names]
    require(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients),
        'each LoRA tensor must have an actual finite task gradient')
    norm=sum(float(g.float().square().sum()) for g in gradients)**.5
    require(norm>0,'nonzero real LoRA gradient required')
    return dict(gradient_norm=norm,gradient_tensors=len(gradients),all_finite=True)


def optimizer_phase(banks,role,*,make_optimizer,check=lambda:None):
    """Instantiate one native optimizer, restoring this role's CPU full state.

    Caller captures and deletes the optimizer before any bank switch. The
    saved state is a deep CPU clone, not state_dict aliases in shared memory.
    """
    banks.activate(role)
    check()
    return make_optimizer(banks.model,banks.optimizer_states[role],check=check)


def memory_estimate(model_config,*,height=480,width=832,capacity=61,batch=1,element_size=2):
    """Exact native allocator bytes from actual small modelconfig, no weights.

    Activation/autograd workspace is unknown until actual profiling. Values
    here must not be called measured reserved memory or a resource pass.
    """
    patch=tuple(model_config['patch_size']);dim=model_config['dim']
    layers=model_config['num_layers'];heads=model_config['num_heads']
    require(patch[0]==1 and len(patch)==3 and dim%heads==0
        and all(type(v) is int and v>0 for v in (dim,layers,heads,height,width,capacity,batch,element_size)),
        'actual native model geometry required')
    latent_h,latent_w=height//16,width//16
    require(latent_h%patch[1]==latent_w%patch[2]==0,'latent geometry must divide actual patch')
    tokens=latent_h*latent_w//(patch[1]*patch[2])
    layer_kv=batch*capacity*tokens*dim*2*element_size
    layer_cross=batch*512*dim*2*element_size
    return dict(schema='completion_dmd_exact_kv_cpu_estimate_v2',model_config_geometry=dict(
        patch_size=list(patch),dim=dim,num_layers=layers,num_heads=heads),latent_height=latent_h,
        latent_width=latent_w,tokens_per_latent=tokens,capacity_latents=capacity,
        full_resident_kv_bytes=layer_kv*layers,full_cross_kv_bytes=layer_cross*layers,
        cpu_layer_offload=True,offloaded_history_context='identical_full_generated_prefix_no_eviction',
        gpu_layer_kv_scratch_bytes=layer_kv,gpu_layer_cross_scratch_bytes=layer_cross,
        packed_generator_replay_latents=122,packed_score_long_latents=25,packed_score_short_latents=13,
        activation_workspace_bytes=None,native_reserved_bytes_measured=None,
        resources_qualified=False,slow_resource_fallback_not_acceleration=True)


@torch.no_grad()
def probe_kv_offload_parity(model,config,first,positive,actions,*,check=lambda:None):
    """Real GPU small-prefix dense/offload parity, before the full61 profile."""
    snapshots=[]
    for offload in (False,True):
        backend=PositiveHistory(model,config,capacity=4,cpu_layer_offload=offload,check=check)
        try:
            backend.initialize(first,positive)
            noise=torch.ones((first.shape[0],3,*first.shape[2:]),device=first.device,dtype=torch.bfloat16)
            velocity=backend.velocity(noisy_block=noise,history=first,action_block=actions[:,:12],
                timestep=torch.tensor([624.],device=first.device),block_index=0)
            backend.commit(noise,0)
            snapshots.append(dict(velocity=velocity.detach().cpu().clone(),kv=[
                {k:v.detach().cpu().clone() for k,v in layer.items() if torch.is_tensor(v)} for layer in backend.kv],
                forwards=backend.forward_count,position=backend.position,bytes=backend.host_transfer_bytes,
                seconds=backend.host_transfer_seconds))
        finally:backend.close()
    dense,offloaded=snapshots
    require(dense['forwards']==offloaded['forwards']==3 and dense['position']==offloaded['position']==4,
        'dense/offload native prefix forward/cache-length mismatch')
    require(torch.equal(dense['velocity'],offloaded['velocity']) and all(
        torch.equal(dense['kv'][i][k],offloaded['kv'][i][k]) for i in range(len(dense['kv'])) for k in dense['kv'][i]),
        'CPU layer offload changed native prefix values/caches')
    return dict(status='passed',actual_model_forwards=6,verified_prefix_latents=4,
        exact_velocity_equal=True,exact_written_cache_equal=True,evictions=0,
        offload_transfer_bytes=offloaded['bytes'],offload_transfer_seconds=offloaded['seconds'],
        full61_resources_qualified=False)


def probe_native_replay_causality(model,config,video,actions,positive,*,check=lambda:None):
    """Actual final-block perturbation/VJP under installed native TF mask.

    Tests earlier noisy blocks and current clean block are unreadable. Last
    block has no future block; the unchanged TF/ODE mask audit covers earlier
    chunks. This is a finite native probe, not a full Jacobian proof.
    """
    from training.models.action_adapter import build_action_context
    clean=torch.cat((video[:,:58].detach(),torch.zeros_like(video[:,-3:])),dim=1)
    noisy=torch.cat((torch.zeros_like(video[:,:58]),video[:,-3:].detach()),dim=1)
    times=torch.full(video.shape[:2],624.,device=video.device,dtype=torch.float32);times[:,0]=0
    act=build_action_context(actions,height=config.data.height,width=config.data.width,
        device=video.device,dtype=torch.bfloat16)
    def call(x,c):
        check()
        with native_amp(x.device):
            output=model(noisy_image_or_video=x,conditional_dict=dict(prompt_embeds=[positive],
                act_context=act,act_context_scale=config.model.action_scale),timestep=times,
                clean_x=c,aug_t=torch.zeros_like(times),replace_first_timestep_and_noise_latents=True)
        return (output[0] if isinstance(output,tuple) else output)[:,-3:]
    x=noisy.clone().requires_grad_(True);c=clean.clone().requires_grad_(True)
    saved=torch.autograd.graph.save_on_cpu(pin_memory=True) if video.device.type=='cuda' else nullcontext()
    with saved:
        base=call(x,c)
        coefficients=torch.linspace(.2,1.2,base.numel(),device=base.device,dtype=torch.float32).reshape(base.shape)
        gx,gc_=torch.autograd.grad((base.float()*coefficients).sum(),(x,c),allow_unused=True)
    require(gx is not None and bool(torch.isfinite(gx).all()) and float(gx[:,-3:].abs().max())>0,
        'native current noisy block must have a finite actual VJP')
    clean_leak=0. if gc_ is None else float(gc_[:,-3:].abs().max())
    noisy_leak=float(gx[:,:58].abs().max())
    require(clean_leak==0. and noisy_leak==0.,'native final block read current clean or earlier noisy latents')
    with torch.no_grad():
        changed_clean=clean.clone();changed_clean[:,-3:]+=11
        changed_noisy=noisy.clone();changed_noisy[:,:58]+=17
        a,b=call(noisy,changed_clean),call(changed_noisy,clean)
        clean_difference=float((a.float()-base.detach().float()).abs().max())
        noisy_difference=float((b.float()-base.detach().float()).abs().max())
    require(clean_difference==noisy_difference==0.,'native final-block causal perturbation changed output')
    model.zero_grad(set_to_none=True)
    return dict(status='passed',actual_forwards=3,last_block_index=19,packed_latents=122,
        current_clean_vjp_max=clean_leak,earlier_noisy_vjp_max=noisy_leak,
        current_clean_perturbation_max=clean_difference,earlier_noisy_perturbation_max=noisy_difference,
        full_jacobian_proven=False,actual_earlier_block_mask_audit='unchanged_formalTF_ODE_source_required')


def real_fake_native_probe(callbacks,banks,video,actions,positive,negative):
    """Actual-forward and actual-gradient probe required during GPUprofile.

    Does not count as visual/control success. Full mask-equivalence and memory
    gates are recorded separately by the profiled training request.
    """
    from training.completion_dmd_core_v2 import ScoreWindow
    noisy=video[:,:4].detach().clone();sigma=torch.tensor(.5,device=video.device,dtype=torch.bfloat16)
    kwargs=dict(noisy_video=noisy,actions=actions[:,:12],sigma=sigma,timestep=sigma*1000,
        branch='positive',conditions=positive,attention_mode='bidirectional',absolute_start_latent=0)
    banks.activate('real')
    with torch.no_grad():real=callbacks.score(role='real',gradient_enabled=False,**kwargs)
    require(not real.requires_grad and all(p.grad is None for p in banks.model.parameters()),'frozen teacher acquired gradient')
    banks.activate('fake');banks.begin_graph('fake')
    fake=callbacks.score(role='fake',gradient_enabled=True,**kwargs)
    fake[:,1:].float().square().mean().backward()
    stats=gradient_statistics(banks.model,banks.names)
    banks.end_graph('fake');banks.model.zero_grad(set_to_none=True)
    return dict(real_frozen=True,fake_native_backprop=True,**stats)
