import copy
from contextlib import contextmanager

import pytest
import torch

from training import completion_dmd_core_v2 as dmd
from training import completion_dmd_native_v2 as native


def test_native_amp_routes_cuda_bf16_without_initializing_cuda_cpu_fixture(monkeypatch):
    called=[];before=torch.cuda.is_initialized()
    @contextmanager
    def autocast(device_type,*,dtype):
        called.append((device_type,dtype));yield
    monkeypatch.setattr(torch,'autocast',autocast)
    with native.native_amp('cpu'):pass
    assert called==[]
    with native.native_amp('cuda:0'):pass
    assert called==[('cuda',torch.bfloat16)] and torch.cuda.is_initialized()==before


def sample():
    return torch.arange(61,dtype=torch.float32).reshape(1,61,1,1,1).bfloat16(),torch.zeros(1,240,8)


@pytest.mark.parametrize('arm,n,length', [('short',2,13),('long',1,25)])
def test_exact_windows_and_same_tail_noise(arm,n,length):
    video,actions=sample();windows=dmd.score_windows(video,actions,dmd.DMDConfig(arm))
    assert len(windows)==n and all(w.video.shape[1]==length for w in windows)
    assert torch.equal(torch.cat([w.video[:,1:] for w in windows],1),video[:,37:])
    assert windows[0].video[:,0].item()==36
    if arm=='short':assert windows[1].video[:,0].item()==48


def test_config_rejects_shrink_or_other_guidance():
    for cfg in (dmd.DMDConfig('x'),dmd.DMDConfig('long',rollout_blocks=8),dmd.DMDConfig('long',real_cfg=6.)):
        with pytest.raises(ValueError):cfg.validate()


def test_noise_and_sigma_independent_global_rng():
    state=torch.get_rng_state().clone()
    one,seed=dmd.independent_noise((1,24,1,1,1),seed=42,update=3,purpose='generator-score-noise')
    two,other=dmd.independent_noise(one.shape,seed=42,update=3,purpose='fake-training-noise')
    assert seed!=other and not torch.equal(one,two) and torch.equal(state,torch.get_rng_state())
    sigma,t,record=dmd.score_level(seed=42,update=3,purpose='generator-score')
    assert sigma.dtype==torch.bfloat16 and .019<sigma<.981 and t==sigma*1000
    assert record['actual_model_timestep']==float(t)


def test_record_is_real_dmd_not_mse_and_source_pinned():
    r=dmd.DMDConfig('long').record()
    assert r['real_guidance_definition']=='negative+5*(positive-negative)'
    assert r['fake_guidance_definition']=='positive_only_CFG1'
    assert r['source_revision'] in r['source_url'] and r['future_ground_truth_supplied'] is False


@pytest.mark.parametrize('arm,expected', [('short',6),('long',3)])
def test_distribution_gradient_mirrors_conditions_normalizes_future_only(arm,expected):
    video,actions=sample();calls=[]
    def callback(**k):
        assert not torch.is_grad_enabled() and not k['gradient_enabled']
        calls.append({**k,'noisy_video':k['noisy_video'].clone()})
        return torch.full_like(k['noisy_video'], {'fake':2.,'real':1.}[k['role']] if k['branch']=='positive' else 0.)
    grad,record=dmd.distribution_gradient(callback,video=video,actions=actions,positive='p',negative='n',
        config=dmd.DMDConfig(arm),seed=42,update=0)
    assert grad.shape==(1,24,1,1,1) and not grad.requires_grad
    assert record['score_forwards']==expected and torch.all(grad>0)
    for at in range(0,expected,3):
        fake,real,negative=calls[at:at+3]
        assert fake['conditions']==real['conditions']=='p' and negative['conditions']=='n'
        assert torch.equal(fake['noisy_video'],negative['noisy_video'])
        assert torch.equal(fake['actions'],real['actions'])
        # fake=2, realguided=5; fake_x0-real_x0=3*sigma.
        denom=record['windows'][at//3]['normalizer'][0]
        assert torch.allclose(grad[:,at//3*12:(at//3+1)*12] if arm=='short' else grad,
            torch.full_like(grad[:,at//3*12:(at//3+1)*12] if arm=='short' else grad,
                3*record['level']['actual_sigma']/denom),atol=.01)


def test_equal_scores_zero_distribution_gradient_not_endpoint_target():
    video,actions=sample()
    grad,_=dmd.distribution_gradient(lambda **k:torch.zeros_like(k['noisy_video']),video=video,
        actions=actions,positive='p',negative='n',config=dmd.DMDConfig('long'),seed=42,update=0)
    assert torch.count_nonzero(grad)==0


def test_nonfinite_or_mutating_score_fails_closed():
    video,actions=sample()
    for bad in ('nan','mutate'):
        def callback(**k):
            if bad=='mutate':k['noisy_video'][:,0]+=1
            return torch.full_like(k['noisy_video'],float('nan') if bad=='nan' else 0.)
        with pytest.raises(ValueError):dmd.distribution_gradient(callback,video=video,actions=actions,
            positive=None,negative=None,config=dmd.DMDConfig('long'),seed=42,update=0)


@pytest.mark.parametrize('arm', ['short','long'])
def test_fake_loss_is_true_task_gradient_own_targets_detached(arm):
    video,actions=sample();video.requires_grad_(True);weight=torch.nn.Parameter(torch.tensor(.2));calls=[]
    def callback(**k):
        assert k['role']=='fake' and k['branch']=='positive' and k['gradient_enabled']
        assert not k['noisy_video'].requires_grad
        calls.append(k)
        return (k['noisy_video']*weight).bfloat16()
    loss,record=dmd.fake_flow_loss(callback,video=video,actions=actions,positive='p',
        config=dmd.DMDConfig(arm),seed=42,update=2)
    loss.backward()
    assert weight.grad is not None and weight.grad.abs()>0 and video.grad is None
    assert record['scored_future_latents']==24 and len(calls)==(2 if arm=='short' else 1)


def test_fake_ignores_anchor_prediction():
    video,actions=sample();w=torch.nn.Parameter(torch.tensor(1.))
    def callback(**k):
        value=k['noisy_video']*w
        return torch.cat((value[:,:1]*1e10,value[:,1:]),1).bfloat16()
    loss,_=dmd.fake_flow_loss(callback,video=video,actions=actions,positive='p',
        config=dmd.DMDConfig('long'),seed=42,update=0)
    assert float(loss)<10000


def test_replay_has_last_block_real_gradient_and_own_detached_history():
    video,actions=sample();video.requires_grad_(True);weight=torch.nn.Parameter(torch.tensor(.3));calls=[]
    def callback(**k):
        assert k['history'].shape[1]==58 and not k['history'].requires_grad
        assert k['block_index']==19 and k['action_block'].shape[1]==12
        calls.append(k)
        return (k['noisy_block']*weight+k['history'].mean()*.01).bfloat16()
    replay,record=dmd.replay_last_block(callback,video=video,actions=actions,positive='p',seed=42,config=dmd.DMDConfig('long'))
    loss=dmd.generator_surrogate(replay,torch.ones(1,24,1,1,1),config=dmd.DMDConfig('long'))
    loss.backward()
    assert len(calls)==4 and weight.grad is not None and weight.grad.abs()>0 and video.grad is None
    assert record['gradient_latents']==3 and len(set(record['noise_seeds']))==4


def test_surrogate_matches_detached_score_vjp_not_endpoint_mse():
    replay=torch.zeros(1,3,1,1,1,requires_grad=True);gradient=torch.arange(24.).reshape(1,24,1,1,1)
    loss=dmd.generator_surrogate(replay,gradient,config=dmd.DMDConfig('long'));loss.backward()
    assert torch.allclose(replay.grad,gradient[:,-3:]/3,rtol=1e-6,atol=1e-7)


def test_actions_are_periodic_real_records_not_synthesized_direction():
    original=torch.zeros(1,96,8);original[:,0,5]=1
    actual=dmd.repeat_actions(original)
    assert actual.shape==(1,240,8) and actual[:,96,5]==actual[:,192,5]==1
    assert torch.equal(actual[:,:96],original)


class TinyBank(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Parameter(torch.tensor(2.),requires_grad=False)
        self.attn=torch.nn.Module();self.attn.lora_a=torch.nn.Linear(1,1,bias=False)
        self.attn.lora_b=torch.nn.Linear(1,1,bias=False)


def bank():
    model=TinyBank();state={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    return model,dmd.StateBanks(model,state)


def test_banks_share_backbone_but_real_frozen_and_states_do_not_alias():
    model,banks=bank();base=model.base.clone();real=copy.deepcopy(banks.banks['real'])
    banks.activate('fake');opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    banks.begin_graph('fake');sum(p.square().sum() for n,p in model.named_parameters() if n in banks.names).backward()
    with pytest.raises(ValueError):banks.activate('real')
    opt.step();banks.end_graph('fake');banks.capture('fake',opt)
    assert any(not torch.equal(banks.banks['fake'][n],banks.banks['generator'][n]) for n in banks.names)
    assert all(torch.equal(real[n],banks.banks['real'][n]) for n in real) and torch.equal(base,model.base)
    snap=banks.state_dict();next(iter(banks.banks['fake'].values())).add_(1)
    assert any(not torch.equal(snap['banks']['fake'][n],banks.banks['fake'][n]) for n in banks.names)
    banks.load_state_dict(snap);banks.activate('real')
    assert all(not dict(model.named_parameters())[n].requires_grad for n in banks.names)
    assert all(torch.count_nonzero(v)==0 for n,v in banks.banks['real'].items() if '.lora_b.' in n)


def test_banks_reject_corrupt_frozen_teacher_and_missing_native_states():
    _,banks=bank();value=banks.state_dict()
    value['banks']['real'][banks.names[0]]+=1
    with pytest.raises(ValueError):banks.load_state_dict(value)
    value=banks.state_dict();value['optimizer_states'].pop('fake')
    with pytest.raises(ValueError):banks.load_state_dict(value)


def test_cpu_memory_estimate_uses_actual_16x_vae_and_patch2_not_rgb_pixels():
    result=native.memory_estimate(dict(patch_size=[1,2,2],dim=3072,num_layers=30,num_heads=24))
    assert result['tokens_per_latent']==390
    assert result['full_resident_kv_bytes']==30*61*390*3072*2*2
    assert result['gpu_layer_kv_scratch_bytes']*30==result['full_resident_kv_bytes']
    assert result['activation_workspace_bytes'] is None and not result['resources_qualified']
    assert result['slow_resource_fallback_not_acceleration']


def test_native_replay_invokes_packed_self_context_not_unsupported_history_keyword():
    model,banks=bank();banks.activate('generator');banks.begin_graph('generator');calls=[]
    class Model:
        def __call__(self,**kwargs):
            calls.append(kwargs)
            return kwargs['noisy_image_or_video']*dict(model.named_parameters())[banks.names[0]].sum()
    config=type('Config',(),dict(model=type('M',(),dict(action_scale=1.))(),data=type('D',(),dict(height=16,width=16))()))()
    callback=native.WanCallbacks(Model(),config,banks,checkpoint_replay=False)
    value=torch.ones(1,3,1,1,1,dtype=torch.bfloat16)
    output=callback.replay(noisy_block=value,history=torch.ones(1,58,1,1,1,dtype=torch.bfloat16),
        actions=torch.zeros(1,240,8),action_block=torch.zeros(1,12,8),sigma=torch.ones(1),
        timestep=torch.ones(1),block_index=19,branch='positive',cfg_scale=1.,conditions=torch.zeros(1,2))
    assert output.shape==value.shape
    kwargs=calls[0]
    assert 'history_x' not in kwargs and kwargs['clean_x'].shape[1]==61
    assert torch.count_nonzero(kwargs['clean_x'][:,-3:])==0
    assert torch.count_nonzero(kwargs['noisy_image_or_video'][:,:58])==0


class CacheBlock(torch.nn.Module):
    def forward(self,x,*,kv_cache,crossattn_cache,current_start):
        count=x.shape[1];stop=current_start+count
        kv_cache['k'][:,current_start:stop]=x.reshape(1,count,1,1)
        kv_cache['v'][:,current_start:stop]=x.reshape(1,count,1,1)*2
        kv_cache['global_end_index'].fill_(stop);kv_cache['local_end_index'].fill_(stop)
        crossattn_cache['is_init']=True
        return x+kv_cache['k'][:,:stop].mean()


class CacheBackbone(torch.nn.Module):
    local_attn_size=-1;use_relative_rope=False;independent_first_frame=True;num_frame_per_block=3
    patch_size=(1,1,1);num_heads=1;dim=1
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([CacheBlock(),CacheBlock()])
    def forward(self,x,*,kv_cache,crossattn_cache,current_start,**unused):
        x=x.permute(0,2,1,3,4)
        for index,block in enumerate(self.blocks):
            x=block(x,kv_cache=kv_cache[index],crossattn_cache=crossattn_cache[index],current_start=current_start)
        return x.permute(0,2,1,3,4)


def initializers():
    def kv(holder,batch,dtype,device):
        holder.kv_cache1=[dict(k=torch.zeros(1,holder.args.image_or_video_shape[1],1,1,dtype=dtype,device=device),
            v=torch.zeros(1,holder.args.image_or_video_shape[1],1,1,dtype=dtype,device=device),
            global_end_index=torch.zeros(1,dtype=torch.long,device=device),
            local_end_index=torch.zeros(1,dtype=torch.long,device=device),
            ref_token_len=torch.zeros(1,dtype=torch.long,device=device)) for _ in range(holder.num_transformer_blocks)]
    def cross(holder,batch,dtype,device):
        holder.crossattn_cache=[dict(k=torch.zeros(1,2,1,1,dtype=dtype,device=device),
            v=torch.zeros(1,2,1,1,dtype=dtype,device=device),is_init=False) for _ in range(holder.num_transformer_blocks)]
    return kv,cross


def test_dense_and_layer_offload_same_written_cache_values_lengths_counts(monkeypatch):
    from types import SimpleNamespace
    model=SimpleNamespace(model=CacheBackbone());config=SimpleNamespace(model=SimpleNamespace(action_scale=1.),data=SimpleNamespace(height=16,width=16))
    results=[]
    first=torch.ones(1,1,1,1,1,dtype=torch.bfloat16)
    for offload in (False,True):
        backend=native.PositiveHistory(model,config,capacity=7,cpu_layer_offload=offload,initializers=initializers())
        try:
            backend.initialize(first,torch.zeros(1,2))
            history=first
            for block in range(2):
                noisy=torch.full((1,3,1,1,1),float(block+2),dtype=torch.bfloat16)
                with torch.no_grad():
                    out=backend.velocity(noisy_block=noisy,history=history,action_block=torch.zeros(1,12,8),timestep=torch.tensor([624.]),block_index=block)
                    backend.commit(noisy,block)
                history=torch.cat((history,noisy),1)
            results.append(dict(out=out.clone(),kv=dmd.cpu_tree(backend.kv),position=backend.position,forwards=backend.forward_count,bytes=backend.host_transfer_bytes))
        finally:backend.close()
        assert all(not b._forward_hooks and not b._forward_pre_hooks for b in model.model.blocks)
    a,b=results
    assert a['forwards']==b['forwards']==5 and a['position']==b['position']==7 and b['bytes']>0
    assert torch.equal(a['out'],b['out'])
    assert all(torch.equal(x[k],y[k]) for x,y in zip(a['kv'],b['kv']) for k in x)


def test_failed_offload_call_removes_only_its_own_hooks():
    from types import SimpleNamespace
    model=SimpleNamespace(model=CacheBackbone());other=CacheBackbone()
    external=model.model.blocks[0].register_forward_hook(lambda *args:None)
    count=len(model.model.blocks[0]._forward_hooks)
    backend=native.PositiveHistory(model,SimpleNamespace(model=SimpleNamespace(action_scale=1.)),capacity=4,
        initializers=initializers(),check=lambda:(_ for _ in ()).throw(RuntimeError('fixture_failure')))
    try:
        with pytest.raises(RuntimeError):backend.initialize(torch.ones(1,1,1,1,1,dtype=torch.bfloat16),torch.zeros(1,2))
    finally:backend.close()
    assert len(model.model.blocks[0]._forward_hooks)==count
    assert not model.model.blocks[0]._forward_pre_hooks and not other.blocks[0]._forward_hooks
    external.remove()
