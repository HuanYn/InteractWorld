"""Explicit goal-keyframe experiment, NOT autonomous world-state prediction.

The supplied target is a generation condition and must be disclosed to users.
No goal RGB is pasted into outputs. Last latent affects the final4 decoded frames;
these frames alone cannot certify a generated transition or discovery.
"""
import torch


def endpoint_rollout(model, *, first, goal, noise, conditions, sigmas, mode):
    if mode not in ('clean','flow_noised'): raise ValueError('Unsupported endpoint mode')
    if noise.shape[1]!=25 or first.shape!=goal.shape or first.shape[1]!=1:
        raise ValueError('One start/goal latent and25 total slots required')
    if first.dtype!=noise.dtype or goal.dtype!=noise.dtype or first.device!=noise.device or goal.device!=noise.device:
        raise ValueError('Endpoint latents must match noise dtype/device before sampling')
    current=noise.clone(); goal_noise=noise[:,-1:].clone(); trace=[]
    def constrain(value,sigma):
        value[:,:1]=first
        value[:,-1:]=goal if mode=='clean' else (1-sigma)*goal+sigma*goal_noise
    constrain(current,sigmas[0])
    for step,(sigma,next_sigma) in enumerate(zip(sigmas[:-1],sigmas[1:])):
        timestep=(sigma*1000).expand(current.shape[:2]).clone()
        timestep[:,0]=0
        if mode=='clean': timestep[:,-1]=0
        velocity,_=model(current,conditional_dict=conditions,timestep=timestep,
                         replace_first_timestep_and_noise_latents=True)
        current=current+(next_sigma-sigma)*velocity
        constrain(current,next_sigma)
        if not torch.isfinite(current).all(): raise ValueError('Nonfinite endpoint state')
        trace.append({'step':step,'mode':mode,'first_clamped':True,'last_constrained':True})
    if not torch.equal(current[:,:1],first) or not torch.equal(current[:,-1:],goal):
        raise RuntimeError('Endpoint latent constraint failed')
    return current,trace


def cpu_contract():
    checks={}
    for mode in ('clean','flow_noised'):
        noise=torch.randn((1,25,1,2,2),generator=torch.Generator().manual_seed(109))
        first=torch.zeros_like(noise[:,:1]);goal=torch.ones_like(first)
        def fake(x,conditional_dict,timestep,**kwargs):
            assert (timestep[:,0]==0).all()
            if mode=='clean': assert (timestep[:,-1]==0).all()
            return torch.ones_like(x),None
        result,_=endpoint_rollout(fake,first=first,goal=goal,noise=noise,conditions={},
                                  sigmas=torch.tensor([1.,.5,0.]),mode=mode)
        assert torch.equal(result[:,:1],first) and torch.equal(result[:,-1:],goal)
        assert torch.allclose(result[:,1:-1],noise[:,1:-1]-1,atol=1e-6)
        checks[mode]=True
    return {'fixture_only':True,'endpoint_and_middle_update_checks':checks,'visual_success':False}
