"""Opt-in sparse past-reference probe; never changes the production renderer.

Two clean latent slots (past reference, current frame) precede 24 future slots.
Not a causal model, continuous historical prefix, or trained memory mechanism.
The reference slot is stripped BEFORE VAE decode to isolate DiT conditioning.
"""
import torch


def prefix_rollout(model, *, prefix, future_noise, conditions, sigmas):
    clean_slots=prefix.shape[1]
    if clean_slots not in (2,25) or future_noise.shape[1] != 24:
        raise ValueError('Exactly2 sparse or25 historical clean slots and24 future slots')
    if prefix.shape[0:1]+prefix.shape[2:] != future_noise.shape[0:1]+future_noise.shape[2:]:
        raise ValueError('Prefix/noise geometry mismatch')
    current=torch.cat((prefix, future_noise),dim=1).clone()
    trace=[]; first_velocity=None
    for step,(sigma,next_sigma) in enumerate(zip(sigmas[:-1],sigmas[1:])):
        timestep=(sigma*1000).expand(current.shape[:2]).clone()
        timestep[:,:clean_slots]=0
        velocity,_=model(current,conditional_dict=conditions,timestep=timestep,
                         replace_first_timestep_and_noise_latents=True)
        if step==0: first_velocity=velocity[:,clean_slots:].detach().cpu()
        current=current+(next_sigma-sigma)*velocity
        current[:,:clean_slots]=prefix
        if not torch.isfinite(current).all(): raise ValueError('Nonfinite probe state')
        if not torch.equal(current[:,:clean_slots],prefix): raise RuntimeError('Prefix changed')
        trace.append({'step':step,'prefix_clamped':True,'zero_t_prefix':True,'clean_slots':clean_slots,
                      'state_dtype':str(current.dtype),'velocity_dtype':str(velocity.dtype)})
    return current[:,clean_slots-1:],trace,first_velocity


def cpu_contract(clean_slots=2):
    """Software-only: verifies clamping, equal-noise layout and slot removal."""
    future=torch.randn((1,24,1,2,2),generator=torch.Generator().manual_seed(109))
    current=torch.ones((1,1,1,2,2))
    seen=[]
    def fake(x,conditional_dict,timestep,**kwargs):
        assert (timestep[:,:clean_slots]==0).all()
        seen.append(x[:,:clean_slots].clone())
        return torch.ones_like(x),None
    prefix=torch.cat(((current*3).repeat(1,clean_slots-1,1,1,1),current),dim=1)
    output,trace,_=prefix_rollout(fake,prefix=prefix,future_noise=future,
        conditions={},sigmas=torch.tensor([1.,.5,0.]))
    assert output.shape[1]==25 and torch.equal(output[:,:1],current)
    assert all(torch.equal(x,prefix) for x in seen)
    assert torch.allclose(output[:,1:],future-1,atol=1e-6)
    return {'prefix_clamped':True,'current_first_output':True,'24_future_slots':True,
            'kind':'CPU fixture only, not model success'}
