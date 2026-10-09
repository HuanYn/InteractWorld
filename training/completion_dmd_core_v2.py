"""Actual distribution-matching math and self-history orchestration.

This independent module imports no Wan model or launcher.  Native callbacks
remain responsible for their attention masks; passing these CPU checks is not
a 24GB resource qualification.  A frozen real score and a trained fake score
are different models, not endpoint labels supplied by a ground-truth video.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import torch

from training import completion_ode_core_v2 as ode

NAMESPACE = 'completion_longforcing_selfrollout_seed42_v2'
SOURCE_REVISION = '8202903297918bcd12810fc4230a4db7fe065e64'
SOURCE_URL = 'https://github.com/thu-ml/Causal-Forcing/blob/'+SOURCE_REVISION+'/model/dmd.py'
SOURCE_FILE_SHA256 = '6f62375fef268270005612bf8d831199738fc59c72d146fc56392183e09d1eef'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(torch.is_tensor(value) and value.is_floating_point()
            and bool(torch.isfinite(value).all()), name+' must be a finite floating tensor')


@dataclass(frozen=True)
class DMDConfig:
    arm: str
    rollout_blocks: int = 20
    score_blocks: int = 8
    short_window_blocks: int = 4
    gradient_latents: int = 3
    real_cfg: float = 5.
    fake_cfg: float = 1.
    weight: float = 1.
    normalizer_epsilon: float = 1e-6
    sigma_shift: float = 5.
    min_sigma: float = .02
    max_sigma: float = .98
    fake_warmup_updates: int = 2

    def validate(self):
        require(self.arm in ('short', 'long'), 'explicit short or long DMD arm required')
        expected = DMDConfig(self.arm)
        require(self == expected, 'fixed DMD horizon/guidance/sigma/weight contract changed')

    def record(self):
        self.validate()
        return dict(schema='completion_selfrollout_dmd_contract_v2', **self.__dict__,
            student_sampler='four_positive_endpoint_predictions_three_independent_flow_renoises',
            rollout_seconds=15., score_seconds=6., score_windows=2 if self.arm == 'short' else 1,
            real_attention='bidirectional', fake_attention='bidirectional',
            generator_attention='causal_same_own_clean_history',
            real_guidance_definition='negative+5*(positive-negative)',
            fake_guidance_definition='positive_only_CFG1',
            normalizer='future_window_mean_abs_generated_minus_real_endpoint_clamped_1e-6',
            generator_gradient='fake_endpoint_minus_real_endpoint_divided_by_normalizer',
            fake_objective='positive_branch_future_only_flow_MSE_noise_minus_own_sample',
            generator_objective='detached_score_difference_surrogate_not_endpoint_regression',
            gradient_scope='only_last_three_latents; prior_own_history_detached',
            nonfinite_policy='raise_not_nan_to_num', source_url=SOURCE_URL,
            source_revision=SOURCE_REVISION, source_file_sha256=SOURCE_FILE_SHA256,
            project_adaptation_not_official_recipe=True,
            future_ground_truth_supplied=False)


def stream_seed(seed: int, update: int, purpose: str) -> int:
    require(type(seed) is int and seed >= 0 and type(update) is int and update >= 0
            and isinstance(purpose, str) and purpose, 'explicit independent deterministic stream required')
    digest = hashlib.sha256(f'{NAMESPACE}|{seed}|{update}|{purpose}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2**63-1)


def independent_noise(shape, *, seed, update, purpose, device='cpu'):
    require(len(shape) == 5 and all(type(n) is int and n > 0 for n in shape), 'B,T,C,H,W noise shape required')
    actual = stream_seed(seed, update, purpose)
    generator = torch.Generator(device='cpu').manual_seed(actual)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device), actual


def score_level(*, seed, update, purpose, device='cpu', config=None):
    """Uniform base sigma -> shift5 -> [.02,.98] -> native BF16 sigma/t.

    One scalar is shared across all scored frames/windows of this update.
    Generator-score and fake-training levels have independent fixed streams.
    """
    config = DMDConfig('long') if config is None else config
    config.validate()
    actual = stream_seed(seed, update, purpose+'-sigma')
    generator = torch.Generator(device='cpu').manual_seed(actual)
    base = torch.rand((), generator=generator, dtype=torch.float32)
    shifted = (config.sigma_shift*base / (1+(config.sigma_shift-1)*base)).clamp(config.min_sigma, config.max_sigma)
    sigma = shifted.to(dtype=torch.bfloat16, device=device)
    timestep = sigma*1000.
    return sigma, timestep, dict(seed=actual, base_sigma=float(base), shifted_sigma=float(shifted),
        actual_sigma=float(sigma), actual_model_timestep=float(timestep), shift=5.)


def repeat_actions(actions, blocks=20):
    finite(actions, 'actions')
    require(actions.ndim == 3 and actions.shape[1:] == (96, 8)
            and bool(((actions == 0) | (actions == 1)).all()) and type(blocks) is int and blocks > 0,
            'one original binary 96x8 action trajectory required')
    rows = blocks*12
    return actions.repeat(1, (rows+95)//96, 1)[:, :rows].clone()


@dataclass(frozen=True)
class ScoreWindow:
    start_block: int
    stop_block: int
    video: torch.Tensor
    actions: torch.Tensor

    @property
    def tail_slice(self):
        # Relative to the common final eight blocks, excluding its anchor.
        return slice((self.start_block-12)*3, (self.stop_block-12)*3)


def score_windows(video, actions, config):
    config.validate(); finite(video, 'own rollout'); finite(actions, 'rollout actions')
    require(video.ndim == 5 and video.shape[1] == 61
            and tuple(actions.shape) == (video.shape[0], 240, 8), 'exact20-block own rollout/actions required')
    boundaries = ((12,16),(16,20)) if config.arm == 'short' else ((12,20),)
    result=[]
    for start, stop in boundaries:
        # Index3*start is the previous block's last latent, never a GT image.
        result.append(ScoreWindow(start, stop, video[:, start*3:stop*3+1].detach().clone(),
            actions[:, start*12:stop*12].detach().clone()))
    return tuple(result)


def _score_input(window, tail_noise, sigma):
    noise = tail_noise[:, window.tail_slice]
    future = window.video[:, 1:].float()
    require(noise.shape == future.shape, 'score noise/window mismatch')
    noisy_future = ((1-sigma.float())*future + sigma.float()*noise).to(window.video.dtype)
    return torch.cat((window.video[:, :1], noisy_future), dim=1)


def _velocity(callback, *, role, noisy, window, sigma, timestep, branch, conditions, grad):
    snapshot = noisy.detach().clone()
    anchor = noisy[:, :1].detach().clone()
    value = callback(role=role, noisy_video=noisy, actions=window.actions, sigma=sigma,
        timestep=timestep, branch=branch, conditions=conditions,
        attention_mode='bidirectional', gradient_enabled=grad,
        absolute_start_latent=window.start_block*3)
    finite(value, role+' score velocity')
    require(value.shape == noisy.shape and value.dtype == noisy.dtype, 'native score shape/dtype changed')
    require(torch.equal(noisy.detach(), snapshot) and torch.equal(noisy[:, :1].detach(), anchor),
        'score callback mutated its fixed anchor or noised future')
    return value


@torch.no_grad()
def distribution_gradient(callback, *, video, actions, positive, negative, config, seed, update):
    """Normalized DMD score difference, no gradient in either score network.

    Return all24 scored future gradients for audit; only their final3 entries
    become generator gradients. Short/long use identical underlying tail noise.
    """
    windows = score_windows(video, actions, config)
    noise, noise_seed = independent_noise((video.shape[0],24,*video.shape[2:]), seed=seed,
        update=update, purpose='generator-score-noise', device=video.device)
    sigma,timestep,level = score_level(seed=seed, update=update, purpose='generator-score',
        device=video.device, config=config)
    gradients, records, forwards = [], [], 0
    for window in windows:
        noisy = _score_input(window, noise, sigma)
        fake = _velocity(callback, role='fake', noisy=noisy, window=window, sigma=sigma,
            timestep=timestep, branch='positive', conditions=positive, grad=False); forwards+=1
        real_p = _velocity(callback, role='real', noisy=noisy, window=window, sigma=sigma,
            timestep=timestep, branch='positive', conditions=positive, grad=False); forwards+=1
        real_n = _velocity(callback, role='real', noisy=noisy, window=window, sigma=sigma,
            timestep=timestep, branch='negative', conditions=negative, grad=False); forwards+=1
        real = real_n.float()+config.real_cfg*(real_p.float()-real_n.float())
        fake_x0 = noisy[:,1:].float()-sigma.float()*fake[:,1:].float()
        real_x0 = noisy[:,1:].float()-sigma.float()*real[:,1:]
        normalizer = (window.video[:,1:].float()-real_x0).abs().mean(dim=(1,2,3,4),keepdim=True)
        normalizer = normalizer.clamp_min(config.normalizer_epsilon)
        gradient = (fake_x0-real_x0)/normalizer
        finite(gradient, 'DMD score difference')
        gradients.append(gradient)
        records.append(dict(start_block=window.start_block, stop_block=window.stop_block,
            scored_future_latents=window.video.shape[1]-1, normalizer=normalizer.flatten().tolist(),
            gradient_rms=float(gradient.square().mean().sqrt()), fixed_clean_anchor=True,
            real_cfg=5., fake_cfg=1.))
    return torch.cat(gradients,dim=1).detach(), dict(level=level, noise_seed=noise_seed,
        score_forwards=forwards, windows=records, score_network_gradients=False,
        generated_history_only=True)


def fake_flow_loss(callback, *, video, actions, positive, config, seed, update):
    """Train actual fake-score flow on detached student samples, not GT targets."""
    windows=score_windows(video,actions,config)
    noise,noise_seed=independent_noise((video.shape[0],24,*video.shape[2:]),seed=seed,
        update=update,purpose='fake-training-noise',device=video.device)
    sigma,timestep,level=score_level(seed=seed,update=update,purpose='fake-training',device=video.device,config=config)
    losses=[]
    for window in windows:
        noisy=_score_input(window,noise,sigma)
        predicted=_velocity(callback,role='fake',noisy=noisy,window=window,sigma=sigma,
            timestep=timestep,branch='positive',conditions=positive,grad=True)
        target=noise[:,window.tail_slice]-window.video[:,1:].float()
        loss=(predicted[:,1:].float()-target).square().mean()
        finite(loss,'fake flow objective'); losses.append(loss)
    return torch.stack(losses).mean(),dict(level=level,noise_seed=noise_seed,
        score_forwards=len(windows),scored_future_latents=24,loss_kind='own_sample_future_flow_MSE',
        anchor_excluded=True,generator_target_detached=True)


def replay_last_block(callback, *, video, actions, positive, seed, config):
    """Differentiable four-endpoint replay of block19; only prior own past read.

    Callback must implement causal attention; history includes exactly first+
    19completed blocks. No KV graph or future clean can enter this API.
    """
    config.validate(); score_windows(video,actions,config)
    require(torch.is_grad_enabled(),'generator replay must retain a real autograd graph')
    history=video[:,:58].detach().clone()
    initial_history=history.clone()
    streams=ode.fixed_noise_streams(seed=seed,block_index=19,shape=(video.shape[0],3,*video.shape[2:]),namespace=NAMESPACE)
    state=streams.initial.to(device=video.device,dtype=video.dtype)
    grid=ode.teacher_grid()
    for round_index,index in enumerate(ode.GRID_INDICES[:-1]):
        sigma,timestep=grid.levels(state,index)
        velocity=callback(noisy_block=state,history=history,actions=actions,
            action_block=actions[:,228:240],sigma=sigma,timestep=timestep,block_index=19,
            branch='positive',cfg_scale=1.,conditions=positive)
        finite(velocity,'generator replay velocity')
        require(velocity.shape==state.shape and velocity.dtype==state.dtype,
            'native generator replay shape/dtype changed')
        require(torch.equal(history,initial_history),'generator modified detached own past')
        endpoint=state.float()-sigma.float().reshape(-1,1,1,1,1)*velocity.float()
        if round_index==3:
            return endpoint.to(video.dtype),dict(denoising_forwards=4,block_index=19,
                history_latents=58,gradient_latents=3,noise_seeds=list(streams.seeds),cfg_scale=1.)
        next_sigma=grid.sigmas[ode.GRID_INDICES[round_index+1]]
        state=((1-next_sigma)*endpoint+next_sigma*streams.renoise[round_index].to(video.device)).to(video.dtype)
    raise AssertionError('unreachable')


def generator_surrogate(replayed, gradient, *, config):
    config.validate(); finite(replayed,'replayed generator endpoint');finite(gradient,'detached DMD gradient')
    require(replayed.requires_grad and replayed.shape[1]==3 and gradient.shape[1]==24
        and gradient.shape[:1]+gradient.shape[2:]==replayed.shape[:1]+replayed.shape[2:]
        and not gradient.requires_grad,'only last3 generator gradients, detached score required')
    target=(replayed.float()-gradient[:,-3:]).detach()
    return config.weight*.5*(replayed.float()-target).square().mean()


def own_rollout(backend, *, first, actions, positive, seed, config):
    config.validate()
    return ode.sample_causal_clip(backend,first=first,actions=actions,positive_conditions=positive,
        seed=seed,blocks=20,noise_namespace=NAMESPACE)


class StateBanks:
    """One shared backbone; distinct CPU parameter/native-optimizer snapshots.

    Callers must finish backward before ``capture``/``activate``. A live graph
    is explicitly owned by a role; switching then raises, rather than changing
    parameters underneath checkpoint recomputation. Frozen nonbank parameters
    are never copied/optimized by this class.
    """
    def __init__(self, model, initial, *, names=None):
        finite_values=list(initial.values())
        require(finite_values and all(v.dtype==torch.float32 for v in finite_values),'fullFP32 LoRA bank required')
        self.model=model; self.named=dict(model.named_parameters())
        self.names=tuple(sorted(initial if names is None else names))
        require(set(self.names)==set(initial) and all(n in self.named and self.named[n].shape==initial[n].shape for n in self.names),
            'exact named bank parameter mapping required')
        self.banks={role:{n:initial[n].detach().cpu().clone() for n in self.names} for role in ('generator','fake','real')}
        for n in self.names:
            if '.lora_b.' in n:
                self.banks['real'][n].zero_()
        require(any('.lora_b.' in n for n in self.names),'real teacher requires explicit zero-B LoRA bypass')
        self.optimizer_states={'generator':None,'fake':None}
        self.current=None;self.live_graph=None;self.switches=0

    def activate(self,role):
        require(role in self.banks and self.live_graph is None,'cannot switch role with a live autograd graph')
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n in self.names:
                self.named[n].copy_(self.banks[role][n]);self.named[n].requires_grad_(role!='real')
        self.current=role;self.switches+=1

    def begin_graph(self,role):
        require(role==self.current and role!='real' and self.live_graph is None,'graph owner must be active trainable bank')
        self.live_graph=role

    def end_graph(self,role):
        require(self.live_graph==role,'backward completion role mismatch');self.live_graph=None

    def capture(self,role,optimizer=None):
        require(role==self.current and role!='real' and self.live_graph is None,'finish backward before capturing native state')
        for n in self.names:
            finite(self.named[n],n);self.banks[role][n]=self.named[n].detach().cpu().clone()
        if optimizer is not None:
            self.optimizer_states[role]=cpu_tree(optimizer.state_dict())

    def state_dict(self):
        require(self.live_graph is None,'cannot checkpoint live graph')
        return dict(schema='completion_dmd_shared_backbone_banks_v2',names=list(self.names),
            banks=cpu_tree(self.banks),optimizer_states=cpu_tree(self.optimizer_states),switches=self.switches)

    def load_state_dict(self,value):
        require(value.get('schema')=='completion_dmd_shared_backbone_banks_v2' and value.get('names')==list(self.names)
            and set(value.get('banks',{}))=={'generator','fake','real'}
            and set(value.get('optimizer_states',{}))=={'generator','fake'},'full independent bank state required')
        for role in self.banks:
            require(set(value['banks'][role])==set(self.names),'bank name map changed')
            for n in self.names:
                v=value['banks'][role][n];finite(v,n)
                require(v.dtype==torch.float32 and v.shape==self.named[n].shape and v.device.type=='cpu','CPUFP32 bank contract changed')
                if role=='real':
                    require(torch.equal(v,self.banks['real'][n]),'frozen real teacher bank changed')
        self.banks=cpu_tree(value['banks']);self.optimizer_states=cpu_tree(value['optimizer_states'])
        self.switches=int(value['switches']);self.current=None


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value
