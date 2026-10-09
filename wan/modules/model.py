# InterActWorld modifications: action/reference conditioning and causal-training integration.
# InterActWorld reproduction: adapted from ABot-World; see NOTICE.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from utils.action_alignment import (
    expand_frame_conditioning_to_tokens,
    inject_action_features,
)
from utils.controlled_action import inject_controlled_action_features

from .attention import attention

__all__ = ['WanModel']


class _WindowConditioningCache:
    """Explicitly scoped, immutable-condition cache for one inference window.

    Nothing is attached to the model. Exiting the context invalidates the object
    and drops its GPU tensors, including on an interrupted/failed Euler rollout.
    """

    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self.signature = None
        self.values = {}
        self._check_mode(owner)

    @staticmethod
    def _check_mode(owner):
        if owner.training or torch.is_grad_enabled():
            raise ValueError('conditioning cache requires eval and the caller\'s no_grad context')

    @staticmethod
    def _stamp(tensor):
        return (id(tensor), tensor._version, tuple(tensor.shape), tensor.dtype, tensor.device)

    def validate(self, owner, x, context, actions):
        if self.closed or owner is not self.owner:
            raise ValueError('conditioning cache is closed or belongs to another model')
        self._check_mode(owner)
        modules = [owner.text_embedding]
        if actions is not None:
            modules.append(owner.act_control_adapter)
        if any(module.training or any(p.requires_grad for p in module.parameters()) for module in modules):
            raise ValueError('cached projections must stay frozen and in eval mode')
        inputs = tuple(self._stamp(value) for value in context)
        action_inputs = None if actions is None else tuple(self._stamp(value) for value in actions)
        # No noise identity/version: noise and timestep must change during Euler.
        videos = [x] if torch.is_tensor(x) else x
        geometry = tuple((tuple(value.shape), value.dtype, value.device) for value in videos)
        device_type = videos[0].device.type
        autocast = torch.is_autocast_enabled(device_type)
        precision = (autocast, torch.get_autocast_dtype(device_type) if autocast else None)
        parameters = tuple(self._stamp(p) for module in modules for p in module.parameters())
        signature = (inputs, action_inputs, geometry, precision, parameters, owner.text_len)
        if self.signature is None:
            self.signature = signature
        elif signature != self.signature:
            raise ValueError('conditioning cache inputs, geometry, precision or weights changed')

    def get(self, key, compute):
        if self.closed or self.signature is None:
            raise ValueError('conditioning cache is not active and validated')
        if key not in self.values:
            self.values[key] = compute()
        return self.values[key]

    def close(self):
        self.values.clear()
        self.signature = None
        self.owner = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise ValueError('conditioning cache is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output)

class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)

class GanAttentionBlock(nn.Module):

    def __init__(self,
                 dim=1536,
                 ffn_dim=8192,
                 num_heads=12,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        # self.norm1 = WanLayerNorm(dim, eps)
        # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        #   eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.cross_attn = WanGanCrossAttention(dim, num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        context,
        # seq_lens,
        # grid_sizes,
        # freqs,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        # e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # # self-attention
        # y = self.self_attn(
        #     self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        #     freqs)
        # # with amp.autocast(dtype=torch.float32):
        # x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context):
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token  # * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            # x = x + y * e[5]
            return y

        x = cross_attn_ffn(x, context)
        return x
class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            backend='auto')

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = attention(qq, kk, vv, backend='auto')

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return super().forward(x.float()).type_as(x)
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs).type_as(v),
            k=rope_apply(k, grid_sizes, freqs).type_as(v),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            backend='auto')

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


    def forward_reference(self, x, seq_lens, grid_sizes, freqs, ref_tokens, ref_freqs):
        """Unpadded video reads image memory; image queries read only images.

        Both streams use these exact existing projections (including any LoRA).
        WanModel selects valid image slots before any biased projection.
        """
        from training.reference_memory import apply_reference_rope, split_reference_attention

        def project(tokens):
            shape = (*tokens.shape[:2], self.num_heads, self.head_dim)
            return (self.norm_q(self.q(tokens)).view(shape),
                    self.norm_k(self.k(tokens)).view(shape), self.v(tokens).view(shape))

        q, k, v = project(x)
        qr, kr, vr = project(ref_tokens)
        result = split_reference_attention(
            rope_apply(q, grid_sizes, freqs).type_as(v),
            rope_apply(k, grid_sizes, freqs).type_as(v), v,
            q_ref=apply_reference_rope(qr, ref_freqs),
            k_ref=apply_reference_rope(kr, ref_freqs), v_ref=vr,
            ref_valid=torch.ones(ref_tokens.shape[:2], device=x.device, dtype=torch.bool),
        )
        return self.o(result.video.flatten(2)), self.o(result.reference.flatten(2))


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # Studio may install a window-scoped, branch-isolated text-only cache.
        # No provider is installed by default or by the training path.
        provider = getattr(self, '_studio_text_kv_provider', None)
        if provider is None:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        else:
            def project_text_kv():
                return (self.norm_k(self.k(context)).view(b, -1, n, d),
                        self.v(context).view(b, -1, n, d))
            k, v = provider(context, project_text_kv)

        # compute attention
        x = attention(q, k, v, k_lens=context_lens, backend='auto')

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanI2VCrossAttention(WanCrossAttention):
    def __init__(self,
                dim,
                num_heads,
                window_size=(-1, -1),
                qk_norm=True,
                eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = attention(q, k_img, v_img, k_lens=None, backend='auto')
        # compute attention
        x = attention(q, k, v, k_lens=context_lens, backend='auto')

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 cross_attn_type="t2v_cross_attn"):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        #     e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attentionm
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        #     x = x + y * e[2].squeeze(2)
        x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            # with torch.amp.autocast('cuda', dtype=torch.float32):
            #     x = x + y * e[5].squeeze(2)
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

    def forward_reference(self, x, ref_tokens, ref_e, ref_freqs, e, seq_lens,
                          grid_sizes, freqs, context, context_lens):
        """Image-only reference stream sharing self-attention and FFN weights.

        Reference time modulation is the model's actual t=0 embedding. Reference
        tokens skip text cross-attention, receive no action/noise, and never read
        video tokens. This is an explicit local design, not an official ABot
        equivalence claim. Only the video stream is decoded into flow targets.
        """
        ev = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        er = (self.modulation.unsqueeze(0) + ref_e).chunk(6, dim=2)
        y, yr = self.self_attn.forward_reference(
            self.norm1(x) * (1 + ev[1].squeeze(2)) + ev[0].squeeze(2),
            seq_lens, grid_sizes, freqs,
            self.norm1(ref_tokens) * (1 + er[1].squeeze(2)) + er[0].squeeze(2),
            ref_freqs,
        )
        x = x + y * ev[2].squeeze(2)
        ref_tokens = ref_tokens + yr * er[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        x = x + self.ffn(self.norm2(x) * (1 + ev[4].squeeze(2)) + ev[3].squeeze(2)) * ev[5].squeeze(2)
        ref_tokens = ref_tokens + self.ffn(
            self.norm2(ref_tokens) * (1 + er[4].squeeze(2)) + er[3].squeeze(2)
        ) * er[5].squeeze(2)
        return x, ref_tokens


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        #     e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        #     x = (
        #         self.head(
        #             self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))

        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))


        # if e.dim() > 2:
        #     e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        #     e = [e.squeeze(2) for e in e]
        # else:
        #     e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        
        # x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 act_control_in_dim=32,
                 downscale_factor_control_adapter=16):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v', 'ci2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # CI2V conditions on the clean leading latent frame and actions. It has
        # no CLIP image-token prefix, matching the released causal model.
        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, cross_attn_type)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if model_type == 'ci2v':
            self.act_control_adapter = SimpleAdapter(act_control_in_dim, self.dim, 
                                                    kernel_size=self.patch_size[1:], stride=self.patch_size[1:],
                                                    downscale_factor=downscale_factor_control_adapter)
            self.act_control_adapter.requires_grad_(False)
        
        if model_type in ['i2v', 'ti2v', 'ci2v']:
            self.img_emb = MLPProj(1280, dim)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func=None,
    ):
        """Implement the current Diffusers gradient-checkpointing hook."""
        self.gradient_checkpointing = enable
        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func

    def inference_conditioning_cache(self):
        """Create an opt-in cache; the caller must use/close it within no_grad."""
        return _WindowConditioningCache(self)

    def _prepare_reference_memory(self, ref_latents, ref_mask, videos, seq_len,
                                  y, clip_fea, cache):
        """Select valid image slots before patch projection; no trainable state.

        Opt-in schema: floating [B,K,C,1,H,W] and bool [B,K]. Active references
        currently require B=1, global attention, exact unpadded patch grids, and
        no extra image/CLIP/conditioning-cache path. None/empty/all-invalid
        references return to the original forward without reference arithmetic.
        """
        def require(condition, message):
            if not condition:
                raise ValueError(message)

        if ref_latents is None:
            require(ref_mask is None, 'ref_mask requires ref_latents')
            return None
        require(torch.is_tensor(ref_latents) and ref_latents.ndim == 6
                and ref_latents.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                and ref_latents.device.type in ('cpu', 'cuda') and ref_latents.layout == torch.strided,
                'ref_latents must be floating dense CPU/CUDA [B,K,C,1,H,W]')
        b, slots, channels, frames, height, width = ref_latents.shape
        require(b > 0 and channels > 0 and frames == 1 and height > 0 and width > 0,
                'ref_latents require positive dimensions and one image frame per slot')
        require(torch.is_tensor(ref_mask) and ref_mask.dtype == torch.bool
                and tuple(ref_mask.shape) == (b, slots) and ref_mask.layout == torch.strided
                and ref_mask.device == ref_latents.device,
                'ref_mask must be dense bool [B,K] on the reference device')
        require(len(videos) == b, 'reference and video batch sizes differ')
        if not bool(ref_mask.any()):
            return None
        require(b == 1, 'active reference memory currently supports B=1 only')
        require(y is None and clip_fea is None and cache is None,
                'active reference memory does not support y, clip_fea or conditioning cache')
        require(tuple(self.window_size) == (-1, -1), 'reference memory requires global attention')
        video = videos[0]
        require(torch.is_tensor(video) and video.ndim == 4 and video.layout == torch.strided,
                'reference memory requires dense video [C,T,H,W]')
        require(channels == self.in_dim == video.shape[0]
                and ref_latents.device == video.device and ref_latents.dtype == video.dtype,
                'reference/video channels, device and dtype must agree')
        require(self.patch_size[0] == 1 and all(d > 0 and d % p == 0
                for shape in (video.shape[1:], ref_latents.shape[3:])
                for d, p in zip(shape, self.patch_size)),
                'reference memory requires exact patch-divisible video/image grids and temporal patch=1')
        grid = tuple(d // p for d, p in zip(video.shape[1:], self.patch_size))
        require(seq_len is None or (type(seq_len) is int and seq_len == math.prod(grid)),
                'active reference memory requires exact seq_len without video padding')
        require(max(grid) <= self.freqs.shape[0], 'video grid exceeds the model RoPE table')
        valid_slots = torch.nonzero(ref_mask[0], as_tuple=False).flatten()
        # Invalid NaNs must never enter a convolution or a biased projection.
        images = ref_latents[0].index_select(0, valid_slots)
        require(bool(torch.isfinite(images).all()), 'valid reference images must be finite')
        features = self.patch_embedding(images)
        ref_grid = tuple(features.shape[2:])
        tokens_per_slot = math.prod(ref_grid)
        tokens = features.flatten(2).transpose(1, 2).reshape(1, -1, self.dim)
        from training.reference_memory import build_reference_rope_frequencies
        frequencies = build_reference_rope_frequencies(
            self.freqs, num_slots=slots, tokens_per_slot=tokens_per_slot,
            ref_grid=ref_grid, device=video.device,
        ).reshape(slots, tokens_per_slot, 1, -1)
        # Keep original slot coordinates even when an earlier slot is invalid.
        frequencies = frequencies.index_select(0, valid_slots).reshape(-1, 1, frequencies.shape[-1])
        return tokens, frequencies

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        clip_fea=None,
        act_context=None,
        act_context_scale=1.0,
        inference_conditioning_cache=None,
        ref_latents=None,
        ref_mask=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            ref_latents / ref_mask (optional):
                Explicit image-memory opt-in: [B,K,C,1,H,W] and bool [B,K].
                Valid references currently require B=1 and exact unpadded global
                video attention; y/clip_fea/conditioning caches are unsupported.
                References are image-only, unnoised, action/text-independent.
                They share backbone weights and gradients, but are not decoded.
                With active references, seq_len=None infers the exact video
                patch-grid length for either a batched tensor or a one-item list.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        cache = inference_conditioning_cache
        if cache is not None:
            if not isinstance(cache, _WindowConditioningCache):
                raise ValueError('expected a scoped inference conditioning cache')
            cache.validate(self, x, context, act_context)
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        reference = self._prepare_reference_memory(
            ref_latents, ref_mask, x, seq_len, y, clip_fea, cache)
        ref_tokens, ref_freqs = reference if reference is not None else (None, None)

        if seq_len is None:
            if reference is not None:
                # The reference validator already checked B=1 and divisibility.
                # The real wrapper leaves seq_len=None; x[0] works for its
                # [B,C,T,H,W] tensor and the model's one-video list interface.
                seq_len = math.prod(d // p for d, p in zip(x[0].shape[1:], self.patch_size))
            else:
                seq_len = x.shape[2] * x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2] * self.patch_size[0])


        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        if act_context is not None and self.act_control_adapter is not None:
            x_new = []
            if cache is None:
                y_action = [self.act_control_adapter(u.unsqueeze(0)) for u in act_context]
            else:
                y_action = cache.get('action_features', lambda: [
                    self.act_control_adapter(u.unsqueeze(0)) for u in act_context])
            for u, v in zip(x, y_action):
                t_f = u.shape[2]
                c_f = v.shape[2]
                if getattr(self, 'controlled_action_enabled', False):
                    u = inject_controlled_action_features(self, u, v)
                elif t_f > c_f:
                    u = inject_action_features(u, v, scale=act_context_scale)
                else:
                    u = u + v * act_context_scale
                x_new.append(u)
            x = x_new

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), 1)
        bt, Lt = t.shape
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, Lt)).type_as(x))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        if Lt > 1:
            e = expand_frame_conditioning_to_tokens(e, grid_sizes, seq_len)
            e0 = expand_frame_conditioning_to_tokens(e0, grid_sizes, seq_len)

        if reference is not None:
            # sin/cos(t=0) and learned biases are not a fabricated zero vector.
            ref_time = self.time_embedding(sinusoidal_embedding_1d(
                self.freq_dim, torch.zeros(1, device=device)).type_as(ref_tokens))
            ref_e0 = self.time_projection(ref_time).unflatten(1, (6, self.dim)).unsqueeze(1)

        # context
        context_lens = None
        if cache is None:
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
        else:
            context = cache.get('text_projection', lambda: self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ])))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        
        def create_custom_forward(module):
            def custom_forward(*inputs, **kw):
                return module(*inputs, **kw)
            return custom_forward


        for block in self.blocks:
            if reference is not None:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x, ref_tokens = torch.utils.checkpoint.checkpoint(
                        block.forward_reference, x, ref_tokens, ref_e0, ref_freqs,
                        **kwargs, use_reentrant=False,
                    )
                else:
                    x, ref_tokens = block.forward_reference(
                        x, ref_tokens, ref_e0, ref_freqs, **kwargs)
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs, use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor=8, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * downscale_factor*downscale_factor, out_dim, kernel_size=kernel_size, stride=stride, padding=0) #kernel_size=[2,2]

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x) #(h, w) → (h//8, w//8), c → c×64

        # Convolution operation
        x_conv = self.conv(x_unshuffled)#b*f,c*64, h//8,w//8 -> bs*f, out_dim, h//16, w//16 (latents对齐空间)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3)) #bs,f, out_dim, h//16, w//16

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4) #bs, out_dim, f, h//16, w//16

        return out
