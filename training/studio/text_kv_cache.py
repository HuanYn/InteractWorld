"""Opt-in, one-window CFG text K/V cache. No video/history state is cached.

Uses the actual frozen LoRA linear modules and norm_k, preserving their dtype,
math and full model call count. Default inference/training never installs the
provider. Callers must enter under no_grad after setting the final model state.
"""
from __future__ import annotations

import math
import torch

from training.demo.contracts import require

_HOOK = '_studio_text_kv_provider'


def _cross_attention_type():
    # Lazy import: importing this utility alone never loads optional CUDA code.
    from wan.modules.model import WanCrossAttention
    return WanCrossAttention


def _stamp(tensor):
    return (id(tensor), tensor._version, tuple(tensor.shape), tensor.dtype,
            tensor.device, tensor.requires_grad)


class CachedTextCFGFlow:
    """Context-managed TextCFGFlow equivalent with separate positive/negative KV.

    Model parameters must remain frozen/eval for the whole scope. Standard
    in-place mutations/load_state_dict, replacements, LoRA scale and precision
    changes are rejected. Unsupported unsafe writes through Tensor.data must not
    occur inside the scope; this is not a weight-content hashing mechanism.
    """

    def __init__(self, model, scale, negative):
        require(type(scale) in (int, float) and math.isfinite(scale) and scale >= 1,
                'text K/V cache requires finite CFG scale>=1')
        self.model, self.scale, self.negative = model, scale, negative
        self.forward_calls = 0
        self._opened = self._closed = False
        self._active_branch = None
        self._providers = []
        self._entries = {'positive': {}, 'negative': {}}
        self._counts = {name: {'hits': 0, 'misses': 0} for name in self._entries}
        self._binding = None
        self._state = None
        self._peak_bytes = self._bytes = 0
        self._layer_count = 0

    @property
    def stats(self):
        branches = {}
        for name, counts in self._counts.items():
            entries = self._entries[name]
            size = sum(value.numel() * value.element_size()
                       for _, values in entries.values() for value in values)
            branches[name] = dict(counts, entries=len(entries), active_entries=len(entries),
                cache_bytes=size, kv_pairs_computed=counts['misses'], kv_pairs_reused=counts['hits'],
                completed_linear_projection_calls=2 * counts['misses'], completed_norm_k_calls=counts['misses'])
        return dict(kind='scoped_cfg_text_cross_attention_kv_v1', scope='one_inference_window',
            layers=self._layer_count, forward_calls=self.forward_calls,
            branches=branches,
            cache_bytes=self._bytes, peak_cache_bytes=self._peak_bytes,
            closed=self._closed, video_history_cached=False, text_projection_cached=False)

    def _signature(self):
        require(not torch.is_grad_enabled() and not self.model.training,
                'text K/V cache requires eval and caller no_grad')
        modules = tuple((name, id(module), type(module), module.training, getattr(module, 'scaling', None))
                        for name, module in self.model.named_modules())
        parameters = tuple((name, _stamp(value)) for name, value in self.model.named_parameters())
        buffers = tuple((name, _stamp(value)) for name, value in self.model.named_buffers())
        require(not any(item[3] for item in modules) and not any(item[1][-1] for item in parameters),
                'text K/V cache requires all model modules frozen and eval')
        return modules, parameters, buffers, self.scale

    def __enter__(self):
        require(not self._opened and not self._closed, 'text K/V cache cannot reopen or nest itself')
        self._state = self._signature()
        cls = _cross_attention_type()
        layers = [(name, module) for name, module in self.model.named_modules() if isinstance(module, cls)]
        require(layers, 'no supported WanCrossAttention layers found')
        require(all(type(module) is cls for _, module in layers),
                'text K/V cache supports exact WanCrossAttention only; I2V/subclasses are unsupported')
        require(all(not hasattr(module, _HOOK) for _, module in layers), 'text K/V provider already installed')
        self._layer_count = len(layers)
        self._opened = True
        try:
            for name, module in layers:
                def provider(context, compute, name=name):
                    return self._get(name, context, compute)
                setattr(module, _HOOK, provider)
                self._providers.append((module, provider))
        except BaseException:
            self.close()
            raise
        return self

    def _get(self, name, context, compute):
        require(self._opened and not self._closed and self._active_branch in self._entries,
                'text K/V provider used outside its active CFG branch')
        require(not torch.is_grad_enabled(), 'text K/V cache requires caller no_grad')
        entries = self._entries[self._active_branch]
        signature = (tuple(context.shape), context.dtype, context.device)
        if name in entries:
            saved_signature, values = entries[name]
            require(signature == saved_signature, 'text K/V projected context shape/dtype/device changed')
            self._counts[self._active_branch]['hits'] += 1
            return values
        values = compute()
        require(all(torch.is_tensor(value) and not value.requires_grad for value in values),
                'text K/V projections unexpectedly require gradients')
        entries[name] = signature, values
        self._counts[self._active_branch]['misses'] += 1
        self._bytes += sum(value.numel() * value.element_size() for value in values)
        self._peak_bytes = max(self._peak_bytes, self._bytes)
        return values

    def _check(self, conditional_dict):
        require(self._opened and not self._closed, 'text K/V cache is not active or is closed')
        require(self._active_branch is None, 'text K/V cache does not support reentrant model calls')
        require(self._signature() == self._state, 'text K/V model weights/scaling/dtype/device/mode changed')
        require(all(getattr(module, _HOOK, None) is provider for module, provider in self._providers),
                'text K/V provider binding changed')
        prompt = conditional_dict.get('prompt_embeds')
        require(isinstance(prompt, (tuple, list)) and prompt and all(torch.is_tensor(t) for t in prompt),
                'text K/V cache needs explicit text tensors')
        require(conditional_dict.get('clip_fea') is None,
                'text K/V cache does not support appended image-token conditioning')
        negative = ()
        if self.scale != 1:
            require(torch.is_tensor(self.negative), 'CFG requires a verified negative embedding')
            negative = (_stamp(self.negative),)
        devices = sorted({value.device.type for value in self.model.parameters()})
        precision = tuple((name, torch.is_autocast_enabled(name), torch.get_autocast_dtype(name)) for name in devices)
        binding = tuple(_stamp(value) for value in prompt), negative, precision
        if self._binding is None:
            self._binding = binding
        else:
            require(binding == self._binding, 'text K/V prompt or negative binding changed')

    def _forward(self, branch, args, conditional_dict, kwargs):
        self._active_branch = branch
        try:
            self.forward_calls += 1
            return self.model(*args, conditional_dict=conditional_dict, **kwargs)
        finally:
            self._active_branch = None

    def __call__(self, *args, conditional_dict, **kwargs):
        self._check(conditional_dict)
        require(all(kwargs.get(key) is None for key in ('kv_cache', 'crossattn_cache', 'inference_conditioning_cache')),
                'text K/V cache cannot be combined with other cache paths')
        positive = self._forward('positive', args, conditional_dict, kwargs)
        if self.scale == 1:
            return positive
        # Recheck before the negative branch so an interrupted/custom forward
        # cannot mutate model/input bindings and silently reuse old projections.
        self._check(conditional_dict)
        negative_conditions = dict(conditional_dict, prompt_embeds=[self.negative])
        negative = self._forward('negative', args, negative_conditions, kwargs)
        return negative[0] + self.scale * (positive[0] - negative[0]), None

    def close(self):
        for module, provider in self._providers:
            if getattr(module, _HOOK, None) is provider:
                delattr(module, _HOOK)
        self._providers.clear()
        for entries in self._entries.values():
            entries.clear()
        self._bytes = 0
        self._active_branch = self._binding = self._state = None
        self.model = self.negative = None
        self._closed = True

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
