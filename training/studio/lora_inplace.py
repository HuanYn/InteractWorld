"""Scoped unit-scale frozen LoRA addition; default model/training stay unchanged.

All original base/A/B projections still run. Only the newly allocated base
output is changed in place, avoiding multiply-by-one and a separate add output.
This is not weight merging, quantization, a rank change, or a speed claim.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from training.demo.contracts import require
from training.models.lora import LoRALinear


def _parameter_stamp(parameter):
    return (id(parameter), parameter._version, tuple(parameter.shape),
            parameter.dtype, parameter.device, parameter.requires_grad)


def _layer_state(module):
    require(type(module) is LoRALinear, 'inplace LoRA requires exact LoRALinear type')
    require(type(module.scaling) in (int, float) and module.scaling == 1,
            'inplace LoRA requires exact unit scaling')
    require(type(module.dropout) is nn.Dropout and module.dropout.p == 0,
            'inplace LoRA requires zero dropout')
    children = (module.base, module.lora_a, module.lora_b, module.dropout)
    require(all(type(layer) is nn.Linear for layer in children[:3]),
            'inplace LoRA requires unmodified exact Linear projections')
    require(not module.training and all(not layer.training for layer in children)
            and all(not parameter.requires_grad for parameter in module.parameters()),
            'inplace LoRA requires eval and frozen parameters')
    require(all('forward' not in layer.__dict__ and not layer._forward_hooks
                and not layer._forward_pre_hooks for layer in children),
            'inplace LoRA does not support overridden/hooked child projections')
    return (tuple(id(layer) for layer in children), module.scaling, module.rank, module.alpha,
            tuple((name, _parameter_stamp(value)) for name, value in module.named_parameters()))


class InplaceUnitScaleLoRA:
    """Use under caller no_grad, after freezing/eval and selecting LoRA strength.

    Parameter stamps reject standard in-place updates, replacements and dtype/
    device changes. Unsafe Tensor.data writes are outside this scope's contract;
    no weight hashing or copying is performed.
    """

    def __init__(self, model):
        self.model = model
        self._installed = []
        self._states = {}
        self._counts = {}
        self._opened = self._closed = False

    @property
    def stats(self):
        return dict(kind='frozen_unit_scale_lora_inplace_add_v1', modules_patched=len(self._counts),
            active_modules=len(self._installed), total_calls=sum(v['calls'] for v in self._counts.values()),
            completed_calls=sum(v['completed_calls'] for v in self._counts.values()),
            modules={name: dict(value) for name, value in self._counts.items()}, closed=self._closed,
            merged_weights=False, quantized=False, rank_changed=False, parameters_modified=False,
            inputs_modified=False)

    def __enter__(self):
        require(not self._opened and not self._closed, 'inplace LoRA context cannot reopen or nest itself')
        require(not torch.is_grad_enabled(), 'inplace LoRA requires caller no_grad')
        require(all(not module.training for module in self.model.modules())
                and all(not parameter.requires_grad for parameter in self.model.parameters()),
                'inplace LoRA requires all model modules eval and parameters frozen')
        layers = [(name, module) for name, module in self.model.named_modules() if isinstance(module, LoRALinear)]
        require(layers, 'no LoRALinear modules found')
        for name, module in layers:
            require('forward' not in module.__dict__, 'LoRA forward already overridden; overlapping contexts forbidden')
            self._states[name] = _layer_state(module)
        self._opened = True
        try:
            for name, module in layers:
                self._counts[name] = dict(calls=0, completed_calls=0, base_calls=0, lora_a_calls=0, lora_b_calls=0)

                def forward(layer, inputs, name=name):
                    return self._forward(name, layer, inputs)

                module.forward = MethodType(forward, module)
                self._installed.append(module)
        except BaseException:
            self.close()
            raise
        return self

    def _forward(self, name, module, inputs):
        require(self._opened and not self._closed, 'inplace LoRA context is closed')
        count = self._counts[name]
        count['calls'] += 1
        require(not torch.is_grad_enabled(), 'inplace LoRA requires caller no_grad')
        require(not self.model.training and _layer_state(module) == self._states[name],
                'inplace LoRA model/parameter/scaling/dtype/device binding changed')
        require(torch.is_tensor(inputs) and not inputs.requires_grad,
                'inplace LoRA requires a tensor input without active gradients')
        base_out = module.base(inputs)
        count['base_calls'] += 1
        lowrank = module.lora_a(module.dropout(inputs))
        count['lora_a_calls'] += 1
        lowrank = module.lora_b(lowrank)
        count['lora_b_calls'] += 1
        require(base_out.shape == lowrank.shape and base_out.dtype == lowrank.dtype
                and base_out.device == lowrank.device and not base_out.requires_grad,
                'inplace LoRA requires matching newly computed branch outputs')
        # Exact nn.Linear children without user hooks allocate a fresh output.
        # No operation below mutates inputs, weights or low-rank activations.
        result = base_out.add_(lowrank)
        count['completed_calls'] += 1
        return result

    def close(self):
        for module in self._installed:
            # Entry rejects pre-existing overrides, so deleting this instance
            # attribute restores the original LoRALinear.forward descriptor.
            delattr(module, 'forward')
        self._installed.clear()
        self._states.clear()
        self.model = None
        self._closed = True

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
