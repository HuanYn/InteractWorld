"""Opt-in CPU assembly of step2 LoRA with its verified R005 parent adapter.

This is a runtime component diagnostic, not a new trained checkpoint. The
existing source loader still validates both immutable checkpoint lineages.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

from training.demo.contracts import ACTION_STAGE, require


ADAPTER_PREFIX = 'model.act_control_adapter.'


def _adapter_difference(state, donor, names):
    import torch
    result = dict(adapter_elements=0, fp32_different_elements=0, bf16_different_elements=0,
                  fp32_max_abs_delta=0., bf16_max_abs_delta=0.)
    # Keep temporary FP32 copies bounded even for the large spatial convolutions.
    for name in names:
        current, original = state[name], donor[name]
        require(current.device.type == original.device.type == 'cpu'
                and current.is_floating_point() and original.is_floating_point()
                and current.dtype == original.dtype and current.shape == original.shape,
                f'adapter shape/dtype/device mismatch: {name}')
        current, original = current.detach().reshape(-1), original.detach().reshape(-1)
        result['adapter_elements'] += current.numel()
        for start in range(0, current.numel(), 1_048_576):
            a, b = current[start:start + 1_048_576].float(), original[start:start + 1_048_576].float()
            require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                    f'nonfinite adapter: {name}')
            result['fp32_different_elements'] += int(torch.count_nonzero(a != b))
            result['fp32_max_abs_delta'] = max(result['fp32_max_abs_delta'], float((a - b).abs().max()))
            a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
            require(bool(torch.isfinite(a16).all()) and bool(torch.isfinite(b16).all()),
                    f'nonfinite BF16 adapter: {name}')
            result['bf16_different_elements'] += int(torch.count_nonzero(a16 != b16))
            result['bf16_max_abs_delta'] = max(result['bf16_max_abs_delta'], float((a16.float() - b16.float()).abs().max()))
    result.update(fp32_identical=result['fp32_different_elements'] == 0,
                  bf16_identical=result['bf16_different_elements'] == 0)
    require(result['adapter_elements'] > 0 and not result['bf16_identical'],
            'parent adapter is identical after BF16 conversion; no informative generation')
    return result


def apply_action_adapter_donor(request, state, config, lineage):
    """Return a new state mapping and explicit two-source provenance, or no-op.

    Only the exact recorded parent of native97 step2 is supported. Tensor values
    in the existing state and donor are never changed; no checkpoint is written.
    This function can run in CPU preparation before any GPU reservation.
    """
    if 'action_adapter_donor_source' not in request:
        return state, None
    import torch
    from training.studio.worker import read_source

    donor_source = request['action_adapter_donor_source']
    require(request.get('mode') == 'quality_probe' and isinstance(donor_source, str)
            and Path(donor_source).is_absolute() and '..' not in Path(donor_source).parts,
            'adapter donor requires an explicit absolute quality-probe source')
    initialization = lineage.get('initialization') or {}
    require(lineage.get('stage') == ACTION_STAGE and lineage.get('step') == 2
            and lineage.get('config', {}).get('data', {}).get('num_frames') == 97
            and initialization.get('mode') == 'length_resampled_transition_weights_only'
            and initialization.get('source_stage') == ACTION_STAGE
            and initialization.get('source_step') == 1040,
            'adapter hybrid requires the recorded native97 step2 transfer from R005 step1040')
    donor_state, donor_config, _, donor_lineage, _, _, _ = read_source({'source_job': donor_source})
    require(donor_lineage.get('stage') == ACTION_STAGE and donor_lineage.get('step') == 1040
            and donor_lineage.get('sha256') == initialization.get('sha256')
            and Path(donor_lineage['path']).resolve() == Path(initialization['path']).resolve(),
            'adapter donor is not the exact recorded R005 parent')
    require(is_dataclass(config) and is_dataclass(donor_config)
            and asdict(config) == asdict(donor_config), 'adapter donor model contract differs')
    require(isinstance(state, dict) and isinstance(donor_state, dict) and set(state) == set(donor_state),
            'adapter donor trainable tensor keys differ')
    names = sorted(name for name in state if name.startswith(ADAPTER_PREFIX))
    other = set(state).difference(names)
    require(names and other and all('.lora_a.' in name or '.lora_b.' in name for name in other),
            'hybrid supports only the complete action adapter and LoRA state')
    require(all(torch.is_tensor(value) and torch.is_tensor(donor_state[name])
                and value.shape == donor_state[name].shape for name, value in state.items()),
            'adapter donor trainable tensor shape differs')
    difference = _adapter_difference(state, donor_state, names)
    assembled = dict(state)
    assembled.update({name: donor_state[name] for name in names})
    require(all(assembled[name] is state[name] for name in other), 'non-adapter state changed')

    def identity(source_job, source):
        return dict(source_job=source_job, checkpoint_path=source['path'],
                    checkpoint_sha256=source['sha256'], step=source['step'])

    metadata = dict(schema_version=1, kind='runtime_parent_action_adapter_swap_v1',
        assembly='in_memory_only_no_new_checkpoint',
        lora_source=identity(request['source_job'], lineage),
        action_adapter_source=identity(donor_source, donor_lineage),
        replaced_prefix=ADAPTER_PREFIX, replaced_tensor_names=names, difference=difference,
        other_trainable_tensors_preserved=True, model_contract_matched=True,
        quality_improvement_verified=False,
        limitation='Adapter from49-frame R005; LoRA from native97 step2. This hybrid was not jointly trained; '
                   'one generated comparison cannot isolate a unique causal mechanism.')
    return assembled, metadata


def composition_fields(metadata):
    """No extra fields for legacy runs; never claim a hash for assembled weights."""
    if metadata is None:
        return {}
    return dict(runtime_model_kind='component_hybrid',
                checkpoint_role='lora_source_only_not_full_runtime_model',
                model_composition=metadata)
