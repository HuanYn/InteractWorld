"""Instance-scoped, token-exact low-memory construction of the native TF mask.

This changes construction only: all blocks remain partial and retain the native
token predicate. No dense [packed_tokens, packed_tokens] temporary is allocated.
The original causal_model.py and non-TF/reference mask paths are untouched.
"""
from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask

VERSION = "completion-tf-mask-memory-v1"
BLOCK_SIZE = 128


def _geometry(num_frames, frame_seqlen, num_frame_per_block, independent_first_frame):
    for name, value in (("num_frames", num_frames), ("frame_seqlen", frame_seqlen),
                        ("num_frame_per_block", num_frame_per_block)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(independent_first_frame) is not bool:
        raise ValueError("independent_first_frame must be bool")
    ranges = []
    first = 1 if independent_first_frame else 0
    if first:
        ranges.append((0, frame_seqlen))
    for start in range(first, num_frames, num_frame_per_block):
        ranges.append((start * frame_seqlen,
                       min(start + num_frame_per_block, num_frames) * frame_seqlen))
    clean_end = num_frames * frame_seqlen
    padded = ((2 * clean_end + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    return ranges, clean_end, padded


def build_components(device, num_frames, frame_seqlen, num_frame_per_block=1,
                     independent_first_frame=False):
    """Return block visibility plus the unchanged token predicate and geometry.

    Visibility is a superset only at 128-token granularity. In particular, a
    block straddling clean/noisy or frame boundaries must still run mask_mod.
    CPU construction prevents GPU token-square temporaries and scalar syncs.
    """
    ranges, clean_end, padded = _geometry(
        num_frames, frame_seqlen, num_frame_per_block, independent_first_frame)
    context_ends = torch.zeros(padded, dtype=torch.long)
    noise_context_starts = torch.zeros(padded, dtype=torch.long)
    noise_context_ends = torch.zeros(padded, dtype=torch.long)
    noise_noise_starts = torch.zeros(padded, dtype=torch.long)
    noise_noise_ends = torch.zeros(padded, dtype=torch.long)
    for start, end in ranges:
        context_ends[start:end] = min(end, clean_end)
        a, b = clean_end + start, clean_end + end
        noise_noise_starts[a:b] = a
        noise_noise_ends[a:b] = b
        noise_context_ends[a:b] = start
    context_ends, noise_context_starts, noise_context_ends, noise_noise_starts, noise_noise_ends = (
        x.to(device) for x in (context_ends, noise_context_starts, noise_context_ends,
                              noise_noise_starts, noise_noise_ends))

    def attention_mask(b, h, q_idx, kv_idx):
        clean_mask = (q_idx < clean_end) & (kv_idx < context_ends[q_idx])
        c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        c2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
        noise_mask = (q_idx >= clean_end) & (c1 | c2)
        return (q_idx == kv_idx) | clean_mask | noise_mask

    nblocks = padded // BLOCK_SIZE
    visibility = torch.eye(nblocks, dtype=torch.bool)

    def mark(row, start, end):
        if start < end:
            visibility[row, start // BLOCK_SIZE:(end + BLOCK_SIZE - 1) // BLOCK_SIZE] = True

    for row in range(nblocks):
        qstart, qend = row * BLOCK_SIZE, (row + 1) * BLOCK_SIZE
        for start, end in ranges:
            if qstart < end and qend > start:
                mark(row, 0, end)
            if qstart < clean_end + end and qend > clean_end + start:
                mark(row, 0, start)
                mark(row, clean_end + start, clean_end + end)
    return visibility, attention_mask, {
        "version": VERSION, "clean_tokens": clean_end,
        "packed_tokens": 2 * clean_end, "padded_tokens": padded,
        "block_rows": nblocks, "block_size": BLOCK_SIZE,
        "visibility_bytes": visibility.numel() * visibility.element_size(),
        "token_predicate_retained": True, "all_blocks_partial": True,
    }


def prepare_mask(device, num_frames, frame_seqlen, num_frame_per_block=1,
                 independent_first_frame=False):
    visibility, predicate, info = build_components(
        device, num_frames, frame_seqlen, num_frame_per_block, independent_first_frame)
    dense_blocks = visibility[None, None].to(torch.int32)
    counts = dense_blocks.sum(-1).to(dtype=torch.int32, memory_format=torch.contiguous_format)
    indices = torch.argsort(dense_blocks, dim=-1, descending=True, stable=True).to(
        dtype=torch.int32, memory_format=torch.contiguous_format)
    return BlockMask.from_kv_blocks(
        counts.to(device), indices.to(device),
        full_kv_num_blocks=None, full_kv_indices=None,
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE), mask_mod=predicate,
        seq_lengths=(info["padded_tokens"], info["padded_tokens"]))


def install_scoped(model):
    """Patch only one constructed wrapper/model instance; return a small receipt.

    The private worker owns this model for its entire lifetime. No module/class
    globals, model parameters, optimizer state or RNG are changed.
    """
    target = getattr(model, "model", model)
    if not callable(getattr(target, "_prepare_teacher_forcing_mask", None)):
        raise TypeError("expected native causal model with teacher-forcing mask")
    existing = target.__dict__.get("_completion_tf_mask_memory_version")
    if existing is not None:
        if existing != VERSION or target.__dict__.get("_prepare_teacher_forcing_mask") is not prepare_mask:
            raise RuntimeError("conflicting instance TF mask override")
        return {"status": "installed", "version": VERSION, "already_installed": True}
    if "_prepare_teacher_forcing_mask" in target.__dict__:
        raise RuntimeError("refusing to overwrite an existing instance TF mask override")
    target._prepare_teacher_forcing_mask = prepare_mask
    target._completion_tf_mask_memory_version = VERSION
    target.block_mask = None
    target._block_mask_cache_key = None
    target._block_mask_cache = {}
    return {"status": "installed", "version": VERSION, "scope": "model_instance_only",
            "reference_mask_unchanged": True, "native_token_predicate_retained": True,
            "full_blocks": False, "dense_token_square_allocated": False}
