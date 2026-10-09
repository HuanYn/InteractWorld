"""CPU crops of explicitly bound raw initial RGB frames; no I/O or detection.

The caller must decode the original video at ``source_start`` and attest to
that provenance. Tensor pixels alone cannot prove their frame of origin. JPEG
annotation previews, generated frames and existing latents are not inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch.nn import functional as F


SOURCE_KIND = 'original_raw_initial_frame'
OUTPUT_SIZE = 256
PADDING_RGB = (127, 127, 127)


class ReferenceCrop(NamedTuple):
    crop: torch.Tensor | None
    valid: bool


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def crop_reference(
    raw_rgb_chw: torch.Tensor,
    row: Mapping[str, Any],
    *,
    sample_id: str,
    source_start: int,
    source_kind: str,
) -> ReferenceCrop:
    """Return a uint8 RGB [3,256,256] crop, or ``(None, False)``.

    Input is one dense CPU uint8 RGB tensor in CHW order. The sidecar row and
    caller must agree on sample_id, source_start and original-raw-initial-frame
    provenance; sidecar width/height must match the supplied tensor exactly.
    bbox is [left,top,right,bottom], with exclusive right/bottom coordinates.

    Valid crops use bilinear resize (align_corners=False, antialias=True),
    preserving aspect ratio to the nearest pixel (positive half-up dimension
    rounding), then centered RGB127 padding. An odd padding pixel goes on the
    right/bottom. Output pixels are rounded to uint8. No source pixels outside
    bbox participate. Invalid annotations require no bbox and a rejection
    reason; they return no image, so a placeholder cannot become a valid slot.

    This only checks the declared provenance and annotation geometry. It does
    not detect people, establish annotation correctness, encode a VAE or build
    dataset/cache/mask tensors. The caller retains the sidecar provenance.
    """
    _require(torch.is_tensor(raw_rgb_chw), 'source must be a torch RGB CHW tensor')
    _require(raw_rgb_chw.device.type == 'cpu' and raw_rgb_chw.layout == torch.strided,
             'source must be a dense CPU RGB tensor')
    _require(raw_rgb_chw.dtype == torch.uint8 and raw_rgb_chw.ndim == 3
             and raw_rgb_chw.shape[0] == 3 and min(raw_rgb_chw.shape[1:]) > 0,
             'source must be uint8 RGB [3,H,W] with positive dimensions')
    _require(isinstance(row, Mapping), 'reference sidecar row must be a mapping')
    required = {'sample_id', 'source_start', 'source_kind', 'source_width',
                'source_height', 'bbox', 'valid', 'rejection_reason'}
    _require(required <= row.keys(), 'reference sidecar row is missing required fields')
    _require(isinstance(sample_id, str) and bool(sample_id.strip())
             and row['sample_id'] == sample_id, 'source sample_id does not match sidecar')
    _require(type(source_start) is int and source_start >= 0
             and type(row['source_start']) is int and row['source_start'] == source_start,
             'source_start must match the exact initial source frame')
    _require(source_kind == SOURCE_KIND and row['source_kind'] == SOURCE_KIND,
             'reference source must be original_raw_initial_frame, never preview/generated/latent')
    width, height = row['source_width'], row['source_height']
    _require(type(width) is int and type(height) is int and width > 0 and height > 0
             and tuple(raw_rgb_chw.shape) == (3, height, width),
             'source dimensions do not match sidecar width/height')
    _require(type(row['valid']) is bool, 'sidecar valid must be an explicit bool')

    bbox, reason = row['bbox'], row['rejection_reason']
    if not row['valid']:
        _require(bbox is None and isinstance(reason, str) and bool(reason.strip()),
                 'invalid reference requires bbox=None and a rejection_reason')
        return ReferenceCrop(None, False)

    _require(reason is None, 'valid reference cannot also have a rejection_reason')
    _require(isinstance(bbox, (list, tuple)) and len(bbox) == 4
             and all(type(value) is int for value in bbox),
             'bbox must contain four integer pixel coordinates')
    left, top, right, bottom = bbox
    _require(0 <= left < right <= width and 0 <= top < bottom <= height,
             'bbox must be nonempty and within source dimensions; no clipping or fallback')
    crop_width, crop_height = right - left, bottom - top
    longest = max(crop_width, crop_height)
    scaled_width = max(1, (crop_width * OUTPUT_SIZE + longest // 2) // longest)
    scaled_height = max(1, (crop_height * OUTPUT_SIZE + longest // 2) // longest)
    selected = raw_rgb_chw[:, top:bottom, left:right]
    resized = F.interpolate(selected.unsqueeze(0).float(), size=(scaled_height, scaled_width),
                            mode='bilinear', align_corners=False, antialias=True)[0]
    resized = resized.round().clamp(0, 255).to(torch.uint8)
    output = torch.full((3, OUTPUT_SIZE, OUTPUT_SIZE), PADDING_RGB[0], dtype=torch.uint8)
    x_offset = (OUTPUT_SIZE - scaled_width) // 2
    y_offset = (OUTPUT_SIZE - scaled_height) // 2
    output[:, y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized
    return ReferenceCrop(output, True)
