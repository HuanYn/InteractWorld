"""Read immutable, purpose-separated image-reference VAE caches by sample ID.

Only reference tensors are loaded. Existing video/action/text caches stay with
their original dataset. No reference is synthesized from a video latent or a
later frame; an invalid annotated image is an explicit zero/false slot.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch

from training.demo.contracts import contained, require, sha256

STAGE = 'teacher-reference-pilot-v1'
PAYLOAD_KIND = 'teacher_character_reference_vae_v1'
RECEIPT_SCHEMA = 'teacher_reference_index_receipt_v1'
REFERENCE_SHAPE = (1, 48, 1, 16, 16)  # K,C,T,H,W; DataLoader adds B.
PURPOSE_COUNTS = {'train': 6, 'qualification': 4}


def reference_sample_id(episode_id, source_start):
    require(isinstance(episode_id, str) and re.fullmatch(r'[A-Za-z0-9_-]+', episode_id)
            and type(source_start) is int and source_start >= 0, 'safe episode ID and nonnegative integer source_start required')
    return f'{episode_id}-raw-start-{source_start}'


def tensor_sha256(value):
    """Canonical float32 bytes; tensor shape/dtype are checked separately."""
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


class ReferenceFeatureStore:
    """One training OR qualification reference index, with no role fallback.

    ``load(sample_id)`` returns BF16 ``ref_latents[K,C,T,H,W]`` and bool
    ``ref_mask[K]``; ``batched=True`` adds a leading B=1 for inference. Each
    call reads a fresh small tensor so caller edits cannot poison later samples.
    Operator completion/settlement checks belong to validate_completed_cache.
    """
    def __init__(self, index_path, *, purpose, expected_index_sha256=None):
        require(purpose in PURPOSE_COUNTS, 'explicit train or qualification purpose required')
        self.index_path = Path(index_path).resolve()
        self.purpose = purpose
        require(self.index_path.name == f'{purpose}.references.jsonl', 'reference index purpose filename differs')
        receipt_path = Path(str(self.index_path) + '.receipt.json')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        digest = sha256(self.index_path)
        require(expected_index_sha256 is None or digest == expected_index_sha256, 'expected reference index hash differs')
        require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('stage') == STAGE
                and receipt.get('status') == 'completed' and receipt.get('purpose') == purpose
                and receipt.get('index') == str(self.index_path) and receipt.get('index_sha256') == digest,
                'reference index receipt binding differs')
        require(receipt.get('reference_shape') == list(REFERENCE_SHAPE)
                and receipt.get('observed_vae_output_shape') == [1, 1, 48, 16, 16]
                and receipt.get('source_kind') == 'original_raw_initial_frame'
                and receipt.get('original_split') == 'train', 'reference geometry/source/encoder contract differs')
        manifest = contained(receipt['manifest'], self.index_path.parent.parent / 'manifests')
        require(manifest.name == f'{purpose}.reference.jsonl' and sha256(manifest) == receipt['manifest_sha256'],
                'reference manifest hash/purpose differs')
        rows, sources = _read_rows(self.index_path), _read_rows(manifest)
        count = PURPOSE_COUNTS[purpose]
        require(len(rows) == len(sources) == receipt.get('windows') == count, 'fixed reference purpose count differs')
        self.rows = {}
        source_map = {row['sample_id']: row for row in sources}
        require(len(source_map) == count, 'duplicate reference manifest sample')
        for row in rows:
            identity = row['sample_id']
            require(identity not in self.rows and identity in source_map
                    and identity == reference_sample_id(row['episode_id'], row['source_start']),
                    'duplicate, absent or mismatched reference sample ID')
            require(all(row.get(k) == v for k, v in source_map[identity].items())
                    and row['purpose'] == purpose and row['original_split'] == 'train'
                    and type(row['valid']) is bool, 'reference row changed its manifest identity/purpose')
            payload_path = contained(row['payload_path'], self.index_path.parent / 'tensors')
            require(payload_path.name == identity + '.pt', 'reference payload name differs from sample ID')
            self.rows[identity] = row
        require(sum(row['valid'] for row in rows) == receipt.get('valid_count') == receipt.get('actual_vae_encodes')
                == (6 if purpose == 'train' else 3), 'reference valid/actual encode count differs')
        self.binding = dict(stage=STAGE, purpose=purpose, index=str(self.index_path), index_sha256=digest,
            receipt=str(receipt_path), receipt_sha256=sha256(receipt_path), manifest=str(manifest),
            manifest_sha256=receipt['manifest_sha256'], annotation_sha256=receipt['annotation_sha256'],
            source_cache_result_sha256=receipt['source_cache_result_sha256'], vae_sha256=receipt['vae_sha256'])

    def load(self, sample_id, *, batched=False):
        require(type(batched) is bool, 'batched must be explicit bool')
        require(sample_id in self.rows, 'sample is absent from this reference purpose; no fallback')
        for key, digest_key in (('index', 'index_sha256'), ('receipt', 'receipt_sha256'), ('manifest', 'manifest_sha256')):
            require(sha256(self.binding[key]) == self.binding[digest_key], 'reference binding changed after store creation')
        row = self.rows[sample_id]
        path = contained(row['payload_path'], self.index_path.parent / 'tensors')
        require(sha256(path) == row['payload_sha256'], 'reference payload hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(payload.get('kind') == PAYLOAD_KIND and payload.get('stage') == STAGE
                and payload.get('sample_id') == sample_id and payload.get('purpose') == self.purpose
                and payload.get('episode_id') == row['episode_id'] and payload.get('source_start') == row['source_start']
                and payload.get('annotation_sha256') == self.binding['annotation_sha256']
                and payload.get('vae_sha256') == self.binding['vae_sha256'], 'reference payload lineage differs')
        latent, mask = payload.get('ref_latents'), payload.get('ref_mask')
        require(torch.is_tensor(latent) and tuple(latent.shape) == REFERENCE_SHAPE
                and latent.dtype == torch.bfloat16 and latent.device.type == 'cpu'
                and bool(torch.isfinite(latent).all()), 'finite BF16 reference [1,48,1,16,16] required')
        require(torch.is_tensor(mask) and tuple(mask.shape) == (1,) and mask.dtype == torch.bool
                and mask.device.type == 'cpu' and mask.item() is row['valid'], 'reference bool mask differs from annotation')
        require(tensor_sha256(latent) == row['latent_sha256_float32']
                and (row['valid'] or torch.count_nonzero(latent).item() == 0), 'invalid slot or reference values differ')
        require(payload.get('raw_rgb_sha256') == row['raw_rgb_sha256']
                and payload.get('crop_rgb_sha256') == row['crop_rgb_sha256'], 'reference pixel provenance differs')
        return dict(ref_latents=latent.unsqueeze(0) if batched else latent,
                    ref_mask=mask.unsqueeze(0) if batched else mask)

    def for_window(self, episode_id, source_start, *, batched=False):
        return self.load(reference_sample_id(episode_id, source_start), batched=batched)
