"""Read-only expanded60 references; old6+4 are aliases, not re-encoded tensors.

Only the new54 cache is understood here; the old pilot's strict6/4 reader is
delegated to unchanged. References never alter the exact-window sample/noise
schedule. Invalid slots remain in training with a zero tensor and false mask.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from training.demo.contracts import contained, require, sha256
from training.reference_dataset import (REFERENCE_SHAPE, ReferenceFeatureStore,
    reference_sample_id, tensor_sha256)

STAGE = 'teacher-expanded-reference-v1'
PAYLOAD_KIND = 'expanded_character_reference_vae_v1'
RECEIPT_SCHEMA = 'expanded_reference_index_receipt_v1'
PURPOSE_COUNTS = {'train': 60, 'qualification': 4}
VALID_COUNTS = {'train': 47, 'qualification': 3}
LEGACY_INDEX_SHA256 = {
    'train': '7545309953912797342b5ca0fa2466246d87c5c0e3daa5e6014b84defbcef1fd',
    'qualification': '939c9b8085b008646e84ec8865ff0fa9b049ac615e5832f9ce2b9a32022b9a4a',
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def check_index_binding(info):
    require(set(info) == {'path', 'sha256', 'receipt', 'receipt_sha256', 'manifest', 'manifest_sha256', 'windows'},
            'complete reference index binding required')
    require(str(Path(info['path']).resolve()) + '.receipt.json' == str(Path(info['receipt']).resolve()),
            'reference receipt path differs')
    for name, digest in (('path', 'sha256'), ('receipt', 'receipt_sha256'), ('manifest', 'manifest_sha256')):
        require(sha256(info[name]) == info[digest], 'reference index/receipt/manifest pin differs')


class ExpandedReferenceStore:
    """Purpose-separated60/4 index with immutable old10 lineage.

    Constructor and operator validation read small metadata only. ``load`` reads
    exactly the requested CPU reference payload, checking its hash and tensors.
    K=1; batched=True adds B=1. No reference is created for an unknown sample.
    """
    def __init__(self, binding, *, purpose):
        require(purpose in PURPOSE_COUNTS, 'explicit train or qualification purpose required')
        check_index_binding(binding)
        self.binding, self.purpose = copy.deepcopy(binding), purpose
        self.index_path = Path(binding['path']).resolve()
        require(self.index_path.name == f'{purpose}.references.jsonl', 'reference purpose filename differs')
        receipt = json.loads(Path(binding['receipt']).read_text(encoding='utf-8'))
        require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('stage') == STAGE
                and receipt.get('status') == 'completed' and receipt.get('purpose') == purpose
                and receipt.get('index') == str(self.index_path) and receipt.get('index_sha256') == binding['sha256']
                and receipt.get('manifest') == binding['manifest']
                and receipt.get('manifest_sha256') == binding['manifest_sha256'], 'expanded reference receipt differs')
        manifest = contained(binding['manifest'], self.index_path.parent.parent / 'manifests')
        require(manifest.name == f'{purpose}.reference.jsonl'
                and receipt.get('reference_shape') == list(REFERENCE_SHAPE)
                and receipt.get('observed_vae_output_shape') == [1, 1, 48, 16, 16]
                and receipt.get('source_kind') == 'original_raw_initial_frame'
                and receipt.get('future_reference_frames_used') is False
                and receipt.get('old_reference_reencodes') == 0, 'expanded reference geometry/source differs')
        self.lineage = receipt['lineage']
        for path, digest in self.lineage['files'].items():
            require(sha256(path) == digest, 'expanded reference source record changed')
        self.legacy = {}
        require(set(self.lineage['legacy_indexes']) == set(PURPOSE_COUNTS), 'both legacy purpose pins required')
        for role, info in self.lineage['legacy_indexes'].items():
            check_index_binding(info)
            require(info['sha256'] == LEGACY_INDEX_SHA256[role], 'original legacy reference index pin differs')
            self.legacy[role] = ReferenceFeatureStore(info['path'], purpose=role, expected_index_sha256=info['sha256'])
            require(self.legacy[role].binding['vae_sha256'] == self.lineage['vae_sha256'], 'old/new VAE identity differs')
        sources, rows = read_rows(manifest), read_rows(self.index_path)
        count = PURPOSE_COUNTS[purpose]
        require(len(sources) == len(rows) == binding['windows'] == receipt.get('windows') == count,
                'expanded purpose count differs')
        source_map = {r['sample_id']: r for r in sources}
        require(len(source_map) == count, 'duplicate expanded manifest sample')
        self.rows = {}
        for row in rows:
            sample = row['sample_id']
            require(sample not in self.rows and sample in source_map
                    and sample == reference_sample_id(row['episode_id'], row['source_start'])
                    and row['purpose'] == purpose and row['original_split'] == 'train'
                    and type(row['valid']) is bool
                    and all(row.get(k) == v for k, v in source_map[sample].items()),
                    'reference sample/manifest/purpose differs')
            if row['cache_source'] == 'legacy10':
                original = self.legacy[purpose].rows.get(sample)
                require(original is not None and all(row.get(k) == v for k, v in original.items()),
                        'legacy alias changed original reference row')
            else:
                require(row['cache_source'] == 'new54' and purpose == 'train'
                        and row['cache_origin'] == 'native97_pool256', 'new reference role/origin differs')
                path = contained(row['payload_path'], self.index_path.parent / 'tensors')
                require(path.name == sample + '.pt', 'expanded payload name differs')
            self.rows[sample] = row
        require(sum(r['valid'] for r in rows) == receipt.get('valid_count') == VALID_COUNTS[purpose]
                and sum(r['cache_source'] == 'new54' for r in rows) == (54 if purpose == 'train' else 0)
                and receipt.get('actual_new_vae_encodes') == (41 if purpose == 'train' else 0),
                'fixed47/13 train and3/1 qualification reference counts differ')
        require({r['sample_id'] for r in rows if r['cache_source'] == 'legacy10'} == set(self.legacy[purpose].rows),
                'expanded index must retain every original purpose reference')

    def load(self, sample_id, *, batched=False):
        require(type(batched) is bool and sample_id in self.rows, 'explicit batching and known purpose sample required')
        check_index_binding(self.binding)
        row = self.rows[sample_id]
        if row['cache_source'] == 'legacy10':
            return self.legacy[self.purpose].load(sample_id, batched=batched)
        path = contained(row['payload_path'], self.index_path.parent / 'tensors')
        require(sha256(path) == row['payload_sha256'], 'expanded reference payload hash differs')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        expected = dict(kind=PAYLOAD_KIND, stage=STAGE, sample_id=sample_id, purpose=self.purpose,
            episode_id=row['episode_id'], source_start=row['source_start'],
            annotation_sha256=self.lineage['annotation_sha256'], vae_sha256=self.lineage['vae_sha256'],
            raw_rgb_sha256=row['rgb_sha256'], crop_rgb_sha256=row['crop_rgb_sha256'], png_sha256=row['png_sha256'])
        require(all(payload.get(k) == v for k, v in expected.items()), 'expanded reference payload lineage differs')
        latent, mask = payload.get('ref_latents'), payload.get('ref_mask')
        require(torch.is_tensor(latent) and latent.device.type == 'cpu' and latent.dtype == torch.bfloat16
                and tuple(latent.shape) == REFERENCE_SHAPE and bool(torch.isfinite(latent).all()),
                'finite CPU BF16[1,48,1,16,16] reference required')
        require(torch.is_tensor(mask) and mask.device.type == 'cpu' and mask.dtype == torch.bool
                and tuple(mask.shape) == (1,) and mask.item() is row['valid'], 'reference validity mask differs')
        require(tensor_sha256(latent) == row['latent_sha256_float32']
                and (row['valid'] or torch.count_nonzero(latent).item() == 0), 'invalid slot or latent digest differs')
        return dict(ref_latents=latent.unsqueeze(0) if batched else latent,
                    ref_mask=mask.unsqueeze(0) if batched else mask)

    def for_window(self, episode_id, source_start, *, batched=False):
        return self.load(reference_sample_id(episode_id, source_start), batched=batched)


class ExpandedReferenceDataset:
    """Attach refs without changing the parent's exact window/flow/noise path.

    ``contract`` is the unmodified shared sampling contract, and
    ``reference_contract`` is separate arm/cache provenance. CPU test doubles
    can implement the same episodes/contract/sampling_spec/load interface.
    """
    def __init__(self, base_dataset, store, *, arm):
        require(arm in ('no_reference', 'reference') and store.purpose == 'train', 'fixed arm and train reference store required')
        self.base, self.store, self.arm = base_dataset, store, arm
        self.episodes, self.contract = base_dataset.episodes, base_dataset.contract
        expected = {reference_sample_id(r['episode_id'], r['source_start']): r for r in self.episodes}
        require(len(self.episodes) == len(expected) == 60 and set(expected) == set(store.rows),
                'references must cover all60 exact windows, including invalid slots')
        for sample, row in expected.items():
            ref = store.rows[sample]
            require(all(ref[k] == row[k] for k in ('episode_id', 'source_start', 'window_index', 'cache_origin', 'target')),
                    'reference/full exact-window identity differs')
        self.reference_contract = dict(stage=STAGE, arm=arm, binding=copy.deepcopy(store.binding),
            train_windows=60, valid_references=47, invalid_references=13, window_filtering=False)

    def __len__(self):
        return len(self.base)

    def sampling_spec(self, index):
        return self.base.sampling_spec(index)

    def __getitem__(self, index):
        value = self.base[index]
        require('ref_latents' not in value and 'ref_mask' not in value, 'base dataset unexpectedly supplies references')
        if self.arm == 'no_reference':
            return value
        spec = self.sampling_spec(index)
        return dict(value, **self.store.for_window(spec['episode_id'], spec['source_start']))
