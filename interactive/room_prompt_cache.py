"""Exact CPU merge of already-encoded text; no encoder/model/GPU is invoked."""
from pathlib import Path


LIMIT = 64_000_000
INTERIOR_RIGHT = ('First person view from the current position inside the dim storage room. '
    'Slowly turn the camera a little to the RIGHT to inspect the metal shelves and archive boxes, '
    'then hold the view. Keep the same rigid shelves, box arrangement, concrete floor, warm '
    'overhead light and doorway edges. No change of room, no zoom, no people, no new door and no opening animation.')


def protocol(*,old_cache,old_sha256,existing_cache,existing_sha256,output_root):
    import hashlib
    for path,digest in ((old_cache,old_sha256),(existing_cache,existing_sha256)):
        source=Path(path)
        if (not isinstance(digest,str) or len(digest)!=64 or source.stat().st_size>LIMIT
                or hashlib.sha256(source.read_bytes()).hexdigest()!=digest):
            raise ValueError('Actual operator-selected prompt-cache identity differs')
    return dict(id='room-prompt-cache-v1', mode='CPU_exact_existing_tensor_merge',
        old_cache=str(old_cache), old_sha256=old_sha256, old_actual_bytes=Path(old_cache).stat().st_size,
        existing_cache=str(existing_cache), existing_sha256=existing_sha256, added_key='interior_right',
        added_text=INTERIOR_RIGHT, output_root=str(output_root), output_limit_bytes=LIMIT,
        cpu_seconds_limit=120, no_cuda=True, no_encoding=True, no_download=True,
        original_tensors_and_negative_preserved=True,
        visual_success_verified=False, training_qualified=False)


def merge_exact(base, fresh):
    import torch
    if ('interior_right' in base['prompts'] or 'interior_right' in base['embeddings']
            or fresh['prompts'].get('interior_right') != INTERIOR_RIGHT):
        raise ValueError('Exact new key/text is required without overriding an old key')
    tensor = fresh['embeddings']['interior_right']
    if (tuple(tensor.shape) != (512, 4096) or tensor.dtype != torch.bfloat16
            or tensor.device.type != 'cpu' or not bool(torch.isfinite(tensor).all())):
        raise ValueError('Existing interior_right embedding contract changed')
    if not torch.equal(base['embeddings']['negative'], fresh['embeddings']['negative']):
        raise ValueError('Existing negative tensor differs from the original')
    merged = dict(base)
    merged['prompts'] = {**base['prompts'], 'interior_right': fresh['prompts']['interior_right']}
    merged['embeddings'] = {**base['embeddings'], 'interior_right': tensor}
    return merged
