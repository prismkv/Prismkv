"""Inference-only encoder bundles with explicit shapes/scales and file hashes."""
from pathlib import Path
import hashlib
import json
import torch
from .core import SharedSAE
from .hf_reference import Encoder


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_encoders(encoders, directory, *, metadata=None):
    """Write one small tensor-only file per layer/KV group; never overwrite."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for (layer, group), bundle in sorted(encoders.items()):
        if layer < 0 or group < 0 or bundle.key_scale <= 0 or bundle.query_scale <= 0:
            raise ValueError("invalid group or saved scale")
        sae = bundle.sae
        name = f'l{layer:02d}_g{group:02d}.pt'
        path = root/name
        torch.save(dict(W_enc=sae.W_enc.detach().float().cpu(),
                        b_enc=sae.b_enc.detach().float().cpu()), path)
        records.append(dict(layer=layer,group=group,file=name,sha256=_sha(path),
            head_dim=sae.head_dim,features=sae.features,key_mean=sae.key_mean,
            query_topk=sae.query_topk,key_scale=float(bundle.key_scale),
            query_scale=float(bundle.query_scale)))
    (root/'manifest.json').write_text(json.dumps(dict(schema=1,
        format='PrismKV inference encoder bundle; decoder/optimizer omitted',
        metadata=metadata or {},groups=records), indent=2), encoding='utf8')


def load_encoders(directory, *, device='cpu'):
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema') != 1:
        raise ValueError("unsupported encoder-bundle schema")
    result = {}
    for row in manifest['groups']:
        path = (root/row['file']).resolve()
        if path.parent != root or _sha(path) != row['sha256']:
            raise ValueError("encoder path/hash mismatch")
        unit = (row['layer'],row['group'])
        if unit in result:
            raise ValueError("duplicate layer/KV group")
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        if set(tensors) != {'W_enc','b_enc'}:
            raise ValueError("unexpected checkpoint contents")
        # Loading an encoder should not advance the caller's random-number stream.
        with torch.random.fork_rng(devices=[]):
            sae = SharedSAE(row['head_dim'],row['features'],row['key_mean'],row['query_topk'])
        if tensors['W_enc'].shape != sae.W_enc.shape or tensors['b_enc'].shape != sae.b_enc.shape:
            raise ValueError("encoder shape mismatch")
        if not all(torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("nonfinite encoder")
        with torch.no_grad():
            sae.W_enc.copy_(tensors['W_enc'])
            sae.b_enc.copy_(tensors['b_enc'])
        sae.to(device).eval().requires_grad_(False)
        if row['key_scale'] <= 0 or row['query_scale'] <= 0:
            raise ValueError("positive Q/K scales required")
        result[unit] = Encoder(sae,row['key_scale'],row['query_scale'])
    return result, manifest
