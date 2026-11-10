"""Validate the historical H200 workload and model fingerprints."""
import gzip
import hashlib
import json
from pathlib import Path

from stage_cases import LONG_SHA, MANIFEST_SHA, crop

PINNED = {
    'config.json': '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab',
    'model.safetensors.index.json': '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df',
    'weights.pt': '8420f99119b8166f8917dc060503280b94f0faca07dc82dbf591e27bc356d8a8',
    'long_source.json': LONG_SHA,
    'serving.json.gz': '06a5257406f887a39d03f3f98a75358fa4da5ea43007c744c7d70bdf517bb83a',
    'model_files.json': '050a9c869a59a3df05158f9bc284572b47dbaa4020f07d44a55ee97697aa7cb1',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_assets(assets, model=None):
    for name, expected in PINNED.items():
        if sha(assets/name) != expected:
            raise ValueError(f'archive identity mismatch: {name}')
    ids = json.loads((assets/'long_source.json').read_bytes())['ids']
    for band in (32, 128, 252):
        if len(crop(ids, band*1024)) != band*1024:
            raise ValueError('long prompt staging is incomplete')
    with gzip.open(assets/'serving.json.gz', 'rt', encoding='utf-8') as file:
        serving = json.load(file)
    if serving['manifest_sha'] != MANIFEST_SHA:
        raise ValueError('serving manifest hash differs from frozen original')
    for band in (8, 32):
        for repeat in (0, 1):
            rows = serving['cohorts'][f'{band}:{repeat}']
            if (len(rows) != 48 or len({r['source'] for r in rows}) != 48 or
                    any(not band*768 <= len(r['ids']) <= band*1024 or not 16 <= r['cap'] <= 64 for r in rows)):
                raise ValueError('invalid frozen serving cohort')
    files = json.loads((assets/'model_files.json').read_bytes())
    weights = json.loads((assets/'model.safetensors.index.json').read_bytes())['weight_map']
    required = {'config.json', 'generation_config.json', 'model.safetensors.index.json', *weights.values()}
    if required - files.keys():
        raise ValueError('archived model file list is incomplete')
    if model is not None:
        for name in sorted(required):
            path = model / name
            if not path.is_file() or path.stat().st_size != files[name]:
                raise ValueError(f'missing or incomplete model file: {path}')
        for name in ('config.json', 'model.safetensors.index.json'):
            if sha(model/name) != PINNED[name]:
                raise ValueError(f'wrong model identity: {name}')
    return sum(files.values())
