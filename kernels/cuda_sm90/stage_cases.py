"""Select two frozen 27B LongBench arrival cohorts from the archived input bank."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random

MANIFEST_SHA = 'be231a0a1c3d6ebb1132c75d954f3df0fa5ead3f1ad2fe9f515c5d365360c894'
LONG_SHA = 'bbab50f4560674bddd7e01870aed0a31b3e440f0b024155832aef1a74b204e8d'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def crop(ids, history):
    if not 256 <= history <= len(ids):
        raise ValueError('source is too short or requested length is below the retained suffix')
    return ids[:history - 256] + ids[-256:]


def select(manifest, data):
    raw = manifest.read_bytes()
    if digest(raw) != MANIFEST_SHA:
        raise ValueError('archived LongBench manifest does not match')
    rows = [row for shard in json.loads(raw)['shards'] for row in shard]
    cohorts = {}
    for band in (8, 32):
        history = band * 1024
        candidates = [row for row in rows if row['H'] >= history]
        for repeat in (0, 1):
            rng = random.Random(20261109 + repeat)
            result = []
            for row in rng.sample(candidates, 48):
                contents = (data / f"{row['name']}.json").read_bytes()
                if digest(contents) != row['row_sha']:
                    raise ValueError(f"archived input does not match: {row['name']}")
                ids = crop(json.loads(contents)['input_ids'], rng.randrange(3 * history // 4, history + 1))
                result.append(dict(source=row['name'], source_sha=row['row_sha'], ids=ids,
                                   cap=rng.randint(16, 64)))
            cohorts[f'{band}:{repeat}'] = result
    return dict(manifest_sha=MANIFEST_SHA, cohorts=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = select(args.manifest, args.data)
    contents = gzip.compress(json.dumps(result, separators=(',', ':')).encode(), mtime=0)
    if args.out is None:
        import sys
        sys.stdout.buffer.write(contents)
    else:
        args.out.write_bytes(contents)


if __name__ == '__main__':
    main()
