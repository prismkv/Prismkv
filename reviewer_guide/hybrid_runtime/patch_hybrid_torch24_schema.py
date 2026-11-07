"""Add the explicit grouped-MM schema required by Torch 2.4 to HF 5.8.0."""

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA = 'a320a4638fdac837672241be185dd18783cefab1bb47a3aeeaf00f244652261a'
ORIGINAL = b'    torch.library.custom_op("transformers::grouped_mm_fallback", _grouped_mm_fallback, mutates_args=())'
PATCHED = (b'    torch.library.custom_op(\n'
           b'        "transformers::grouped_mm_fallback",\n'
           b'        _grouped_mm_fallback,\n'
           b'        mutates_args=(),\n'
           b'        schema="(Tensor input, Tensor weight, Tensor offs) -> Tensor",\n'
           b'    )')


def patch(package):
    source = package / 'transformers/integrations/moe.py'
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA or data.count(ORIGINAL) != 1:
        raise ValueError('The target is not the exact unmodified Transformers 5.8.0 source.')
    patched = data.replace(ORIGINAL, PATCHED, 1)
    compile(patched, str(source), 'exec')
    temporary = source.with_name(source.name + '.prismkv.tmp')
    with temporary.open('xb') as stream:
        stream.write(patched)
    temporary.replace(source)
    return hashlib.sha256(patched).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('isolated_target', type=Path, help='Explicit private pip --target directory')
    args = parser.parse_args()
    print(patch(args.isolated_target))
