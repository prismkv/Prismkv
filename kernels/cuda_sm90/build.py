"""Build the standalone SM90 libraries used by the H200 runner."""
import argparse
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'build')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    common = [os.environ.get('NVCC', 'nvcc'), '-O3', '-std=c++17', '--fmad=false',
              '-gencode=arch=compute_90,code=sm_90', '-lineinfo', '-Xptxas=-v',
              '-Xcompiler=-fPIC', '-shared']
    native = HERE/'native_sources'
    libraries = {
        'prism.so': [native/name for name in ('pack.cu', 'route.cu', 'union.cu', 'av.cu',
                                               'moments.cu', 'hist.cu', 'gather.cu')],
        'hopper_top16_v3.so': [HERE/'hopper_top16.cu'],
    }
    for name, sources in libraries.items():
        binary = args.out/name
        subprocess.run([*common, *map(str, sources), '-o', str(binary)], check=True)
        if b'.nv_fatbin' not in binary.read_bytes():
            raise RuntimeError(f'{name} does not contain a CUDA fatbinary')


if __name__ == '__main__':
    main()
