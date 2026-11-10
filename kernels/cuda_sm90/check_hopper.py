"""Check optimized Top16, compaction and AV on synthetic BF16 tensors."""
import argparse
from pathlib import Path

import torch

from native import DecodeOps, Index, Kernels


HERE = Path(__file__).resolve().parent


@torch.inference_mode()
def check(library, route_library):
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError('the check requires a CUDA SM90 GPU')
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    slots, history, tail = 2, 257, 5
    capacity = history + tail
    kernels = Kernels(library)
    legacy = DecodeOps(kernels, slots, history, capacity)
    optimized = DecodeOps(kernels, slots, history, capacity)
    optimized.enable_hopper_route(route_library)
    optimized.enable_hopper_av(pv_mode=1)
    index = Index(slots, history)
    keys = torch.randn(slots, capacity, 4, 256, dtype=torch.bfloat16, device='cuda')
    values = torch.randn_like(keys)
    query = torch.randn(slots, 1, 24, 256, dtype=keys.dtype, device='cuda')
    histories = torch.tensor([history, history-3], dtype=torch.int32, device='cuda')
    for slot, length in enumerate(histories.tolist()):
        entries = []
        for group in range(4):
            count = (17, 9, 1, 0)[group]
            sizes = torch.zeros(3000, device='cuda', dtype=torch.long)
            sizes[:count] = 4
            ids = torch.tensor([t for f in range(count) for t in (0, f+1, length//2, length-1)],
                               device='cuda', dtype=torch.long)
            offsets = torch.nn.functional.pad(sizes.cumsum(0), (1, 0))
            moment = torch.zeros(3000, 256, device='cuda')
            entries.append((sizes, ids, offsets, moment, moment))
        index.fill(slot, length, entries, kernels)

    positions = torch.arange(history, device='cuda')[None, :]
    for case in ('tied', 'distinct'):
        if case == 'distinct':
            index.matrix.normal_(std=.01)
        for ops in (legacy, optimized):
            ops.route(query, index)
        for name in ('features', 'selected', 'raw', 'counts'):
            torch.testing.assert_close(getattr(optimized, name), getattr(legacy, name), rtol=0, atol=0)
        mask = positions < legacy.counts[:, None]
        torch.testing.assert_close(optimized.ids[mask], legacy.ids[mask], rtol=0, atol=0)
        lengths = histories + tail
        reference = legacy.attend(query, keys, values, histories, lengths).float()
        for splits in (32, 64, 128):
            result = optimized.attend(query, keys, values, histories, lengths, splits=splits).float()
            relative = ((result-reference).norm()/reference.norm()).item()
            if relative >= .005:
                raise RuntimeError(f'{case}, splits={splits}, relative L2={relative:.6f}')
            print(f'{case}, splits={splits}, relative L2={relative:.6f}')
        lengths[-1] = 0
        inactive = optimized.attend(query, keys, values, histories, lengths)[-1]
        torch.testing.assert_close(inactive, torch.zeros_like(inactive), rtol=0, atol=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=HERE/'build/prism.so')
    parser.add_argument('--route-library', type=Path, default=HERE/'build/hopper_top16_v3.so')
    args = parser.parse_args()
    check(args.library, args.route_library)
