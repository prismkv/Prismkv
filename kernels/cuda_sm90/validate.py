"""Seeded CUDA witnesses. Run before allocating the full model."""
import torch

from measure import event_times, sync
from native import DecodeOps, Index


def tail_check(kernels, H, tail, slots):
    torch.manual_seed(20261109)
    capacity = H+tail
    keys = torch.randn(slots, capacity, 4, 256, device='cuda', dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    q = torch.randn(slots, 1, 24, 256, device='cuda', dtype=keys.dtype)
    ops = DecodeOps(kernels, slots, H, capacity)
    index = Index(slots, H)
    histories = torch.tensor([H-s*3 for s in range(slots)], device='cuda', dtype=torch.int32)
    lengths = histories + tail
    expected = []
    for slot in range(slots):
        entries = []
        for group in range(4):
            # Cover overlap, empty buckets/groups, the final old token, high
            # bitset words, and a partial last word.
            n = torch.zeros(3000, device='cuda', dtype=torch.long)
            count = (17, 9, 1, 0)[group]
            n[:count] = 4
            hist = H-slot*3
            ids = torch.tensor([t for f in range(count) for t in (0, f+1, hist//2, hist-1)],
                               device='cuda', dtype=torch.long)
            mu = torch.zeros(3000, 256, device='cuda')
            var = torch.zeros_like(mu)
            # All scores tie intentionally to exercise the route's feature-ID tie order.
            offsets = torch.nn.functional.pad(n.cumsum(0), (1, 0))
            entries.append((n, ids, offsets, mu, var))
            old = ids[:min(count, 16)*4].unique()
            idx = torch.cat((old, torch.arange(hist, hist+tail, device='cuda')))
            scores = (q[slot, 0, group*6:(group+1)*6].float() @ keys[slot, idx, group].float().T)/16
            av = scores.softmax(-1) @ values[slot, idx, group].float()
            expected.append((group, count, old, av))
        index.fill(slot, H-slot*3, entries, kernels)
    ops.route(q, index)
    for mode in (0, 1, 2, 3):
        ops.merge(slots, index, mode=mode)
        for i, (g, count, ids, _) in enumerate(expected):
            n = min(16, count)
            torch.testing.assert_close(ops.features[i, :n], torch.arange(n, device='cuda'))
            torch.testing.assert_close(ops.ids[i, :ops.counts[i]].long(), ids)
    reference = torch.stack([x[-1] for x in expected]).view(slots, 1, 24, 256)
    result = {}
    for splits in (32, 64, 128):
        out = ops.attend(q, keys, values, histories, lengths, splits=splits)
        relative_l2 = ((out.float()-reference).norm()/reference.norm()).item()
        if relative_l2 >= .01:
            raise RuntimeError(f'AV relative L2 {relative_l2} at H={H}, T={tail}, splits={splits}')
        result[splits] = relative_l2
    # Inactive slots must ignore their uninitialized routing outputs.
    if slots > 1:
        lengths[-1] = 0
        masked = ops.attend(q, keys, values, histories, lengths)
        if masked[-1].abs().max().item() != 0:
            raise RuntimeError('inactive slot has non-zero attention')
    return dict(history=H, tail=tail, slots=slots, relative_l2=result)


def top_check(kernels):
    random = torch.rand(93, 3000, device='cuda')
    sparse = torch.zeros_like(random)
    sparse.view(-1)[:11] = 1.
    for a in (random, (random*8).round(), sparse, torch.zeros_like(random)):
        ids = kernels.positive_top(a, 744)
        target = a.view(-1).topk(744).values
        torch.testing.assert_close(a.view(-1)[ids].sort(descending=True).values, target[target>0])
        if ids.unique().numel() != ids.numel():
            raise RuntimeError('duplicate membership coordinate')


def run(kernels, include_long=True):
    top_check(kernels)
    cases = [(257, 1, 2), (32768, 1024, 1)]
    if include_long:
        cases.append((258048, 4096, 1))
    result = [tail_check(kernels, *case) for case in cases]
    sync()
    return dict(status='complete', cases=result, global_topk='exact selected values, ties allowed')
