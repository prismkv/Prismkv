"""NVIDIA bindings for the paper's membership, routing and indexed AV kernels."""
import ctypes
from pathlib import Path

import torch
from torch.nn.functional import pad

HERE = Path(__file__).resolve().parent
FEATURES, WIDTH, KV_GROUPS, Q_PER_GROUP = 3000, 256, 4, 6


def cuda_empty(*shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device='cuda')


class Kernels:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(Path(path).resolve()))
        uint, integer = ctypes.c_uint64, ctypes.c_int
        for name, args in {
            'pack': 'PPi', 'build_bits': 'PPPii', 'moments': 'PPPPPPi',
            'fused_norm_top': 'PPPPPPi', 'launch_union_warp': 'PPPPPPPiii',
            'batch_append': 'PPPPPii', 'batch_av': 'PPPPPPPPPiiii',
            'gather_kv': 'PPPPPPiiii', 'hist': 'PPii', 'reduce_cut': 'PPPii',
            'count': 'PPiP', 'collect': 'PPPPiP',
        }.items():
            entry = getattr(self.lib, name)
            entry.argtypes = [uint if c == 'P' else integer for c in args] + [uint]
            entry.restype = integer

    def call(self, name, *args):
        stream = torch.cuda.current_stream().cuda_stream
        status = getattr(self.lib, name)(*[x.data_ptr() if torch.is_tensor(x) else x for x in args], stream)
        if status:
            raise RuntimeError(f'{name}: CUDA error {status}')

    def positive_top(self, activations, limit):
        flat = activations.view(-1)
        size = flat.numel()
        blocks = min(4096, (size + 8191) // 8192)
        histogram = cuda_empty(blocks, 2048, dtype=torch.int32)
        total = cuda_empty(2048, dtype=torch.int32)
        cut = cuda_empty(1, dtype=torch.int32)
        counts = cuda_empty((size + 2047) // 2048, dtype=torch.int32)
        self.call('hist', flat, histogram, size, blocks)
        self.call('reduce_cut', histogram, total, cut, blocks, limit)
        self.call('count', flat, counts, size, cut)
        prefix = counts.cumsum(0)
        candidates = int(prefix[-1])
        ids = cuda_empty(candidates, dtype=torch.long)
        if candidates <= 0:
            return ids
        values = cuda_empty(candidates)
        self.call('collect', flat, prefix, ids, values, size, cut)
        return ids[values.topk(limit, sorted=False).indices] if candidates > limit else ids


class IndexBuilder:
    def __init__(self, kernels, weights):
        self.kernels, self.weights = kernels, weights
        self.moment_scratch = cuda_empty(FEATURES, 8, 2 * WIDTH + 1)

    def group(self, keys, layer, group):
        weight, bias, scale = self.weights[layer, group]
        keys = keys.float().contiguous()
        standardized = (keys - keys.mean(0)) / scale
        history = len(keys)
        scores = cuda_empty(history, FEATURES)
        for offset in range(0, history, 4096):
            panel = scores[offset:offset + 4096]
            torch.addmm(bias, standardized[offset:offset + 4096], weight.T, out=panel)
            panel.relu_()
        activations = self.kernels.positive_top(scores, history * 8)
        del scores, standardized
        features, tokens = activations.remainder(FEATURES), activations.div(FEATURES, rounding_mode='floor')
        counts = torch.bincount(features, minlength=FEATURES)
        ids = tokens[torch.argsort(features, stable=True)]
        offsets = pad(counts.cumsum(0), (1, 0))
        mu = cuda_empty(FEATURES, WIDTH)
        var = torch.empty_like(mu)
        self.kernels.call('moments', keys, ids, offsets, self.moment_scratch, mu, var, 8)
        return counts, ids, offsets, mu, var


class Index:
    def __init__(self, slots, history):
        self.history = history
        self.words = (history + 31) // 32
        self.matrix = cuda_empty(slots, KV_GROUPS, 2 * WIDTH, FEATURES)
        self.counts = torch.zeros(slots, KV_GROUPS, FEATURES, dtype=torch.int32, device='cuda')
        self.logn = torch.zeros_like(self.counts, dtype=torch.float32)
        self.bits = torch.zeros(slots, KV_GROUPS, FEATURES, self.words, dtype=torch.int32, device='cuda')

    def set_slot(self, slot, keys, layer, builder):
        length = keys.shape[0]
        entries = [builder.group(keys[:, g], layer, g) for g in range(KV_GROUPS)]
        self.fill(slot, length, entries, builder.kernels)

    def fill(self, slot, length, entries, kernels):
        width = max(1, max(item[1].numel() for item in entries))
        ids = torch.stack([pad(item[1].int(), (0, width - item[1].numel())) for item in entries])
        offsets = torch.stack([item[2].int() for item in entries])
        for group, (count, _, _, mean, var) in enumerate(entries):
            self.counts[slot, group].copy_(count)
            self.logn[slot, group].copy_(count.float().clamp_min(1).log())
            self.matrix[slot, group, :WIDTH].copy_(mean.T / 16)
            self.matrix[slot, group, WIDTH:].copy_(var.T / 512)
        words = (length + 31) // 32
        target = self.bits[slot].zero_()
        if words == self.words:
            kernels.call('build_bits', ids, offsets, target, length, width)
        else:
            temporary = torch.zeros(KV_GROUPS, FEATURES, words, dtype=torch.int32, device='cuda')
            kernels.call('build_bits', ids, offsets, temporary, length, width)
            target[..., :words].copy_(temporary)

    def inventory(self):
        return {name: value.numel() * value.element_size() for name, value in vars(self).items()
                if torch.is_tensor(value)}


class DecodeOps:
    def __init__(self, kernels, slots, history, capacity, splits=None, union_mode=2):
        self.kernels, self.history, self.capacity = kernels, history, capacity
        self.splits = splits or (64 if slots <= 4 else 32)
        self.union_mode = union_mode
        groups = slots * KV_GROUPS
        self.packed = cuda_empty(groups, Q_PER_GROUP, 2 * WIDTH)
        self.scores = cuda_empty(groups, Q_PER_GROUP, FEATURES)
        self.features = cuda_empty(groups, 16, dtype=torch.long)
        self.selected = cuda_empty(groups, 16)
        self.raw = cuda_empty(groups, dtype=torch.int32)
        words = (history + 31) // 32
        self.mask = cuda_empty(groups, words, dtype=torch.int32)
        self.tiles = cuda_empty(groups, (words + 63) // 64, dtype=torch.int32)
        self.ids = cuda_empty(groups, history, dtype=torch.int32)
        self.counts = cuda_empty(groups, dtype=torch.int32)
        self.state = cuda_empty(groups * Q_PER_GROUP, 128, WIDTH + 2)
        self.output = cuda_empty(slots, 1, KV_GROUPS * Q_PER_GROUP, WIDTH, dtype=torch.bfloat16)
        self.hopper_av = None
        self.hopper_pv_mode = 1
        self.hopper_top16 = None

    def enable_hopper_route(self, library):
        from hopper_route import HopperTop16

        self.hopper_top16 = HopperTop16(self, library, mode=5)
        self.hopper_route_workspace = self.hopper_top16.normalized

    def enable_hopper_av(self, pv_mode):
        from hopper_av import HopperAV

        if torch.cuda.get_device_capability(self.output.device) != (9, 0):
            raise ValueError('the Hopper Attention backend requires an SM90 device')
        self.hopper_av = HopperAV(self)
        self.hopper_pv_mode = pv_mode

    def score(self, q, index):
        groups = q.shape[0] * KV_GROUPS
        self.kernels.call('pack', q, self.packed, groups)
        torch.bmm(self.packed[:groups], index.matrix.view(-1, 2 * WIDTH, FEATURES)[:groups],
                  out=self.scores[:groups])
        if self.hopper_top16 is None:
            self.kernels.call('fused_norm_top', self.scores, index.logn, index.counts,
                              self.features, self.selected, self.raw, groups)
        else:
            self.hopper_top16(self.scores, index, self, groups)

    def merge(self, slots, index, features=None, selected=None, mode=None):
        self.kernels.call('launch_union_warp', index.bits, self.features if features is None else features,
                          self.selected if selected is None else selected,
                          self.mask, self.tiles, self.ids, self.counts, slots * KV_GROUPS, self.history,
                          self.union_mode if mode is None else mode)

    def route(self, q, index):
        self.score(q, index)
        self.merge(q.shape[0], index)

    def attend(self, q, k, v, histories, lengths, ids=None, counts=None, splits=None):
        if self.hopper_av is not None:
            return self.hopper_av.attend(q, k, v, histories, lengths, ids=ids, counts=counts,
                                         splits=splits or self.splits, tile=64, warps=4,
                                         pv_mode=self.hopper_pv_mode)
        slots = q.shape[0]
        self.kernels.call('batch_av', q, k, v, self.ids if ids is None else ids,
                          self.counts if counts is None else counts,
                          self.state, self.output, histories, lengths, slots * KV_GROUPS,
                          self.history, k.shape[1], splits or self.splits)
        return self.output[:slots]
