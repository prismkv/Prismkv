"""Opt-in SM90 Top16 implementation with the original FP32 routing scores."""
import ctypes
from pathlib import Path

import torch


class HopperTop16:
    def __init__(self, ops, library, mode):
        if mode not in (1, 2, 3, 4, 5):
            raise ValueError('unsupported Top16 mode')
        self.mode = mode
        self.normalized = torch.empty_like(ops.scores) if mode in (2, 3, 5) else None
        self.shortlist = ops.features.new_empty((ops.features.shape[0], 8, 16)) if mode == 3 else None
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.launch = self.lib.hopper_top16
        self.launch.argtypes = [ctypes.c_uint64] * 8 + [ctypes.c_int] * 2 + [ctypes.c_uint64]
        self.launch.restype = ctypes.c_int

    def __call__(self, scores, index, ops, groups):
        status = self.launch(scores.data_ptr(), index.logn.data_ptr(), index.counts.data_ptr(),
                             ops.features.data_ptr(), ops.selected.data_ptr(), ops.raw.data_ptr(),
                             0 if self.normalized is None else self.normalized.data_ptr(),
                             0 if self.shortlist is None else self.shortlist.data_ptr(), groups, self.mode,
                             torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f'hopper_top16: CUDA error {status}')
