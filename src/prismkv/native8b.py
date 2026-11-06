"""The measured PPU-specialized 8B route/union/indexed-attention kernel path.

Shapes are intentionally explicit: 8 KV groups, 4 Q heads/group, d=128,
3000 buckets, 16 selected buckets. The portable reference supports other shapes.
This backend requires the PPU SDK and is not advertised as a NVIDIA backend.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import time
import torch


def compile_ppu(source_root, output_dir, sdk_root="/usr/local/PPU_SDK"):
    source_root, output_dir, sdk_root = map(Path, (source_root, output_dir, sdk_root))
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = [source_root / f for f in ["pack.cu", "union.cu", "route_parallel.cu", "av_tuned.cu"]]
    command = [str(sdk_root / "CUDA_SDK/bin/nvcc"), "-O3", "--fmad=false",
               "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
               "-I" + str(sdk_root / "include"), "-L" + str(sdk_root / "lib"),
               "-lhggcrt1", *map(str, sources), "-o", str(output_dir / "libprismkv.so")]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    (output_dir / "build.log").write_text(result.stdout + result.stderr, encoding="utf8")
    (output_dir / "build.json").write_text(json.dumps(dict(command=command,
        seconds=time.perf_counter() - start, returncode=result.returncode,
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2), encoding="utf8")
    result.check_returncode()
    return output_dir / "libprismkv.so"


class Native8B:
    def __init__(self, indices, library):
        if len(indices) != 8:
            raise ValueError("native specialization requires 8 KV groups")
        h = indices[0].history
        device = indices[0].mean.device
        if device.type != "cuda":
            raise ValueError("native backend requires a device tensor")
        if any(x.history != h or x.mean.shape != (3000, 128) for x in indices):
            raise ValueError("same history, F=3000, d=128 required")
        self.history = h
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        for name, pointers, integers in [("pack", 2, 0), ("build_bits", 3, 2),
                ("fused_norm_top", 6, 0), ("launch_union_warp", 7, 3),
                ("attention_tuned", 7, 6)]:
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_uint64] * pointers + [ctypes.c_int] * integers + [ctypes.c_uint64]
            fn.restype = ctypes.c_int
        self.n = torch.stack([x.counts for x in indices]).int()
        self.ln = self.n.float().clamp_min(1).log()
        self.matrix = torch.stack([torch.cat((x.mean.T / 128 ** .5, x.variance.T / 256)) for x in indices]).contiguous()
        self.a = torch.empty(8, 4, 256, device=device)
        self.prod = torch.empty(8, 4, 3000, device=device)
        self.features = torch.empty(8, 16, dtype=torch.int64, device=device)
        self.valid = torch.empty(8, 16, device=device)
        self.raw = torch.empty(8, dtype=torch.int32, device=device)
        self.ids = torch.empty(8, h, dtype=torch.int32, device=device)
        self.count = torch.empty(8, dtype=torch.int32, device=device)
        nw = (h + 31) // 32
        self.bits = torch.zeros(8, 3000, nw, dtype=torch.int32, device=device)
        self.mask = torch.empty(8, nw, dtype=torch.int32, device=device)
        self.tiles = torch.empty(8, (nw + 127) // 128, dtype=torch.int32, device=device)
        capacity = max(x.memberships for x in indices)
        lists = [torch.cat(x.lists).int() for x in indices]
        src = torch.stack([torch.nn.functional.pad(x, (0, capacity - len(x))) for x in lists])
        offsets = torch.stack([torch.cat((x.counts.new_zeros(1), x.counts.cumsum(0))).int() for x in indices])
        self.call("build_bits", [src, offsets, self.bits], [h, capacity])
        # Temporary CSR build storage can be freed after this build barrier.
        torch.cuda.synchronize(device)
        self.state = torch.empty(32, 128, 130, device=device)
        self.out = torch.empty(8, 4, 128, dtype=torch.bfloat16, device=device)

    def call(self, name, tensors, integers=()):
        code = getattr(self.lib, name)(*[x.data_ptr() for x in tensors], *integers,
                                      torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f"{name}: device launch error {code}")

    def route(self, q):
        if q.shape != (8, 4, 128) or q.dtype != torch.bfloat16 or not q.is_contiguous():
            raise ValueError("contiguous BF16 Q[8,4,128] required")
        self.call("pack", [q, self.a])
        torch.bmm(self.a, self.matrix, out=self.prod)
        self.call("fused_norm_top", [self.prod, self.ln, self.n, self.features, self.valid, self.raw])
        self.call("launch_union_warp", [self.bits, self.features, self.valid,
                  self.mask, self.tiles, self.ids, self.count], [8, self.history, 2])

    def attention(self, q, k, v, *, splits=64, schedule=0):
        if not 1 <= splits <= 128 or schedule not in (0, 1, 2):
            raise ValueError("unsupported attention partition/schedule")
        if k.shape != v.shape or k.shape[0] != 8 or k.shape[-1] != 128 or k.shape[1] < self.history:
            raise ValueError("K/V[8,total_length,128] required")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise ValueError("BF16 KV with contiguous head dimension required")
        if k.stride(1) != 128 or v.stride(1) != 128:
            raise ValueError("contiguous per-group KV rows required")
        self.call("attention_tuned", [q, k, v, self.ids, self.count, self.state, self.out],
                  [self.history, k.shape[1] - self.history, k.stride(0), v.stride(0), splits, schedule])
        return self.out

    def __call__(self, q, k, v, **kwargs):
        self.route(q)
        return self.attention(q, k, v, **kwargs)

    def buffer_bytes(self):
        """Live tensors owned here; excludes model/KV, caller indices, allocator reserve and build temporaries."""
        return {name: x.numel() * x.element_size()
                for name, x in vars(self).items() if isinstance(x, torch.Tensor)}
