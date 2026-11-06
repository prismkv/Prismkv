"""Deterministic synthetic demonstration, not a downstream benchmark."""
import argparse
import json
import torch
from prismkv.core import SharedSAE, build_index, select_buckets, attend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=int, default=1024)
    p.add_argument("--tail", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    torch.manual_seed(5)
    device = torch.device(a.device)
    q = torch.randn(4, 128, device=device)
    k = torch.randn(a.history + a.tail, 128, device=device)
    v = torch.randn_like(k)
    sae = SharedSAE().to(device)
    index = build_index(k[:a.history], sae, 1.0)
    ids, meta = select_buckets(q, index)
    output = attend(q, k, v, ids, a.history)
    print(json.dumps(dict(history=a.history, full_tail=a.tail,
        bucket_count=len(meta['features']), raw=meta['raw'], unique=meta['unique'],
        unique_old_ratio=meta['unique']/a.history, output_shape=list(output.shape),
        note='random untrained SAE; demonstration only')))


if __name__ == "__main__":
    main()
