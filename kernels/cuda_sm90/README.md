# NVIDIA SM90 kernels and CUDA Graph runner

This path contains the supplementary NVIDIA H200 implementation for the 27B
hybrid checkpoint. The original KV cache uses BF16 and a slot-major layout.
Six query heads share the routed IDs for each KV group. Each query head still
computes its own attention weights.

- `hopper_top16.cu` keeps the FP32 score definition and tie order. It normalizes
  the six query heads in parallel and selects Top16 with warp reductions. The
  file also contains the measured radix and sharded controls.
- `native_sources/union.cu` merges feature bitmaps and emits sorted, unique
  historical IDs. The runtime continues to include the full generated tail.
- `hopper_av.py` directly reads indexed original KV and the contiguous tail.
  It uses Tensor Core tiles, split-K local softmax states and a final FP32
  merge. The main measured variant uses scaled FP16 probabilities for the PV
  product. This arithmetic can change greedy outputs relative to the legacy
  AV kernel even with identical candidates.
- `serving.py` captures one decode graph per active extent over preallocated
  KV and recurrent states. Its FIFO scheduler admits new requests, prefills
  them serially and reuses freed slots. Full and sparse share that scheduler.

## Build and check

The recorded runtime used Python 3.11, PyTorch 2.8.0 with CUDA 12.8, Triton
3.4.0 and FlashAttention 2.8.3. On Linux with this PyTorch already installed,
`setup_runtime.sh` prepares the additional dependencies. A CUDA toolkit with
SM90 support and its CUB headers is required to build the native libraries.

```bash
cd kernels/cuda_sm90
bash setup_runtime.sh
.venv/bin/python build.py
.venv/bin/python check_hopper.py
.venv/bin/python run.py check --short --out build/native_check.json
```

The synthetic checks use BF16 tensors and run without model weights or
benchmark inputs. They compare sorted candidate IDs, attention output within a
tolerance and inactive slots. The measured benchmark runtime fingerprints are
listed with the [H200 results](../../reviewer_guide/results/h200/README.md).

## Model-scale runner

The historical model-scale workload uses a matching local 27B backbone and a
staged asset directory. The expected identities and filenames are in `pack.py`.
The staged `weights.pt` is the historical aggregate encoder file. It has a
different serialization from the per-group encoder distribution in
[`pretrained/`](../../pretrained/README.md). Archived prompt bundles and
backbone shards are separate inputs to this runner.

```bash
.venv/bin/python run.py serve --model "$MODEL" --assets "$ASSETS" \
  --history 128 --batch 4 --engine graph --output-cap 4096 \
  --av-backend hopper --route-backend hopper-warp \
  --method full --out outputs/full.json
.venv/bin/python run.py serve --model "$MODEL" --assets "$ASSETS" \
  --history 128 --batch 4 --engine graph --output-cap 4096 \
  --av-backend hopper --route-backend hopper-warp \
  --method sparse --out outputs/prismkv.json
```

`run.py single` provides the fixed-length single-request protocol.
`benchmark_hopper_route.py` and `diagnostics/serving_graph_stages.py` provide
module timing from archived Full receipts and use `staged/` by default.
Synthetic checks and archived measurements have separate scopes.
