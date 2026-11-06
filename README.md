# PrismKV

Reference implementation and kernel sources for **Finding the Needles in
Long-Context Attention through Sparse Features** (anonymous ICLR 2027 submission).

PrismKV keeps the backbone frozen. A shared query/key SAE assigns historical keys
to overlapping feature buckets. The submitted decode router scores counts,
means, and diagonal variances of the **original keys** using the original query.
It does not encode the query with the SAE online. The final attention uses the
selected original KV rows and the complete unindexed generated tail.

Start with [the reviewer guide](reviewer_guide/README.md). In particular, the
shared-feature observation motivates the organization; it is not a proof that
Q/K matching supervision is necessary for downstream quality.

## Status and scope

- `src/prismkv/core.py`: portable, shape-parametric PyTorch correctness reference.
- `src/prismkv/training.py`: reconstruction, anti-dead-feature, utilization and
  matching objectives, with explicit externally mined training inputs.
- `kernels/ppu_8b/` and `src/prismkv/native8b.py`: PPU-specific measured-path
  kernel sources and a standalone binding. F=3000, 8 KV groups, 4 Q heads per
  group, d=128, 16 selected buckets. They require the PPU SDK.
- `tests/`: CPU mathematical and state-update checks.
- `examples/`: deterministic synthetic replay; not a downstream score.
- `src/prismkv/hf_reference.py`: an all-group Hugging Face quality/reference
  adapter. Tiny random Qwen3 backbone tests cover dense equivalence, cached
  prefill, actual generation, strict caps and incremental updates.
- `src/prismkv/weights.py`: hashed, per-group, inference-only encoder format.

The current suite has 16 passing CPU tests, including four tiny-backbone tests.
The PPU8B kernel library compiles and its standalone GPU correctness witness
passes six synthetic cases through64K history: exact selected features/IDs and
36 AV checks, maximum relativeL2 error0.001683. This does not reproduce the
paper's latency, SAE index-construction performance or pretrained-model quality.
Pretrained encoders for all 288 8B and 64 hybrid-27B dictionaries are included
under `pretrained/`, with exact tensor/scales and source-hash packaging checks.
The 8B training-context fingerprint manifest is included; full checkpoint
lineage, model-scale regression and benchmark launchers are still being
prepared. Test results do not establish
new downstream accuracy or speed improvements.

## Run the portable reference

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/replay.py --history 1024 --tail 32 --device cpu
```

For the optional Hugging Face reference:

```bash
python -m pip install -e '.[hf]'
python -m unittest discover -s tests -v
python examples/generate_hf.py --model /path/to/Qwen3-8B \
  --encoders pretrained/Qwen3-8B --local-files-only
```

The bundle must contain every standard-attention layer/KV group with saved Q/K
scales; a different model's bundle is not interchangeable. See
[the bundle notes](pretrained/README.md) for checkpoint histories and validation
limits. The generation example targets the ordinary Qwen3 causal
LM loader. The hybrid27B loader/runtime and its native timing path need separate
integration and validation; matching an attention class name does not establish
that full-model compatibility. Requests are batch-one, unpadded, causal and
without sliding-window/soft-capped attention. This path is not optimized serving.

PyTorch >=2.4 and Python >=3.10 are required. The portable reference uses Python
loops and is intended for inspection, not latency measurement. It gathers KV to
check the answer; the native attention kernel directly reads indexed original KV.

For a PPU-SDK build and standalone synthetic check:

```bash
python -m pip install -e '.[native]'
python examples/check_native8b.py --build-dir output/build_native8b
```

The build directory must not already exist. Alternatively pass `--library` to a
previous build. Results and boundaries are in
[`native8b_validation.json`](reviewer_guide/native8b_validation.json).

## Protocol distinctions

The submitted default selects at most 16 nonempty buckets. The resulting number
of unique historical tokens varies. An optional strict fractional unique-token
cap is an **experimental selection policy**, not an undocumented property of
the submitted Top16 rule. Q-activation routing is also a separate ablation.

Initial full prefill, index construction, graph startup, steady-state routing,
indexed attention, full-model decode, and periodic updates are separate costs.
Report generated-tail reads separately from selected old-history reads.

The original KV cache is single-copy, but the optimized bitmap index is not free:
it uses F*ceil(H/32)*4 bytes **per KV group**, in addition to moments, membership
lists if retained, workspaces and allocator overhead. See the systems guide.

## Anonymity

This tree contains no server addresses, private experiment paths, personal Git
history or reviewer text. Model and dataset licenses remain with their owners.
The repository does not redistribute model weights or benchmark datasets.
