# Hybrid 27B reference: executed scope and environment limits

The submitted model label is Qwen3.8-27B. Its saved configuration has
`model_type=qwen3_5`,64 text layers,16 standard-attention layers and48 linear-attention
layers. Standard attention uses 24 Q heads/4 KV heads/d256. The model label is not
a newly verified public model-download ID; compare the three checkpoint/config/
tokenizer fingerprints in the encoder bundle before using it.

## Execution witness

The public `examples/generate_hf.py` was executed against the actual checkpoint
and all 64 released encoders, with both `--all-read` and sparse routing. Both
returned 42 on the short prompt "What is 6 times 7? Answer with just the number."
There were 128 group steps per arm. All-read selected 3328/3328 old-token opportunities;
sparse selected 1051/3328, with 192 dense-tail accesses each. These aggregate
opportunities are over group-steps, not the number of distinct prompt tokens.
The short-prompt read fraction is not a long-context benchmark read rate.

[Source/output/encoder-hash validation](pretrained_hybrid_validation.json).
No missing/newly-initialized/unexpected-weight warning was present in the two
captured loader outputs. The audit checked the 64 encoder tensor files and saved
model/config/tokenizer index fingerprints; it did not rehash every backbone shard.

The reference uses the same SAE/moment selection rule;48 linear layers remain
untouched. A tiny four-layer hybrid CPU contract also verifies all-read logits,
exact greedy output, both standard-attention groups sparse, and incremental
tail indexing. This is not full 27B quality validation or the paper's optimized
native path.

## Separate runtime, explicit backend

The ordinary 8B optional dependency `.[hf]` pins Transformers4.57.1. That version
does not contain this hybrid model. The observed hybrid witness instead uses
an existing Torch2.4.0/Transformers5.16.0.dev0/numpy1.26.4 runtime on PPU-ZW810;
six framework-source hashes are included above. The existing staged framework
has no verified clean-install recipe or upstream commit mapping yet. Do not
interpret its version string as a complete environment lock, or install the 8B
extra and assume hybrid support. No global dependency installation is required
or performed by this repository.

With a compatible model-supporting runtime already available, the tested CLI
shape is:

```bash
python examples/generate_hf.py --model /path/to/the/matching/hybrid-checkpoint \
  --encoders pretrained/Qwen3.8-27B --local-files-only \
  --attention-backend eager --max-new-tokens 8 \
  --prompt "What is 6 times 7? Answer with just the number."
```

The reference loader selects
the backend only in memory, without rewriting checkpoint files. Eager is
explicit because this older Torch lacks the newer SDPA `enable_gqa` argument.
Eager prefill has its own explicit causal-mask registry; reusing SDPA mask
elision would make that path incorrectly bidirectional. The 24 ordinary CPU tests
include a regression for this issue and a saved-FA2 checkpoint-loader regression.

## Not established

The newer [evaluation pipeline](evaluation.md) also exercised one preselected
8K RULER request with the complete public preparation, generation and official
scoring commands. Full/all-read/sparse output IDs matched on that request,
but teacher NLLs differed. Input-format parity was checked for all 7,000 archived
27B inputs without generating them again.

The short CLI witnesses are not complete LongBench/RULER scores, new accuracy claims,
a native-kernel latency result, a serving implementation or proof that the
historical optimized backend and this reference are token-identical. Full
native benchmark integration, a clean hybrid dependency lock and
broader pretrained-scale comparisons remain incomplete. The historical paper
results must not be replaced by these execution checks.
