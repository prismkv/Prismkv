# Model-matched SAE encoder bundles

These are historical main-method SAE encoders, not backbone model weights.
There is one tensor-only file per standard-attention layer/KV group, containing
only `W_enc` and `b_enc`. Each manifest records shapes, saved Q/K scales, file
SHA256 values, source checkpoint hashes and local model/tokenizer fingerprints.
Decoder weights, optimizer state, machine paths and private checkpoint metadata
are deliberately omitted.

| Bundle | Dictionaries | Head dimension | Features | K activation budget | Q training TopK |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 288 (36 layers x 8 groups) | 128 | 3000 | global mean budget 8 | 16 |
| Qwen3.8-27B, submission label | 64 (16 standard layers x 4 groups) | 256 | 3000 | global mean budget 8 | 16 |

The Qwen3-8B source checkpoints all have 4000 training steps and the same 28
synthetic training contexts: four each of UUID, nested JSON, HTML-to-TSV, code,
staged evidence, theory-of-mind and Countdown. Their auxiliary middle-position
matching samples positions in the middle 20%-80% of history when that region
has at least 2% teacher attention mass; this is not an evidence-span oracle.

The hybrid 27B pack contains 43 inherited and 21 fresh dictionaries. The
inherited route includes continuation on 112+168 training contexts; the fresh
21 use 28 contexts and 4000 steps. Thus these groups do not share one uniform
initialization/data/schedule history. Full lineage and dataset-duplication
auditing remain separate from exact tensor packaging.

Export validation checked each legacy source file against its DONE hash, then
loaded the new format with `weights_only=True` and compared all encoder tensors
and scales exactly. No rounding or quantization was applied. This validates
packaging, not pretrained-scale generation equivalence of the new reference
adapter. In particular, the optional example currently uses a Qwen3 causal-LM
loader; the hybrid 27B runtime still requires its own integration.

```bash
python examples/generate_hf.py --model /path/to/Qwen3-8B \
  --encoders pretrained/Qwen3-8B --local-files-only
```

Use a model and tokenizer matching the manifest. The base 8B config has 40960
positions; the paper's longer-context settings use a separately specified
extension. These files alone do not extend the backbone context window.
Licenses for the underlying backbone and datasets remain with their owners.
