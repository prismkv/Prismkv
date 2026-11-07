# Quality evaluation: inputs, generation, access accounting and official scores

This is the **portable quality/reference path**, not the measured native kernel
or a serving benchmark. It supports the standard Qwen3-8B backbone and the
hybrid 27B backbone's standard-attention layers; recurrent linear-attention
layers are left to the backbone. The latter currently requires the separately
fingerprinted development runtime described in [hybrid_reference.md](hybrid_reference.md).
Installing the ordinary `.[hf]` extra does not provide that hybrid runtime.

## What has actually been checked

See [the hash-checked execution report](evaluation_validation_v1.json):

- 24 ordinary CPU tests, no skips, including an original-backbone Full execution,
  independent reference NLL, preserved partial failures, access conservation,
  first-token decoding constraints and in-memory RoPE overrides.
- All 14,000 archived input cases: 3,750 LongBench and 3,250 RULER inputs for
  each of the two models. Token IDs, output caps and labels match the frozen
  input records. **This does not execute those 14,000 generations.**
- Official-metric routing on 29 synthetic prediction/label fixtures: the 16
  classic LongBench tasks and 13 RULER tasks. These are not model scores.
- One preselected 8K `niah_single_3` case per model, index zero, through the
  public prepare, Full, reference attachment, all-read, sparse and official-score
  commands. All three modes produced the same generated token IDs and scored
  100 on that single case. All 288/64 standard-attention dictionaries were used.
  Sparse reads, including the one dense warmup forward, were 8.26%/7.33% of
  initial-prompt opportunities. This is an execution witness, not a benchmark
  average, new quality improvement or exact historical-backend reproduction.

The [scoring follow-up](scoring_validation_v1.json) verifies the current source
hashes and RULER's official prediction cleanup, which the initial wrapper omitted.
All six archived predictions retain their scores after the correction. A separate
control with a tab inside a predicted span scores 0 before cleanup and 100 with
the official cleanup. No new model generation was needed for this CPU check.

Even with the same generated IDs, BF16 all-read and original-Full teacher NLLs
are not identical. The report retains those differences. No blanket probability
equivalence or downstream-quality guarantee is inferred from the witness.

## Obtain local inputs and official scorer sources

Model checkpoints and benchmark datasets are not redistributed or implicitly
downloaded. Obtain them under their respective licenses. Use local raw task
JSONL files from [LongBench](https://github.com/THUDM/LongBench) (classic version)
or [RULER](https://github.com/NVIDIA/RULER). Keep the official scorer checkout
separate from this repository; its source-file hashes are saved in score output.
LongBench is MIT-licensed and RULER is Apache-2.0-licensed; this package does not
vendor their scorer implementations.

The validated classic LongBench checkout is commit
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`. Its `LongBench/metrics.py`
requires the dependencies listed by that official checkout, including `rouge`,
`jieba` and `fuzzywuzzy`. Pin and record those dependencies in your scoring
environment. RULER metrics are loaded from `scripts/eval/synthetic/constants.py`
and its `postprocess_pred` function is loaded from `scripts/eval/evaluate.py`
without running the official CLI's argument parsing, downloads or shard merging.
The scoring follow-up records the exact hashes of both scorer files.
Do not substitute an ad-hoc substring or exact-match metric for official scoring.

## 1. Prepare one task with an explicit input protocol

For an ordinary Qwen3-8B LongBench task:

```bash
python examples/prepare_benchmark.py \
  --benchmark longbench --protocol qwen8_paper \
  --model /path/to/Qwen3-8B --input /path/to/narrativeqa.jsonl \
  --task narrativeqa --official-root /path/to/LongBench \
  --output-dir output/narrativeqa_inputs
```

For a RULER task, identify the actual raw field names and generation cap rather
than relying on a guessed default:

```bash
python examples/prepare_benchmark.py \
  --benchmark ruler --protocol qwen8_paper \
  --model /path/to/Qwen3-8B --input /path/to/ruler_task.jsonl \
  --task niah_single_3 --context-length 8192 --max-new-tokens 128 \
  --prompt-field prompt --answers-field answers \
  --output-dir output/ruler_inputs
```

The example cap is explicit; use your frozen task's cap for reproduction.
`--limit` intentionally marks a diagnostic subset. The formatter never silently
truncates RULER. Historical VT/CWE answer-only variants require the explicit
`--answer-only` flag and their recorded output cap.

Protocol differences are intentional and recorded:

| Setting | `qwen8_paper` | `qwen27_paper` |
|---|---|---|
| LongBench chat wrapping | 11 chat tasks; raw `trec`, `triviaqa`, `samsum`, `lcc`, `repobench-p` | All 16 tasks use non-thinking chat |
| LongBench context preparation | Midpoint truncation to fit 32,768 minus output cap | Raw prompt midpoint clipped to 65,536 before chat wrapping |
| 64K/128K RULER | Explicit YaRN factor 4, original length 32,768, maximum 131,072 | Model's recorded native context configuration |
| First sparse decode forward | Dense warmup, explicitly recorded | Same |
| Samsum stops | Checkpoint EOS plus newline; ban stops at first output only | Same |

Preparation saves final `input_ids`, canonical token SHA, labels, stop IDs,
per-row output cap, effective model overrides, protocol metadata and a hashed
completion record. It does not claim these are untouched official defaults.

## 2. Run original Full and the sparse reference

The following commands use the RULER prepared directory above; the same entry
point accepts prepared LongBench rows. Every output directory must be new.

```bash
python examples/evaluate_hf.py \
  --model /path/to/Qwen3-8B --encoders pretrained/Qwen3-8B \
  --local-files-only --attention-backend sdpa --mode full \
  --input output/ruler_inputs/rows.jsonl \
  --config-overrides-json output/ruler_inputs/model_overrides.json \
  --prefill-chunk-size 256 --output-dir output/ruler_full

python examples/attach_full_reference.py \
  --inputs output/ruler_inputs/rows.jsonl \
  --full-predictions output/ruler_full/predictions.jsonl \
  --output output/ruler_full_targets.jsonl

python examples/evaluate_hf.py \
  --model /path/to/Qwen3-8B --encoders pretrained/Qwen3-8B \
  --local-files-only --attention-backend sdpa --mode sparse \
  --router moments --selected-buckets 16 --update-interval 4096 \
  --input output/ruler_full_targets.jsonl \
  --config-overrides-json output/ruler_inputs/model_overrides.json \
  --prefill-chunk-size 256 --output-dir output/ruler_sparse
```

`full` uses the original backbone attention without the adapter; `all_read`
is a separate adapter arithmetic control. For the hybrid 27B execution witness,
use its model-matched encoders, `qwen27_paper`, and the documented **eager**
backend/runtime. The SDPA mask-elision policy must not be reused for eager
prefill; the loader and regression tests preserve explicit eager causality.
Saved model configuration files are never rewritten.

Main selection is moments/Top16 without a fractional cap. `q_activation` and
`--unique-cap` are separately labeled experimental policies. Chunks, runtime,
attention backend and RoPE initialization can affect BF16 results; token-format
parity alone does not certify equality with historical benchmark outputs.

## 3. Score the exact paired prediction set

```bash
python examples/score_benchmark.py \
  --benchmark ruler --official-root /path/to/RULER \
  --inputs output/ruler_inputs/rows.jsonl \
  --predictions output/ruler_sparse/predictions.jsonl \
  --output output/ruler_sparse_score.json
```

For LongBench, pass `--benchmark longbench` and its official checkout instead.
The wrapper requires unique, exactly paired IDs and identical canonical input
token hashes. It refuses to score a silently filtered subset. It reports each
official rounded task/length-cell score, the explicit number of cases, and
`macro_of_official_rounded_cells`. The latter is **not** a substitute for the
paper's historical unrounded per-example aggregation. Do not compare the two
without applying the same aggregation convention. Running one task/file does
not automatically constitute a full 16-task or 13-task-by-length evaluation.

## NLL, access and failure semantics

Free generation happens first. If explicit reference token IDs are provided,
teacher forcing then starts from a fresh cache. Full-generated references and
dataset targets have different labels. PPL is `exp(mean token NLL)` over the
entire supplied reference, using unmodified logits before decoding bans; it is
not necessarily perplexity on an independent natural-text corpus. A lower
Full-target PPL on one case is not evidence of improved task quality.

Access records distinguish initial-prompt unique/raw reads, newly indexed
generated tokens, and the complete unindexed generated tail. The initial-prompt
denominator remains fixed after periodic updates. Dense warmup and routed-only
fractions are also separated. Full counters are explicitly **policy-derived**;
adapter counters are instrumented. These are logical reference reads, not
physical memory transactions. They exclude linear-attention layers.

The portable path gathers selected KV for correctness and runs Python loops;
do not time it as the native attention kernel. There is no continuous batching
or shared-prefix serving here. SCBench reuse orchestration, native model-scale
benchmark/latency integration, a clean hybrid dependency recipe, and complete
checkpoint ancestry remain separate unfinished requirements.

A run writes `run.json`, `predictions.jsonl` and a hashed `DONE.json`. Failures
preserve partial output with `FAILED.json`. Existing output paths are refused;
there is no automatic overwrite, result filtering or hidden resume.
