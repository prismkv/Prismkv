# Verified quality/read tables

The RULER and LongProc tables below re-audit existing submitted-model records. They are not new
benchmark runs, new SAE configurations, official-method latency reproductions
or evidence of a causal SAE advantage. The label27B follows the submission;
the model's standard/linear-layer mix is described in the model manifest.

## RULER:13tasks x5lengths x50examples

- [Task-length scores and read counts](ruler27b_task_length.csv):390rows, one
  per method/task/length for Full, Prism, H2O, Quest, SnapKV and PyramidKV.
- [Summary/protocol](ruler27b_summary.json): official scoring-function hash,
  both rounding conventions and audited counts. All3250Full/Prism paired
  reference records and13000classic-baseline results had hash/identity checks.

The full quality denominator is3250. Aligned old-read means use3200cases with
a measured Prism sparse reference; the remaining50remain in quality scores.
The original VT/CWE setup uses an answer-only instruction and512output-token
cap; do not call it an untouched official-default prompt/decoding protocol.
Means cover each method's own output trajectory, not identical per-step/group
access. No measured read is replaced with its nominal target.

## LongProc:880paired examples in16subsets

- [Per-subset quality, reads and output length](longproc27b_subsets.csv).
- [Summary and policy audit](longproc27b_summary.json): all6160result/score
  pairs checked,55examples per subset. HTML uses F1; other families accuracy.

Prism and KVzip have full generated-tail eligibility. All880KVzip/Quest/
SnapKV/PyramidKV tail counters exactly equal the corresponding full-tail
formula across64groups. Prism's native source passes the full tail length;
its old-read counters are not an independent measurement of that tail formula.
H2O has a bounded-total attention set and lower old reads. These are logical
attention-access quality adapters, not measurements of compressed resident
cache memory. The37.740point Prism/KVzip gap is not a causal decomposition.

## Baseline and measurement boundaries

These rows describe the evaluated model-specific accuracy adapters. The27B
Quest accuracy adapter averages raw per-Q page scores; the8B timing adapter
normalizes perQ first. Sharing the final indexed-AV kernel does not make those
two selectors identical. See [budget protocol](../budget_protocol.md) and
[systems](../systems.md). New controlled comparisons and native model-scale
latency validation remain separate work.

## Supplementary organizer pilot

[Organizer pilot](organizer_pilot/README.md) reports a separate fresh58-case
common-attention comparison, including complete per-task CSV and aggregate JSON.
Its results are mixed: simpler organization scores higher on the small
LongBench cohort; SAE scores higher on the RULER cohort. The common indexed Full
and separate Flash Full, exact-count diagnostic, different training histories,
output caps and limited scope are explicit. This is not a complete benchmark,
causal SAE superiority claim, or a latency reproduction.
