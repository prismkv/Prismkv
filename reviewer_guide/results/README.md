# Verified quality/read tables

[Full 27B LongBench ablations](longbench27b_ablations/README.md) report overall,
category and task scores for dictionary width, fixed Key memberships, bucket
cutoff and score terms, with 3,750 examples per configuration.

[H200 system evaluation](h200/README.md) reports paired single-request and
continuous-batching results, index storage, candidate merging and KV layout
controls. It links the separate PPU-ZW810 offline phase audit.

[Expanded 27B shared-feature analysis](shared_features_27b_expanded/README.md)
reports coverage with and without Q–K matching across 116 contexts and all
16 standard attention layers, with vector heatmaps and detailed measurements.

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

## Fixed-bucket scoring terms and a related selector

- [Seven score-term combinations](score_terms_pilot/README.md): same58-case
  pilot, fixed SAE memberships and matched unique counts. Includes every
  count/mean/variance subset, rather than only favorable combinations. It does
  not establish that all three terms are indispensable.
- [Squeezed Attention reference port](squeezed_pilot/README.md):29cases with
  source-style per-Q threshold and an explicitly separate GQA shared-budget
  adaptation. Original-style group union reads are larger. Fourteen adapted
  cases are infeasible, left unscored; paired comparisons use identical cases.
  Torch/backend/context-split differences and lack of runtime claims are explicit.
# Additional actual-read audit

[Qwen3-8B historical-token reads](actual_reads_8b/README.md) provides task/length
counters with explicit zero-event denominators. It also discloses a newly found
50-case shared prefill-first-token mismatch in the old RULER baseline cohort;
counter verification is not certification of a strictly paired quality result.

[SCBench and remaining KVzip read tables](remaining_reads.md) provide the 27B
SCBench mode and task counters plus the 8B and 27B KVzip LongBench and 27B RULER
read rates. The audits cover 23,658 archived result receipts. The tables retain
zero-event cases in the quality denominator and describe the actual sparse-read
denominators separately.

## Native phase latency and allocator memory

[Same-kernel32K/64K diagnostic](native_stage_profile/README.md) reports five
isolated stage graphs, complete module graphs, Torch allocator counters and
a deduplicated native-tensor inventory. It verifies576group output checks and
the original compile manifests. Isolated phases are not additive percentages
of the full chain; allocator peaks are not physical device peaks. No new-card,
serving or full-model-generation claim is made.

## Fixed-index LongProc rerouting control

[Complete16-case pilot](longproc_rerouting_pilot/README.md) compares current-query
reranking against frozen first-step ranking with the same SAE index, indexed
attention and full generated tail. Dynamic/frozen scores56.250/7.132; Full62.374.
The8K HTML case loses quality. This is an offline teacher-count diagnostic, not
KVzip, a causal decomposition of the historical benchmark gap, or latency.

## Llama appendix scope and actual reads

[Llama Table 6 scope and five-length RULER audit](llama_appendix_scope.md)
reports the omitted 8K–32K scope, all archived length scores, common
read denominators and archive provenance limitations.

## Direct Q routing at comparable reads

[Paired 27B LongBench/RULER controls](direct_q_quality.md) compare weighted
Q activations with dense moments under an 8% cap and sample/layer/group
read-mean matching. The archive includes 13,998 verified paired records.

## Quest GQA selection control

[Paired 26-case RULER pilot](quest_gqa_control/README.md) uses the same timed
Quest selector in quality and eager timing, with a shared indexed AV kernel.
Both per-Q normalization and raw-mean aggregation are reported.

## Bucket-budget quality and module latency

[Paired Top4/8/16 pilot](bucket_budget/README.md) reports official scores, actual reads, complete-module Graph latency and separate eager model timing on the same 26 RULER development inputs. Short-context slowdowns and unchanged Top16-compute overhead are retained.

## Expanded budget follow-up

[195 nonpilot RULER cases](bucket_budget_followup/README.md) extend the same budget test to five samples per task at8K/16K/32K. Top8 loses4.453points toTop16; the small-pilot trend does not support a new default. Actual reads and complete-module Graph/eager-model timings are separate.

## SeerAttention-R quality control

The paired Qwen3-8B study compares the released Seer gate with PrismKV on 26 pilot and 195 follow-up RULER prompts. The source-style and full-tail policies share the indexed AV backend. The report gives realized old-history and generated-tail reads by arm and length. [Results and protocol](seer_gqa_control/README.md)
