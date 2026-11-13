# Verified quality/read tables

[Full 27B LongBench ablations](longbench27b_ablations/README.md) report overall,
category and task scores for dictionary width, fixed Key memberships, bucket
cutoff and score terms, with 3,750 examples per configuration.

[27B quality and latency by bucket count](bucket_count_pareto_27b/README.md)
pairs the LongBench macro scores with 128K Graph request timings for
Top1, Top2, Top4, Top8, Top16, and Full Flash.

[Hyperparameter sensitivity](hyperparameter_sensitivity/README.md) adds 128K
decode timings for dictionary width and key memberships and 32K/64K index
update interval controls.

[27B additional LongBench baselines and PPU latency](ppu_baseline_latency.md)
reports LongBench scores and 128K decode timings, including the paired Full and PrismKV CUDA Graph run.

[H200 system evaluation](h200/README.md) reports paired single-request and
continuous-batching results, index storage, candidate merging and KV layout
controls. It links the separate PPU-ZW810 offline phase audit.

[27B shared-feature replication on 200 long contexts](shared_features_27b_replication/README.md)
expands the original 14-context suite at matched 32K and 64K lengths and links
the task-by-layer heatmaps, coverage tables, and case-level results.

The RULER and LongProc tables below re-audit existing submitted-model records. They are not new
benchmark runs, new SAE configurations, official-method latency reproductions
or evidence of a causal SAE advantage. The label27B follows the submission;
the model's standard/linear-layer mix is described in the model manifest.

## RULER:13tasks x5lengths x50examples

- [Score tables by context length](ruler27b_by_length.md) compare all seven
  methods, including KVzip, across the 13 tasks at 8K, 16K, 32K, 64K, and 128K.
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

## Full 27B RULER bucket construction and routing controls

[Full 27B RULER organizer controls](organizer_full_ruler/README.md) compare
SAE, multi-assignment K-means, sparse random projection, and three LSH routing
variants on 3,250 inputs across 8K–128K. The sparse methods share the indexed
Full trajectory's read schedule, calibrated with native PrismKV Top16.
The [earlier 26-example pilot](organizer_pilot/README.md) remains available.

## Additional actual-read audit

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

## Direct Q routing at comparable reads

[Paired 27B LongBench/RULER controls](direct_q_quality.md) compare weighted
Q activations with dense moments under an 8% cap and sample/layer/group
read-mean matching. The archive includes 13,998 verified paired records.

## Quest GQA selection control

[Paired 26-case RULER pilot](quest_gqa_control/README.md) uses the same timed
Quest selector in quality and eager timing, with a shared indexed AV kernel.
Both per-Q normalization and raw-mean aggregation are reported.
