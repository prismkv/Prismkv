# Direct Q activation versus dense moment routing on Qwen3.8-27B

Both arms use the same frozen 3,000-feature SAE and all 64 standard-attention
KV groups. They share prefill, key memberships, indexed attention and dense
generated-tail access. The direct selector uses up to 16 positive SAE
features per Q head. It normalizes their activation weights within each Q,
averages across six Q heads and takes a shared deduplicated token union in
score order. A fixed random priority resolves the boundary bucket.

| Protocol | Suite | Paired scores | Paired read records | Dense moment score | Q activation score | Q delta | Moment reads | Q reads |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8% cap | LongBench | 3,750/3,750 | 3,548 | 54.057 | 53.550 | -0.507 | 7.9999% | 7.9959% |
| 8% cap | RULER | 3,250/3,250 | 3,199 | 96.731 | 95.901 | -0.830 | 7.9999% | 7.9870% |
| Native Top16 read mean | LongBench | 3,748/3,750 | 3,546 | 53.996 | 53.687 | -0.309 | 8.7352% | 8.7329% |
| Native Top16 read mean | RULER | 3,250/3,250 | 3,199 | 96.766 | 95.739 | -1.027 | 7.1891% | 7.1554% |

The fixed-cap moment arm ranks all nonempty buckets and stops at the
rounded 8% unique-read budget. The native Top16 control first runs the
uncapped moment selector. The direct-Q arm then uses its rounded
time-averaged count as a cap for each sample, layer and KV group. This is
an offline comparison with arm-specific generated trajectories. The direct
selector can read fewer tokens if the activated lists do not fill the cap.

Quality macro-averages 16 LongBench tasks or 65 RULER task-length cells,
using archived scores. Read rates average per-case ratios on paired cases
with sparse decode events and exclude generated tails and dense warmup.
Two matched LongBench samples (GovReport and QMSum) failed the budget
assertion. They remain unscored. Allowing any 0–100 score for either arm
on those cases bounds the complete LongBench macro delta to
[−0.372, −0.247] points. No failed sample has been rerun.

| RULER length | 8% cap, moments | 8% cap, Q | Matched, moments | Matched, Q |
| --- | ---: | ---: | ---: | ---: |
| 8K | 96.564 | 94.892 | 96.667 | 94.969 |
| 16K | 97.026 | 96.344 | 97.026 | 95.710 |
| 32K | 96.769 | 95.295 | 96.615 | 95.859 |
| 64K | 96.680 | 96.664 | 96.959 | 95.882 |
| 128K | 96.615 | 96.308 | 96.564 | 96.277 |

The audit checked 13,998 completed two-arm result hashes, input metadata,
source hashes and recorded read budgets. It reuses the original grades;
no scoring rerun, independent significance test or latency comparison was
performed. These controls isolate the selector with fixed SAE weights and
do not establish performance for smaller SAE dictionaries.

[Per-task and length audit data](direct_q_quality.json)
