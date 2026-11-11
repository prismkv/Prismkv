# 27B shared-feature replication on 200 long contexts

This follow-up expands the Appendix B task suite from 14 historical contexts to 200 fresh ones. The new cohort contains 100 nominal 32K and 100 nominal 64K contexts across UUID retrieval, nested JSON, HTML-to-TSV, code patches, staged evidence, theory of mind, and Countdown. Each task and length has 14 or 15 new instances. The original test templates, evidence layouts, and reference-length targets are retained. Instances and demonstrations are newly generated.

## Shared-feature coverage

Each percentage is the fraction of historical keys sharing at least one SAE feature with the current query. Both checkpoints use identical post-RoPE Q/K vectors and exact Full-attention ranks on each trajectory. The 27B model has 16 standard attention layers with four KV groups and six Q heads per group, for 64 KV groups and 384 Q heads across these layers. The new cohort contributes 3,684,864 query-head observations per checkpoint.

| SAE checkpoint | Historical key partition | Old 14, top | Old 14, rest | New 200, top | New 200, rest |
| --- | --- | ---: | ---: | ---: | ---: |
| With Q–K matching | Top 1% vs. rest | 59.32% | 4.73% | 59.71% | 4.75% |
| With Q–K matching | Top 2% vs. rest | 47.92% | 4.41% | 48.26% | 4.43% |
| Without Q–K matching | Top 1% vs. rest | 37.62% | 5.15% | 37.66% | 5.17% |
| Without Q–K matching | Top 2% vs. rest | 31.85% | 4.94% | 31.86% | 4.96% |

These averages include all 16 standard attention layers. Each task-by-length cell receives equal weight. With matching, the Top 1% coverage for the six layers displayed in Table 7 changes from **66.50% to 66.98%**. Without matching, the Top 1% and Top 2% comparisons each favor the high-attention keys in all 200 context averages. The observation holds with a checkpoint trained without the matching objective. The two frozen checkpoints have different training histories, so their difference does not identify the isolated causal effect of that loss. This diagnostic measures feature overlap on Full trajectories. Benchmark scores and realized read rates come from separate free-generation evaluations.

| SAE checkpoint | Length | Old 14 Top 1% | New 200 Top 1% | Old 14 Top 2% | New 200 Top 2% |
| --- | --- | ---: | ---: | ---: | ---: |
| With Q–K matching | 32K | 61.64% | 62.40% | 49.66% | 50.32% |
| With Q–K matching | 64K | 57.01% | 57.02% | 46.19% | 46.20% |
| Without Q–K matching | 32K | 35.93% | 36.22% | 30.64% | 30.83% |
| Without Q–K matching | 64K | 39.31% | 39.10% | 33.06% | 32.89% |

## Protocol and files

- Full greedy generation uses the original 256-token prefill chunks. The probe samples decode positions 1, 9, 17, and so on until EOS or the original per-case cap and retains up to 100 equally spaced positions from this sequence. Replay checks on one original 32K and one original 64K case reproduced the matching-checkpoint archive within (3	imes10^{-8}) in absolute coverage fraction.
- For a prompt of length (H), the historical K support uses the positive entries among the global top (8H) activations per KV group. The current Q uses its positive Top 16. Current K uses the historical positive activation cutoff. For both current Q and current K comparisons, the historical partitions use the current Q's exact Full-attention ranking of prompt keys. The top partition has (lceil pH/100
ceil) keys for a cutoff of (p) percent.
- **32K heatmap** [PDF](coverage_32k.pdf) · [SVG](coverage_32k.svg). **64K heatmap** [PDF](coverage_64k.pdf) · [SVG](coverage_64k.svg). Both use the same color scale and show new and old coverage side by side for each task, layer, cutoff and checkpoint.
- [Summary JSON](comparison.json) contains both cutoffs, their complements, the 32K and 64K strata, and the six-layer comparison. [Task-by-length-by-layer CSV](by_task_length_layer.csv) provides coverage from at least 1 to at least 16 shared features for the current Q and current K comparisons.
- [Per-context, per-layer, per-group measurements](case_layer_group.jsonl.gz) provide the new observations. Within each cutoff, rows are ordered as current Q versus top, current K versus top, current Q versus rest, and current K versus rest. Entries are fractions at thresholds 1 through 16.
- The [27B LongBench ablation tables](../longbench27b_ablations/README.md) document equal-read score-term controls separately.
