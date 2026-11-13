# Full 27B RULER bucket construction and routing controls

Qwen3.8-27B was evaluated on 3,250 RULER examples across 13 tasks and five context lengths (8K, 16K, 32K, 64K, and 128K), with 50 examples per task and length. This is the comparison reported in Appendix D.2, Table 22. Each length averages its 13 official task scores. The macro average weights all 65 task-length cells equally.

| Method | 8K | 16K | 32K | 64K | 128K | Macro average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full indexed attention | 96.769 | 97.077 | 96.769 | 96.856 | 96.615 | 96.817 |
| PrismKV (SAE) | 96.667 | 97.077 | 96.769 | 96.959 | 96.718 | **96.838** |
| Multi-assignment K-means | 95.577 | 92.615 | 89.223 | 87.625 | 87.885 | 90.585 |
| Sparse random projection | 94.708 | 90.931 | 86.548 | 86.179 | 86.987 | 89.071 |
| LSH + bucket-summary routing | 86.446 | 74.256 | 67.354 | 68.448 | 71.638 | 73.629 |
| LSH + learned routing | 78.877 | 71.744 | 67.838 | 67.725 | 69.062 | 71.049 |
| Random-hyperplane LSH + Hamming routing | 57.179 | 58.815 | 62.295 | 69.218 | 73.846 | 64.271 |

PrismKV leads the five sparse controls at every length. Its macro score is 0.021 points above indexed Full and 6.253 points above multi-assignment K-means, the strongest non-SAE control in this evaluation.

## Budget and generation

For each prompt, native PrismKV Top16 sets the number of unique old-history tokens to read at each routed decode step, standard attention layer, and KV group along the indexed Full trajectory. The six sparse methods share those counts. Each method generates independently and retains full access to its generated tail. If an output outlasts the indexed Full trajectory, the last recorded budget is reused. The shared schedule fixes the count at each matched position, while the mean realized reads can vary across independently generated outputs. The dense warmup is outside the routed read counters.

All sparse methods use the same indexed attention operator. K-means assigns each key to its eight nearest frozen centers among 3,000. Sparse random projection assigns each key to up to eight buckets with positive projection scores. Both use the bucket-summary router used by the matched SAE control. The three random-hyperplane LSH variants share bucket memberships and differ in routing. The K-means centers and learned LSH router are frozen during evaluation.

## Per-task data

[Task and length data (CSV)](task_results.csv) includes the official task score and mean realized unique old-history read percentage for each method in every task-length cell. Sparse read percentages average examples with routed decode steps. Indexed Full reads the complete history. Generated-tail reads are excluded from the percentages.

The [26-example pilot](../organizer_pilot/README.md) documents the earlier experiment.

## Separate read-policy comparisons

These Qwen3.8-27B results support the organizer comparison in the response to reviewer 8afp Q1. They cover the same 3,250 RULER examples and use two read policies. Multi-assignment K-means assigns each key to its eight nearest frozen centers and targets the archived PrismKV mean unique reads for each example. Sparse Gaussian projection replaces the SAE encoder with a fixed random projection and uses native Top16 bucket routing. The Full and SAE columns are the archived references used by the Gaussian comparison.

| Context length | Full Attention | PrismKV (SAE) | Multi-assignment K-means | Sparse Gaussian random projection |
| --- | ---: | ---: | ---: | ---: |
| 8K | 96.923 | 96.872 | 95.577 | 93.846 |
| 16K | 96.923 | 96.872 | 94.154 | 94.564 |
| 32K | 96.615 | 96.462 | 91.915 | 94.615 |
| 64K | 96.825 | 96.752 | 91.241 | 95.933 |
| 128K | 96.462 | 96.410 | 89.954 | 95.654 |
| **Overall** | 96.750 | 96.673 | 92.568 | 94.922 |

For the projection comparison, PrismKV and Gaussian projection read 7.03% and 7.73% of old tokens, respectively, weighted by routed steps and KV groups. On all 3,750 LongBench examples, the archived PrismKV and multi-assignment K-means macro scores are 53.963 and 54.061, respectively. The per-step matched-count results appear in the first table above.
