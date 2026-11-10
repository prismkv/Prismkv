# Hyperparameter sensitivity

## 27B dictionary width, key memberships and bucket count

Quality and prompt reads come from all 3,750 LongBench examples and average
the 16 tasks equally. Read rates count unique old prompt positions on routed
steps and exclude the generated tail. The width sweep uses a common 8% old
prompt read rule. The fixed membership sweep uses the same Top16 bucket router
and reads the full union of the selected buckets.

Timing uses a separate batch-1 Qwen3.8-27B workload on PPU-ZW810 with the same
128K prompt and 8,193 decode steps. Both repetitions for each configuration
run on the same card. The card stays fixed within the width and membership
sweeps. Top4 and Top8 use different cards from Full. The decode
column is the mean Graph decode time divided by 8,193. Prefill, initial
index build, periodic updates and Graph capture are outside that column.
The latency request runs to a fixed output length. See also the
[full LongBench ablations](../longbench27b_ablations/README.md) and the
[Top1–Top16 quality and request latency curve](../bucket_count_pareto_27b/README.md).

| Sweep | Configuration | LongBench score | Prompt reads | Graph decode (ms/step) |
|---|---|---:|---:|---:|
| Dictionary width | 1,000 | 53.851 | 7.991% | 54.360 |
| Dictionary width | 2,000 | 53.814 | 7.991% | 54.343 |
| Dictionary width | 3,000 | 53.925 | 7.991% | 54.763 |
| Dictionary width | 4,000 | 53.953 | 7.991% | 55.092 |
| Fixed memberships | 1 per key | 54.004 | 3.353% | 47.064 |
| Fixed memberships | 2 per key | 53.977 | 4.704% | 47.251 |
| Fixed memberships | 4 per key | 54.073 | 6.568% | 47.545 |
| Fixed memberships | 8 per key | 54.141 | 9.150% | 48.188 |
| Fixed memberships | 16 per key | 54.123 | 13.022% | 48.981 |
| Bucket count | Full Flash | 54.034 | 100% | 50.756 |
| Bucket count | Top1 | 49.633 | 0.80% | 46.541 |
| Bucket count | Top2 | 51.781 | 1.44% | 46.591 |
| Bucket count | Top4 | 52.959 | 2.58% | 47.024 |
| Bucket count | Top8 | 53.527 | 4.71% | 47.354 |
| Bucket count | Top16 | 53.996 | 8.67% | 48.062 |

## 8B index update interval

Qwen3-8B uses Top16 routing on PPU-ZW810 at batch size one. The replay samples
89 positions from a fixed Full continuation for one input at each length.
Read counts include the indexed positions and unindexed dense tail, averaged
over sampled steps and KV groups. The latency includes the synchronous update
cost amortized over the 4,608-step replay window. PPL is conditional on the
fixed Full continuation. Both inputs reach their first EOS before step 4,096,
so the 4,096-token update is measured during the forced continuation.

| Prompt | Update interval (tokens) | Unique reads (tokens/group) | Conditional PPL | Eager decode with amortized update (ms/step) |
|---|---|---:|---:|---:|
| 32K | No update | 4,959.9 | 1.2721 | 20.881 |
| 32K | 512 | 3,334.6 | 1.2982 | 20.957 |
| 32K | 2,048 | 3,886.1 | 1.2754 | 20.624 |
| 32K | 4,096 | 4,631.4 | 1.2693 | 20.572 |
| 64K | No update | 7,513.5 | 1.2750 | 22.131 |
| 64K | 512 | 5,601.9 | 1.2836 | 22.539 |
| 64K | 2,048 | 6,243.2 | 1.2739 | 22.291 |
| 64K | 4,096 | 7,161.1 | 1.3065 | 22.311 |

In a separate seven-case LongProc diagnostic with free generation, one
Countdown example scored 100 with no update and with a 512-token interval,
and scored zero with intervals of 2,048 and 4,096. These selected cases
show that the interval can affect task quality. The best interval remains
open across tasks and context lengths.
