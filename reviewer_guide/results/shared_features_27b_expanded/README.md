# Expanded 27B shared-feature analysis

This supplement addresses the concern that the earlier shared-feature evidence used few contexts or displayed only a subset of layers. It evaluates 116 additional contexts across all 16 standard attention layers of the submitted 27B model (zero-based indices 3, 7, …, 63). There are four KV groups and six query heads per group at each layer. The other layers use linear attention.

## Results

The table reports the percentage of historical keys sharing at least one feature with the current query. Ranking uses the current query's exact Full-attention logits over the prompt history. All rows use the same 116 contexts and replay positions.

| SAE checkpoint | Top 1% | Remaining 99% | Top 2% | Remaining 98% |
| --- | ---: | ---: | ---: | ---: |
| Without Q–K matching | 31.60% | 6.09% | 27.26% | 5.92% |
| With Q–K matching | 46.87% | 5.38% | 38.14% | 5.13% |

With the no-matching checkpoint, the coverage difference is positive in all 116 context averages, in all 16 layer averages, and in all 464 task-by-layer averages, under both cutoffs. For Top 2%, the overall gap is 21.35 percentage points, with a descriptive percentile bootstrap interval of [20.74, 21.98] obtained by resampling contexts from this cohort. The layer-average gaps range from 17.53 to 24.63 percentage points. The Top-2% gap is 21.42 points on LongBench and 21.26 points on RULER.

The no-matching checkpoint supports an association between feature sharing and high attention in the absence of a direct matching objective. The matching checkpoint shows stronger separation on this cohort. The checkpoints have different training histories, so their difference does not isolate the causal effect of the matching loss. This probe does not measure task accuracy or establish that matching supervision is necessary. The deployed router uses bucket summaries of the original keys and can select a bucket even when the current query and a member key have no shared SAE feature.

## Cohort and measurement

- LongBench contributes four examples per task across 16 tasks (64 contexts), selected from inputs between 2,048 and 32,768 tokens after formatting. RULER contributes two examples per task and length at the nominal 8K and 16K settings (52 contexts, 13 tasks). Samples are selected by a deterministic hash ordering within each stratum from eligible archived Full runs.
- Both frozen 64-group, 3,000-feature checkpoint packs are evaluated on identical post-RoPE Q/K vectors from Full replay. Eligibility requires at least two recorded output tokens. Replay consumes the first up to 32 recorded tokens before the final recorded token as inputs, giving 1,991 decode positions and 764,544 query-head observations per checkpoint. The data describe this task/length mix and these early output positions.
- Historical keys use positive activations among the global top \(8H\) entries for each prompt and KV group, where \(H\) is prompt length. Current Q uses its positive top 16. Current K uses the historical positive activation cutoff. The same Full Q ranks the history for both the Q-to-K and K-to-K comparisons. The top partition has \(\lceil pH/100\rceil\) prompt tokens for cutoff \(p\) in percent. Generated-tail keys are excluded.
- Coverage is averaged over query heads and replay positions within each context and KV group. Contexts and KV groups then receive equal weight. Context-level resampling summarizes variability in the observed cohort and treats repeated head observations as dependent. Task cells are based on four contexts each.
- The earlier 14-context report uses a different cohort. Its no-matching Top-2% comparison was 31.85% vs. 4.94%. The submitted appendix used Top 1% and displayed six layers. This supplement reports both cutoffs on a new cohort, so the headline percentages differ.

## Files

- All-layer coverage heatmap, with both checkpoints and cutoffs. [PDF](shared_feature_coverage_by_layer.pdf) · [SVG](shared_feature_coverage_by_layer.svg)
- Task-by-layer heatmap for the Top-1% minus remaining-99% gap without matching. [PDF](noqk_top1_enrichment_by_task.pdf) · [SVG](noqk_top1_enrichment_by_task.svg)
- [Layer CSV](by_layer.csv) and [layer/KV-group CSV (gzip)](by_layer_kv_group.csv.gz) include both cutoffs, both checkpoints, and thresholds of at least 1 through 16 shared features. They also include each benchmark and task separately.
- [Context manifest](contexts.csv), [context/layer/KV-group measurements (gzip JSON)](context_layer_group.json.gz), and [aggregate results](summary.json). In the compressed JSON, each cutoff array is ordered as Q-to-Top, K-to-Top, Q-to-Rest, K-to-Rest, with thresholds 1–16 in each row and rates on a 0–1 scale.
