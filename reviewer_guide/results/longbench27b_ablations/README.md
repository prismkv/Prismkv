# Qwen3.8-27B LongBench ablations

Every row uses all 3,750 examples across the 16 classic LongBench tasks in the
paper's evaluation. The tables report the saved official task scores on a
0–100 scale. Overall is the unweighted mean of 16 task scores. Each category
is the unweighted mean of its constituent tasks. The category names follow
the [LongBench task definitions](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md).

[Per-task scores](task_scores.csv) include all 320 arm–task rows. The
[category CSV](category_scores.csv) and [overall CSV](macro_scores.csv)
contain 120 and 20 rows, respectively. [Provenance](provenance.json) records
the original aggregate fingerprints and the task-to-category mapping.

## Quality by category

| Configuration | Read policy | Overall | Single-doc QA | Multi-doc QA | Summarization | Few-shot | Synthetic | Code | Routed old reads |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | Dense | 54.034 | 47.685 | 64.655 | 26.912 | 71.838 | 58.167 | 57.470 | — |
| Width 1,000 | Matched 8% | 53.851 | 47.179 | 64.648 | 26.728 | 72.504 | 58.167 | 56.052 | 7.991% |
| Width 2,000 | Matched 8% | 53.814 | 47.001 | 64.809 | 26.501 | 71.759 | 58.250 | 57.155 | 7.991% |
| Width 3,000 | Matched 8% | 53.925 | 47.204 | 64.746 | 26.737 | 72.136 | 58.167 | 57.003 | 7.991% |
| Width 4,000 | Matched 8% | 53.953 | 47.546 | 64.554 | 26.657 | 72.218 | 58.167 | 56.995 | 7.991% |
| Key BatchTop8 | Natural | 53.996 | 47.652 | 64.686 | 26.632 | 72.487 | 58.167 | 56.615 | 8.736% |
| Key fixed 1 | Natural | 54.004 | 48.009 | 64.770 | 26.673 | 71.675 | 58.167 | 57.177 | 3.353% |
| Key fixed 2 | Natural | 53.977 | 47.507 | 64.735 | 26.878 | 71.809 | 58.167 | 57.258 | 4.704% |
| Key fixed 4 | Natural | 54.073 | 47.464 | 64.633 | 26.978 | 72.427 | 58.167 | 57.166 | 6.568% |
| Key fixed 8 | Natural | 54.141 | 47.706 | 64.552 | 26.789 | 72.844 | 58.167 | 57.124 | 9.150% |
| Key fixed 16 | Natural | 54.123 | 47.818 | 64.722 | 26.755 | 72.287 | 58.167 | 57.439 | 13.022% |
| Top1 buckets | Natural | 49.633 | 43.869 | 61.162 | 24.112 | 64.856 | 58.167 | 47.901 | 0.814% |
| Top2 buckets | Natural | 51.781 | 45.007 | 63.312 | 25.108 | 69.415 | 58.083 | 51.899 | 1.461% |
| Top4 buckets | Natural | 52.959 | 46.148 | 64.429 | 25.991 | 70.659 | 58.167 | 54.667 | 2.607% |
| Top8 buckets | Natural | 53.527 | 46.361 | 65.117 | 26.333 | 71.579 | 58.167 | 55.967 | 4.760% |
| Top16 buckets | Natural | 53.996 | 47.652 | 64.686 | 26.632 | 72.487 | 58.167 | 56.615 | 8.736% |
| Count | Matched 8% | 40.718 | 30.527 | 51.846 | 18.246 | 55.691 | 56.708 | 34.575 | 7.991% |
| Mean | Matched 8% | 53.987 | 47.537 | 64.683 | 26.467 | 72.484 | 58.167 | 56.970 | 7.991% |
| Variance | Matched 8% | 47.778 | 40.887 | 61.676 | 22.562 | 61.439 | 57.833 | 44.548 | 7.991% |
| All terms | Matched 8% | 54.056 | 47.698 | 64.756 | 26.578 | 72.552 | 58.167 | 56.902 | 7.991% |

## Experimental controls

- **Dictionary width.** The four dictionary banks were trained with the same
  training-only trace bank, objective, seed per group, 4,000 steps and mean
  Key membership budget of eight. Width 3,000 was retrained for this sweep.
  Every routed step reads `floor(0.08 × H)` unique old tokens per KV group.
- **Key memberships.** The same submitted 3,000-feature encoders and Top16
  router are used throughout. Fixed-K selects exactly 1, 2, 4, 8 or 16 distinct
  buckets per token, ranked by encoder preactivation. It fills from the next
  preactivations when too few are positive. BatchTop8 is the original
  sequence-wide positive selection with a mean budget of eight. The CSVs
  report realized memberships and the fraction drawn from nonpositive
  preactivations. Quality can reflect changes in both memberships and reads.
- **Bucket cutoff.** The submitted 3,000-feature checkpoint and moment scores
  are fixed. Each arm reads whole buckets after selecting Top1, Top2, Top4,
  Top8 or Top16. All cutoffs use the same Top16 scoring implementation and
  mask excess winners before union and indexed attention.
- **Score terms.** The submitted 3,000-feature checkpoint and memberships are
  fixed. Each single-term arm keeps one of count, mean or diagonal variance.
  The all-terms arm uses the same 8% read rule as the single-term arms. The
  width and score-term controls trim the boundary bucket with a fixed,
  attention-independent token order.

The sparse arms route in all 16 standard attention layers and 64 KV groups.
Each KV group shares its selected old tokens across six Q heads. Generation
is greedy, with the original prompts and output caps, shared dense prefill,
a dense first decode forward and a fully visible generated tail. Routed old
reads are averaged over examples with routed decode steps and exclude both
the dense warmup and generated tail. The Full baseline has no sparse read
counter.

Per-task decode timing, where available, is instrumented eager timing on
PPU-ZW810. The paper-checkpoint index timer includes construction for the
native and exact-budget diagnostic paths together. Timings are specific to
these runs. Blank fields indicate unavailable or inapplicable measurements.
