# Quest GQA selection and shared Attention control

This paired Qwen3-8B pilot covers one lexical-first case from each of 13
RULER tasks at 8K and 32K, 26 cases total. Every arm uses all 36 layers and
288 KV groups. Selection was frozen before inspecting these outcomes.
The same prefill feeds Full Flash, native PrismKV and two Quest-GQA arms.

The per-Q-softmax Quest arm uses the exact previously timed router source.
Both its free generation and timing call that class. The raw-mean arm uses
the same BF16 products and FP32 sum before averaging, so it isolates group
aggregation here. It does not exactly replay the older quality runner's
arithmetic. Both are GQA adaptations, with original-KV indexed Attention.

| Method | Score | Old reads, all forwards | Routed-only old reads |
| --- | ---: | ---: | ---: |
| Full Flash | 88.462 | 100.0000% | 100% dense policy |
| PrismKV | 88.462 | 21.0119% | 7.4923% (24 cases) |
| Quest-GQA, per-Q softmax | 81.795 | 20.5502% | 7.1208% (24 cases) |
| Quest-GQA, raw mean | 82.692 | 20.7246% | 7.1208% (24 cases) |

Whole-forward reads retain all 26 cases, including two cases with no
sparse route. Short answers and the common first dense decode forward make
these rates much higher than routed-only rates. The latter uses 24 cases.
Generated tokens stay dense and are excluded from these old-history ratios.

## Budget and page boundaries

Within each prompt, complete 16-token pages form the indexed prefix. The
remaining 0–15 prompt positions stay dense for both sparse methods, and count
as old reads. Native PrismKV is first replayed on the same 16 Full-generated
input tokens. Both Quest arms use its mean indexed read count rounded down
to a page multiple. This is offline calibration. Free-generation reads and
per-step Prism counts can differ. The pilot does not enforce exact reads on
every method trajectory. The page-rounding gap is below 16 tokens.

## Eager latency on common inputs

The model timing replays the same 16 Full input tokens, with a common first
dense forward and three rotated repeats. Module timing uses one identical
Full Q/K/V snapshot at each case, including routing and indexed Attention,
with four rotated CUDA-event repeats. Tables show medians of per-case medians.
These eager measurements are separate from the paper's Graph results.

| Length | Method | Complete Attention, ms | Eager model step, ms |
| --- | --- | ---: | ---: |
| 8K | Full Flash | 1.623 | 19.020 |
| 8K | PrismKV | 3.008 | 20.371 |
| 8K | Quest-GQA, per-Q softmax | 4.367 | 22.295 |
| 8K | Quest-GQA, raw mean | 3.832 | 21.661 |
| 32K | Full Flash | 5.162 | 20.356 |
| 32K | PrismKV | 3.658 | 20.033 |
| 32K | Quest-GQA, per-Q softmax | 4.352 | 22.205 |
| 32K | Quest-GQA, raw mean | 4.143 | 21.480 |

At 8K, all sparse eager paths are slower than Full. At 32K the within-case
median module speedups are 1.400 for PrismKV, 1.186 for normalized Quest-GQA
and 1.243 for raw-mean Quest-GQA. Corresponding eager model-step speedups are
1.014, 0.909 and 0.940. These ratios are medians of paired ratios. They can
differ from ratios of table medians.

## Verification and limits

All 26 case-result hashes and source pins passed. The audit independently
replayed 104 official grades, checked 14,976 Quest page/budget selections
and verified 22,464 sparse group outputs against the selected-token FP32
Attention reference. Maximum relative L2 error was below 0.002659.
The raw and normalized Quest page sets have mean Jaccard 0.767 on the common
Full snapshots. This establishes a real selection difference between the
two aggregation choices, without assigning the historical benchmark gap to it.

This subset has one sample per task-length cell, fixed 16-step timing,
single-request PPU execution and no confidence interval or full benchmark
rerun. Official Quest's native kernel performance remains unmeasured.
The [upstream code](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/evaluation/quest_attention.py)
provides the source method. This experiment measures the stated adaptation.

- [Exact timed router source](timed_router.py). Only its Quest mode is used.
- [Per-case results and summaries](summary.json).
- [Independent audit](audit.json).
