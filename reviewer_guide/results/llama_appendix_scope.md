# Llama-3.1-8B-Instruct appendix scope and archived reads

The submitted Table 6 RULER column averages 8K, 16K and 32K. The caption
omitted the length range. All seven displayed scores reproduce from those
1,950 saved grades per method. The five-length archive also includes 64K
and 128K, bringing the denominator to 3,250 per method.

| Method | LongBench score | Read % | RULER 8K–32K score | Read % | RULER 8K–128K score | Read % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 49.548 | 100.0000 | 85.556 | 100.0000 | 82.298 | 100.0000 |
| PrismKV | 48.881 | 7.7347 | 81.010 | 7.1202 | 58.415 | 7.5891 |
| H2O | 45.878 | 6.9580 | 20.305 | 6.9453 | 21.583 | 7.4756 |
| Quest | 48.818 | 7.5595 | 79.118 | 7.0473 | 76.363 | 7.5402 |
| SnapKV | 46.776 | 7.7346 | 67.474 | 7.1201 | 68.801 | 7.5890 |
| PyramidKV | 46.522 | 7.7346 | 65.433 | 7.1201 | 67.576 | 7.5890 |
| KVzip | 43.302 | 7.7346 | 33.235 | 7.1201 | 37.023 | 7.5890 |

Quality macro-averages the 16 LongBench tasks or the 13 RULER tasks at
each included length. Read rates average unique old-history reads per
case on the common seven-method subset with recorded reads and an
eligible sparse event. Subset sizes are 3,479/3,750 for LongBench,
1,918/1,950 for RULER 8K–32K and 3,194/3,250 for RULER 8K–128K. Full
reads are 100% by policy. Generated-tail reads are separate.

| Method | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 87.715 | 86.500 | 82.451 | 80.731 | 74.092 |
| PrismKV | 84.492 | 81.262 | 77.274 | 39.049 | 10.000 |
| H2O | 19.936 | 19.108 | 21.872 | 24.049 | 22.949 |
| Quest | 78.321 | 79.933 | 79.100 | 74.741 | 69.718 |
| SnapKV | 63.531 | 66.026 | 72.867 | 73.449 | 68.133 |
| PyramidKV | 62.510 | 64.175 | 69.613 | 73.359 | 68.223 |
| KVzip | 28.177 | 33.769 | 37.759 | 41.903 | 43.508 |

Each length includes 650 grades per method. PrismKV reaches the
generation cap on 62.31% of cases at 64K and 90.00% at 128K, compared
with 5.23% and 5.69% for Full. The 64K PrismKV results come from the
original path. The 128K results combine 328 original-path cases with 322
native-resume cases. Those subsets are disjoint and selected by prior
completion. A controlled same-input replay could test backend effects.

The audit verified 63,000 archived result hashes across nine methods.
For 2,150 original Full or PrismKV records, the frozen configuration hash
provided the completion evidence. The read table uses the frozen reference
mean on 5,075 PrismKV records that lack a top-level read mean. All 1,881
LongBench fallback values agree with the result's saved unique-read ratio.
All 3,194 RULER fallback values agree with the result's nested unique-read
mean divided by history length. LongBench TREC source-hash
metadata disagrees with the current configuration for 200 Full and 200
PrismKV records. Input identity for those historical records remains
unestablished. Scores use the saved grades throughout.

Two additional archived adaptations scored above PrismKV on this model.
DoubleSparse and SparQ scored 49.554 and 49.814 on LongBench, 84.239 and
85.243 on RULER 8K–32K, and 81.053 and 82.005 over all five lengths.
They use separate adapted implementations. The read table covers the
seven methods in Table 6.

[Per-task audit data](llama_appendix_scope.json)
