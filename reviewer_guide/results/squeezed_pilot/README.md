# Squeezed Attention reference-port pilot

29cases:16LongBench(one per task),13RULER(one per task at16K). All64standard KV
groups; unchanged backbone/linear layers. Primary Full, SAE-shared selections and
Squeezed selections share per-Q Flash AV. Separate GQA Full/native SAE references
are included. All29closures/source/own all-read and repeat-Full checks passed.

Source-style per-Q threshold scores58.686 vs SAE56.528 on LongBench, and100vs100
on RULER. Group-unique old reads are23.059%vs8.820%, and19.080%vs6.727%.
Thus source-style quality is not an equal-budget comparison. Per-Q visit and GQA
unique-union columns are distinct; this accuracy port does not measure bandwidth.
Three LongBench cases have no sparse-step read counter but remain in quality.

The added shared-budget GQA adaptation protects the original100old observation
tokens. If this exceeds a target group budget, the case is infeasible/unscored.
There are6such LongBench cases and8RULER cases. Do not assign zero or compare
different denominators. On the SAME10feasible LongBench cases, Full/SAE/adaptation
score52.071/52.250/51.964; all three score100on the same5feasible RULER cases.
All174case-method rows, including infeasibility reasons, are included.

Pinned selector source4a50c64eca2f20232a4e998d754d030ce3a7f6e6. Preserve normalized-K
5%clustering,k-means++seed0,max300Lloyd,raw means,100observed Qpositions,
90th-percentile global PER-TOKEN threshold,count-weighted normalization and full
generated tail. Fixed-label source-calibration parity maxerror3.73e-9.

Port differences: Torch Lloyd vs cuML, FP32 scoring vs BF16 Triton, whole formatted
prefix vs original document/query split,d256GQA vs old MHA kernel. The shared-budget
row is our additional adaptation, not the official threshold algorithm. This is
not an official GPU-kernel/latency reproduction or full-benchmark ranking.
Four of16LongBench outputs hit the unchanged cap in each full-cohort arm.
