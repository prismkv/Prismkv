# Native path, memory and timing scope

## Layout and operations

The extracted 8B specialization uses original BF16 K/V shaped `[8,T,128]` and
Q `[8,4,128]`. It preallocates IDs `[8,H]`, valid counts `[8]`, bucket-score
buffers and split-attention workspaces. The capacity of an ID buffer is not the
number of rows consumed; attention reads only valid counts plus the dense tail.

`pack.cu` packs Q and Q-squared. A batched GEMM scores means and variances.
`route_parallel.cu` performs per-Q normalization, group aggregation and Top16.
`union.cu` ORs selected feature bitmaps, computes population counts/prefix
positions, and compacts unique sorted token IDs. Everything remains on device.
This metadata work is not constant in context length: bitmap words scale with
history length and compaction emits the selected IDs. Fixed-size moment scoring
avoids scanning every raw K, not all index-metadata processing.

`av_tuned.cu` partitions selected IDs and tail across blocks, shares each KV
load among the 4 Q heads and uses vectorized loads over the head dimension.
Sparse rows remain indirect; sorted IDs do not make them fully contiguous.
Partitions compute local online-softmax statistics; a second kernel merges
their maxima, exponential sums and weighted values exactly up to floating-point
rounding. No estimated softmax denominator is substituted.

The launches obey stream dependencies. Empty/short partitions are masked;
cross-block softmax merging is necessary. The work is not synchronization-free.

## Storage

Dense membership bitmaps use `F*ceil(H/32)*4` bytes per KV group. At F=3000,
H=65536 this is 24,576,000 bytes/group. BF16 original KV with d=128 uses
`H*d*4 = 33,554,432` bytes/group: bitmap alone is 73.24% of KV, not negligible.
For d=256 the same bitmap is 36.62% of KV. These are derived allocations,
not measured peak device memory. Moments, IDs, build temporaries, graphs and
allocator reservation must be reported separately. `Native8B.buffer_bytes()`
lists live tensors owned by this binding; it excludes caller/model storage.

## Timing

Report prefill, first index build, graph capture/startup, score GEMM,
normalization/selection, union/compaction, selected attention, full-model decode,
and periodic update separately. Native kernel speedups are not automatically
whole-model or serving speedups. The PPU result does not establish NVIDIA
throughput, concurrent-request behavior or p95 latency.
