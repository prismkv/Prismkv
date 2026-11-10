# H200 system evaluation and offline cost accounting

The online experiments use one NVIDIA H200 and the BF16 Qwen3.8-27B hybrid
checkpoint. Six query heads share each KV group, and the sparse path applies
to the 16 standard attention layers. The Full baseline uses grouped
FlashAttention without KV expansion. The combined PrismKV path uses the
[SM90 implementation](../../../kernels/cuda_sm90/README.md).

## Single request

Each arm uses the same prompt and generates 4,096 tokens with CUDA Graph.
Prefill emits the first token and the table uses the remaining 4,095 decode
steps. Attention includes routing and sums 16 layers at five positions of a
shared Full trace. Capture and model load are recorded separately. Times in
the two latency columns are ms per decode step.

| Prompt | Full → PrismKV attention | Attention speedup | Full → PrismKV decode | Decode speedup | Index build | Decode + index speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 0.893 → 0.948 | 0.94× | 21.027 → 21.070 | 0.998× | 0.293 s | 0.995× |
| 64K | 1.429 → 1.023 | 1.40× | 21.619 → 21.152 | 1.022× | 0.445 s | 1.017× |
| 128K | 2.514 → 1.178 | 2.13× | 22.516 → 21.331 | 1.056× | 0.724 s | 1.047× |
| 252K | 4.661 → 1.464 | 3.18× | 24.441 → 21.528 | 1.135× | 1.277 s | 1.119× |

[Single-request measurements](single_request.csv) include prefill, graph
capture and full request time. The 252K prompt with its 4K output reaches the
256K window.

## Continuous batching

The two arms share requests, arrivals, output caps and a serial-prefill FIFO
scheduler. CUDA Graph replay is used for decode. The table reports a single
paired long-output stress trace per configuration. The 256K window uses a
252K prompt and up to 4K output per request. Peak memory is the highest NVML
sample, in GiB. Total output throughput includes queueing, prefill, index
construction and decode.

| Prompt or window | Max active batch | Decode speedup | Decode + index speedup | Total service speedup | Full → PrismKV peak GiB |
|---|---:|---:|---:|---:|---:|
| 32K | 2 | 1.007× | 1.000× | 1.000× | 58.07 → 60.64 |
| 32K | 4 | 1.015× | 1.006× | 1.005× | 62.43 → 67.70 |
| 32K | 8 | 1.077× | 1.055× | 1.039× | 72.10 → 81.90 |
| 32K | 16 | 1.097× | 1.066× | 1.043× | 92.16 → 110.44 |
| 64K | 2 | 1.052× | 1.040× | 1.030× | 61.68 → 66.60 |
| 64K | 4 | 1.071× | 1.055× | 1.039× | 70.96 → 79.04 |
| 64K | 8 | 1.176× | 1.138× | 1.074× | 88.25 → 104.26 |
| 128K | 2 | 1.122× | 1.099× | 1.055× | 69.82 → 78.56 |
| 128K | 4 | 1.183× | 1.151× | 1.074× | 87.09 → 101.90 |
| 256K window | 2 | 1.266× | 1.220× | 1.070× | 85.96 → 101.59 |

The 32K, batch-16 index uses 17.60 GiB, or 1.10 GiB per reserved slot.
The original KV remains resident. At 128K, batch 4, decode throughput is
103.11 → 121.94 token/s and total service throughput is 54.48 → 58.53
token/s. [Graph serving CSV](graph_serving.csv) and [detailed JSON](graph_serving.json)
include timings, throughput, p95 latency, graph capture and memory allocations.

The 256K short-output Graph control reaches 1.270× decode throughput and
0.985× total service throughput. The [eager service results](eager_serving.json)
also retain short-output regressions at 8K–256K. In the long-output Graph
traces, about 97%–99.7% of outputs follow first EOS because generation
continues to a fixed cap. These results measure sustained decoding and
index amortization. Natural stopping and a production scheduler could
give different service-level results.

## Routing, attention and irregular reads

On the 128K single-request trace, the CUDA Graph attention path including
routing takes 2.514 ms for Full, 1.255 ms with the Tensor Core sparse kernel
and the previous router, and 1.178 ms with the combined optimizations. The
same combined path takes 1.733 ms in eager execution versus 1.652 ms with
only the Tensor Core change.

For 256K, batch 2, each batch step takes 0.718 ms for scores and Top16,
0.186 ms for candidate merge and deduplication, 0.908 ms for the full router
and 1.105 ms for indexed attention. The combined path was timed separately
at 2.022 ms, versus 8.989 ms for Full. In the fixed-candidate layout control
at 252K on layer 31, indexed attention takes 44.2 µs and prepacked attention
takes 43.7 µs. Explicitly gathering into a contiguous buffer and then running
attention takes 106.0 µs. [Stage and layout records](stage_profiles.json)
include the 32K and 128K layout controls and isolated timing definitions.

## Offline costs on PPU-ZW810

The offline phase audit for the evaluated 64 dictionaries covers 324 capture
jobs and 226 optimization-stage records. Shared captures are counted once.

| Timed phase | Summed accelerator-hours |
|---|---:|
| Frozen backbone prefill for Q/K capture | 3.459 |
| Frozen backbone decode for Q/K capture | 3.103 |
| Training sample mining | 0.455 |
| SAE optimization across the checkpoint lineage | 1.664 |
| **Accounted subtotal** | **8.681** |

The subtotal excludes untimed model loading, serialization and preparation,
including the separate early Q/K sources for six dictionaries. These
accelerator-hours are summed across jobs on PPU-ZW810. The
[training phase audit](../../training.md#27b-recorded-offline-phases) and its
[machine-readable ledger](../../training_cost_27b.json) document the accounting
boundary. [Runtime fingerprints](runtime.json) describe the H200 experiments.
