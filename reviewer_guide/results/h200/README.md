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

| Prompt or window | Max active batch | Decode speedup | Decode + index speedup | Full → PrismKV peak GiB |
|---|---:|---:|---:|---:|
| 32K | 2 | 1.007× | 1.000× | 58.07 → 60.64 |
| 32K | 4 | 1.015× | 1.006× | 62.43 → 67.70 |
| 32K | 8 | 1.077× | 1.055× | 72.10 → 81.90 |
| 32K | 16 | 1.097× | 1.066× | 92.16 → 110.44 |
| 64K | 2 | 1.052× | 1.040× | 61.68 → 66.60 |
| 64K | 4 | 1.071× | 1.055× | 70.96 → 79.04 |
| 64K | 8 | 1.176× | 1.138× | 88.25 → 104.26 |
| 128K | 2 | 1.122× | 1.099× | 69.82 → 78.56 |
| 128K | 4 | 1.183× | 1.151× | 87.09 → 101.90 |
| 256K window | 2 | 1.266× | 1.220× | 85.96 → 101.59 |

The 32K, batch-16 index uses 17.60 GiB, or 1.10 GiB per reserved slot.
The original KV remains resident. At 128K, batch 4, decode throughput is
103.11 → 121.94 token/s and total service throughput is 54.48 → 58.53
token/s. [Graph serving CSV](graph_serving.csv) and [detailed JSON](graph_serving.json)
include timings, throughput, p95 latency, graph capture and memory allocations.

At a 256K window and maximum batch 2, total service throughput is 1.070×
Full in the fixed-cap trace. Across the continuous-batching traces, about
97%–99.7% of outputs follow first EOS because generation continues to the
configured cap. These results measure sustained decoding and index
amortization. Natural stopping and a production scheduler could give
different service-level results.

## Routing, attention and irregular reads

On the 128K single-request trace, the CUDA Graph attention path including
routing takes 2.514 ms for Full, 1.255 ms with the Tensor Core sparse kernel
and the previous router, and 1.178 ms with the combined optimizations. The
same combined path takes 1.733 ms in eager execution versus 1.652 ms with
only the Tensor Core change.

The near-256K batch-2 decode snapshot uses 262,080 input tokens per request.
Timings sum the 16 standard attention layers.

| Stage | ms per batch step |
|---|---:|
| Bucket scoring and Top16 selection | 0.718 |
| Candidate merge and deduplication | 0.186 |
| Complete router | 0.908 |
| Indexed attention | 1.105 |
| Complete PrismKV attention path | 2.022 |
| Full attention path | 8.989 |

Complete paths are timed independently. In a separate 252K, batch-1
layout control, layer 31 uses the same candidate IDs in all cases.

| KV access path | µs |
|---|---:|
| Direct indexed attention | 44.2 |
| Attention on prepacked contiguous KV | 43.7 |
| Online explicit gather only | 61.3 |
| Online explicit gather followed by attention | 106.0 |

[Stage and layout records](stage_profiles.json) include the 32K and 128K
layout controls and the isolated timing definitions.

## Offline costs on PPU-ZW810

The offline audit covers the 64 evaluated dictionaries. It counts shared
captures once and includes 332 distinct Q/K capture jobs. Capture worker
windows include backbone prefill and decode, as well as loading and
serialization within the recorded windows.

| Timed phase | Summed PPU-ZW810 device-hours |
|---|---:|
| Q/K capture worker windows | 8.853 |
| Training sample mining | 0.455 |
| SAE optimization across the checkpoint lineage | 1.664 |
| **Total accounted offline cost** | **10.972** |

See the [worker-window audit](../../training.md#27b-offline-cost-including-capture-worker-windows)
for the accounting boundary and raw records. The [earlier phase-only audit](../../training.md#27b-recorded-offline-phases)
uses a narrower capture scope and excludes untimed loading and serialization.
[Runtime fingerprints](runtime.json) describe the H200 experiments.
