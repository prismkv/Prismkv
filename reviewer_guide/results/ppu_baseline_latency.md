# 27B additional LongBench baselines and PPU latency

Quality is the macro average of 16 LongBench tasks over all 3,750 examples.
Latency is from a separate 128K, batch-1, eager test with Qwen3.8-27B BF16
on PPU-ZW810. Each latency is the median of three 128-step runs. The table
uses the Full and PrismKV controls from the BLASST run as shared references.

| Method | LongBench macro | Whole-model decode (ms/token) |
|---|---:|---:|
| Full Attention | 54.031 | 53.71 |
| PrismKV | 53.963 | 52.11 |
| Twilight | 54.054 | 74.25 |
| RetrievalAttention | 54.004 | 1,404.02 |
| Squeezed Attention | 53.901 | 91.81 |
| BLASST | 53.845 | 76.73 |
| SeerAttention-R | 53.239 | 61.25 |

## Same-run controls

Each row below gives the controls measured alongside that baseline. Old-history
reads count distinct positions per KV group on five shared Full-trajectory
probes. Attention covers all 16 standard attention layers and includes
routing, plus CPU work and transfers where applicable. Index build is timed
after prefill.

| Baseline | Paired Full decode (ms/token) | Paired PrismKV decode (ms/token) | Baseline old reads | Baseline attention (ms/token) | Baseline index build (s) |
|---|---:|---:|---:|---:|---:|
| Twilight | 51.71 | 52.56 | 9.22% | 29.65 | 0.25 |
| RetrievalAttention | 52.14 | 51.62 | 8.71% | 1,377.68 | 1,192.42 |
| Squeezed Attention | 51.43 | 51.66 | 9.22% | 37.45 | 268.56 |
| BLASST | 53.71 | 52.11 | 9.22% | 32.30 | 0.15 |
| SeerAttention-R | 51.13 | 51.05 | 9.22% | 13.46 | 0.20 |

The timed Twilight adaptation includes Quest page selection, packed-INT4
candidate scoring and binary-search Top-p. The RetrievalAttention adaptation
performs graph search and attention over retrieved keys on the CPU, then
merges those results with device attention. These timings characterize the
tested PPU adaptations on this prompt.
