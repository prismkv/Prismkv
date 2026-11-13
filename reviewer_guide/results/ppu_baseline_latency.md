# 27B additional LongBench baselines and PPU latency

Quality is the macro average of 16 LongBench tasks over 3,750 examples.
The latency measurements use the same 131,072-token prompt, Qwen3.8-27B BF16,
batch size 1, and PPU-ZW810. We report the median of three 128-step runs.

| Method | LongBench average | 128K whole-model decode (ms/token) |
|---|---:|---:|
| Full Attention | 54.031 | 50.38 |
| PrismKV | 53.963 | 47.85 |
| Twilight | 54.054 | 74.25 |
| RetrievalAttention | 54.004 | 1,404.02 |
| Squeezed Attention | 53.901 | 91.81 |
| BLASST | 53.845 | 76.73 |
| SeerAttention-R | 53.239 | 61.25 |

Full and PrismKV were timed together and PrismKV saves 2.53 ms/token
(1.053× speedup). The five other rows come from separate baseline runs.
[Per-repeat measurements and execution settings](ppu_128k_graph_repeat.json)

The timed Twilight adaptation includes Quest page selection, packed-INT4
candidate scoring and binary-search Top-p. The RetrievalAttention adaptation
performs graph search and attention over retrieved keys on the CPU, then
merges those results with device attention. These timings characterize the
tested PPU adaptations on this prompt.

## Additional eager timing details

Old-history reads count distinct positions per KV group on five shared
Full-trajectory probes. Attention covers all 16 standard attention layers and includes
routing, plus CPU work and transfers where applicable. Index build is timed
after prefill.

| Baseline | Baseline old reads | Baseline attention (ms/token) | Baseline index build (s) |
| --- | ---: | ---: | ---: |
| Squeezed Attention | 9.22% | 37.45 | 268.56 |
| BLASST | 9.22% | 32.30 | 0.15 |
| SeerAttention-R | 9.22% | 13.46 | 0.20 |
