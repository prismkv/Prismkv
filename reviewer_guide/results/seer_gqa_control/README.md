# Paired SeerAttention-R quality control

We compare four readers on Qwen3-8B with one shared prefill per prompt, greedy decoding, common stopping rules and one dense decode forward. All 36 layers and 288 KV groups participate. The 26-case pilot uses one historical RULER example per task at 8K and 32K. The 195-case follow-up uses the earlier frozen budget cohort, with five examples per task at 8K, 16K and 32K. The follow-up inputs are disjoint from the pilot and the worker is byte-identical across the runs.

The two Seer rows use the [official gate source](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e) and [released Qwen3-8B weights](https://huggingface.co/SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates/tree/b31445f675867a4730b2e53411f3796ea7f9337f). The learned projection combines the four Q heads in each KV group. Both ports use a corrected compressed-cache write cursor and the same original-KV attention backend as PrismKV. The source-style policy ranks all available 64-token blocks and forces the last block. The full-tail adaptation ranks only complete historical blocks and reads the complete tail, including the prompt remainder. Both policies receive the mean PrismKV historical index budget from a 16-step Full-trajectory replay, rounded down to blocks per prompt. Realized reads vary with rounding, last-block protection and generated trajectories.

## Pilot

| Cohort | Arm | Score | Routed old reads | All-forward old reads | Routed generated reads |
|---|---|---:|---:|---:|---:|
| Overall (26) | Full | 88.462 | — | 100.000% | — |
| Overall (26) | PrismKV Top16 | 88.462 | 8.150% (24) | 21.557% | 100.000% |
| Overall (26) | Seer source-style | 73.718 | 7.025% (24) | 20.577% | 98.996% |
| Overall (26) | Seer full tail | 74.103 | 7.720% (24) | 21.271% | 100.000% |
| 8K (13) | Full | 92.308 | — | 100.000% | — |
| 8K (13) | PrismKV Top16 | 92.308 | 7.361% (12) | 21.123% | 100.000% |
| 8K (13) | Seer source-style | 82.052 | 5.749% (12) | 19.822% | 98.526% |
| 8K (13) | Seer full tail | 82.821 | 6.867% (12) | 20.956% | 100.000% |
| 32K (13) | Full | 84.615 | — | 100.000% | — |
| 32K (13) | PrismKV Top16 | 84.615 | 8.940% (12) | 21.992% | 100.000% |
| 32K (13) | Seer source-style | 65.385 | 8.301% (12) | 21.331% | 99.467% |
| 32K (13) | Seer full tail | 65.385 | 8.573% (12) | 21.586% | 100.000% |

## Follow-up

| Cohort | Arm | Score | Routed old reads | All-forward old reads | Routed generated reads |
|---|---|---:|---:|---:|---:|
| Overall (195) | Full | 91.009 | — | 100.000% | — |
| Overall (195) | PrismKV Top16 | 89.880 | 7.677% (194) | 15.035% | 100.000% |
| Overall (195) | Seer source-style | 73.000 | 6.573% (194) | 14.393% | 99.019% |
| Overall (195) | Seer full tail | 72.641 | 7.144% (194) | 14.911% | 100.000% |
| 8K (65) | Full | 93.333 | — | 100.000% | — |
| 8K (65) | PrismKV Top16 | 92.564 | 7.452% (65) | 14.511% | 100.000% |
| 8K (65) | Seer source-style | 77.410 | 5.754% (65) | 13.298% | 98.166% |
| 8K (65) | Seer full tail | 75.872 | 6.739% (65) | 14.140% | 100.000% |
| 16K (65) | Full | 90.462 | — | 100.000% | — |
| 16K (65) | PrismKV Top16 | 91.333 | 6.556% (64) | 14.682% | 100.000% |
| 16K (65) | Seer source-style | 68.487 | 5.542% (64) | 13.749% | 99.318% |
| 16K (65) | Seer full tail | 70.949 | 6.045% (64) | 14.303% | 100.000% |
| 32K (65) | Full | 89.231 | — | 100.000% | — |
| 32K (65) | PrismKV Top16 | 85.744 | 9.007% (65) | 15.911% | 100.000% |
| 32K (65) | Seer source-style | 73.103 | 8.408% (65) | 16.130% | 99.579% |
| 32K (65) | Seer full tail | 71.103 | 8.631% (65) | 16.289% | 100.000% |

Read rates are equal-weight means of per-case logical KV-position fractions. Routed means use cases with routed forwards, with the case count in parentheses. All-forward means include the dense decode forward. Old reads count the fixed prompt and generated reads use the available generated tail. Score means use all assigned prompts. Detailed per-case counts and caps appear in the CSV. All sparse arms use a 64-token index boundary, so this PrismKV row has its own score and read measurements separate from the earlier 16-token-aligned budget study.

The score audit replays 884 official scores. The port verifies token IDs against the block masks and compares indexed AV against selected-token FP32 attention. The largest relative L2 difference across the four Seer cohorts is 0.002165. The study measures quality and logical reads on a historical RULER subset. The common AV backend and cache correction define the scope of the comparison.

- [Per-case scores and counts](case_scores_and_reads.csv)
- [Pilot audit and generated token IDs](pilot_summary.json)
- [Follow-up audit and generated token IDs](followup_summary.json)
- [Paired protocol](protocol.md)
