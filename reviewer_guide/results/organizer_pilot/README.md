# Supplementary RULER organizer pilot

The pilot covers 26 RULER inputs, with one example per task at each of 16K and
64K. The sparse methods route all 64 KV groups in the standard attention layers
of the 27B backbone.

The indexed Full baseline and the sparse methods use the same attention operator.
Each input also has a Full FlashAttention reference. We verified input identities,
all-read and repeat-Full checks, and the matched token counts before exporting.

[task_results.csv](task_results.csv) reports RULER task scores and reads. The `tables.ruler` section of [summary.json](summary.json) contains
macro scores and output lengths. Full uses the complete history, as marked by
`all_history_full_policy` in the CSV. Every method can access the generated tail.

## Results and interpretation

| Method | RULER macro score |
|---|---:|
| Indexed Full | 96.154 |
| Full FlashAttention | 96.154 |
| SAE, matched reads | 96.154 |
| Multi-assignment K-means, matched reads | 91.346 |
| Random projection, matched reads | 86.538 |

On these RULER inputs, K-means retains more attention mass on the Full
trajectory than SAE and scores lower in free generation.

## Matching and limits

Each matched arm receives the same per-step/layer/KVgroup unique-count schedule
measured from native SAE Top16 on the indexed-Full trajectory. Boundary priority
does not use true attention. If an arm outlasts Full, the final schedule repeats
and is logged. Reads match exactly on the identical Full-query probes; averages
over free generations can differ because trajectories and lengths differ.
This is an offline diagnostic, not a deployable future-budget oracle.

SAE uses 43 inherited and 21 fresh dictionaries; the frozen 3,000-center K-means
organizer uses 28 training contexts. Training history is not matched. Each K is
assigned to its nearest eight centers. Random projection uses up to eight positive scores,
the same moment router and the same reader; it is not a named official LSH
baseline. The pilot is too small to claim significance or full-benchmark ranking.
The original incomplete cross-kernel experiment remains excluded; every input
was rerun in this common-operator protocol, without dropping its failed input.
