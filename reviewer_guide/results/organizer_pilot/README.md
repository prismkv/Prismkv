# Supplementary organizer pilot (not a full benchmark)

This fresh pilot compares 32 LongBench inputs (two per task) and 26 RULER inputs
(one per task at16K/64K), with all64standard-attention KVgroups sparse.
The backbone and its48linear-attention layers are unchanged.

Main Full and sparse arms share the same indexed attention operator. An extra
Flash Full reference is retained on every input. All58case closures, common-
operator all-read and repeat-Full checks, input identities and matched selected
counts were checked before exporting. This does not reproduce official-method
GPU kernels or provide latency results.

`task_results.csv` contains every per-task score/read result (two cases each).
`summary.json` contains macro scores, read means, output caps and retrieval probes.
Full reads are blank in the counter column because they were not counted there;
`all_history_full_policy` explicitly identifies their complete-history policy.
The generated tail is eligible in every arm. Nine of32LongBench generations
reach their unchanged task cap in each arm.

## Results and interpretation

Matched SAE / multi-assignment K-means / random projection score
49.735 /50.736 /51.065 on LongBench, and96.154 /91.346 /86.538 on RULER.
Indexed Full scores51.345 /96.154; Flash Full51.542 /96.154.
These mixed results do not establish universal SAE superiority. K-means retains
more teacher attention even on RULER while its free-generation score is lower;
attention mass alone does not certify task correctness.

## Matching and limits

Each matched arm receives the same per-step/layer/KVgroup unique-count schedule
measured from native SAE Top16 on the indexed-Full trajectory. Boundary priority
does not use true attention. If an arm outlasts Full, the final schedule repeats
and is logged. Reads match exactly on the identical Full-query probes; averages
over free generations can differ because trajectories and lengths differ.
This is an offline diagnostic, not a deployable future-budget oracle.

SAE uses43inherited and21fresh dictionaries; the frozen3000-center K-means
organizer uses28training contexts. Training history is not matched. Each K is
assigned to its nearest8centers. Random projection uses up to8positive scores,
the same moment router and the same reader; it is not a named official LSH
baseline. The pilot is too small to claim significance or full-benchmark ranking.
The original incomplete cross-kernel experiment remains excluded; every input
was rerun in this common-operator protocol, without dropping its failed input.
