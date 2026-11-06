# Fixed-bucket score-term pilot

All seven nonempty subsets of count, mean and variance, on the same58inputs as
the organizer pilot.32LongBench/26RULER, all64standard-attention KVgroups; no SAE
or backbone training. Count=log(n), mean=q dot mu/16, variance=q² dot var/512.
Per-Q softmax and six-Q averaging unchanged; empty buckets excluded.

Every ablation gets the same Full-derived per-step/layer/group unique-count
schedule and attention-independent boundary priority, using identical original
SAE lists and indexed AV. Separate native Top16 and Flash Full references remain.
Last budget repeats if a generation outlasts Full. This is an offline diagnostic,
not an online future-budget rule. All58closures, counts and own parity checks
were verified. All290task-method rows, two examples each, are included.

Count-only scores34.815/13.718 on LongBench/RULER; mean-only51.699/96.154;
variance-only40.179/66.923; full three-term49.735/96.154. The evidence does not
establish that all three terms are necessary, that mean-only significantly beats
Full, or that it should universally replace the submitted score. Query-independent
count is insufficient here. Small historical adaptive cohort, not a full benchmark.

`task_results.csv` reports quality, actual old unique reads and cap termination.
Absent Full counters are blank, not zero; both Full references read all history.
`summary.json` also contains same-Full-query recall/mass observation means.
Probe counts match exactly; free-generation averages can differ with output paths
and lengths. No timings here should be used as latency claims.
