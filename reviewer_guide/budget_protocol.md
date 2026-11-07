# What a token-read budget means

Always separate:

1. Raw selected-list entries, including cross-list duplicate token IDs.
2. Unique selected old-history token IDs, after union.
3. Full generated-tail length.
4. All non-sparsified layers/groups, if any.

The paper's bucket-count policy gives a variable unique fraction. The inspected
LongBench, RULER, LongProc and SCBench protocols use scalar per-case targets
derived from PrismKV's mean old-token reads. Actual reads can vary by step,
layer and KV group. The tables report the recorded counters and their
denominators, including the treatment of zero-event cases and dense warmup.
[Audited quality and read tables](results/README.md)

For a controlled new comparison, define a reference prefix and old-history H,
freeze the target unique count at each step/group, and apply the same ceiling
and tail/retention policy to both routers. A boundary truncation is a diagnostic
policy change and must be labeled. Page-based methods may have granularity error.

The portable `unique_cap` supports a strict ceiling, not exact count matching:
whole buckets that would overflow are skipped, so actual reads can be below cap.
It never consults true attention to choose boundary tokens.

Cache eviction and re-querying retained full history are different protocols.
The LongProc/SCBench advantage cannot be attributed solely to retrieval unless
retention, tail accessibility and effective access budgets are controlled.
