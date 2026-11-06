# Algorithm and variants

1. Collect post-position-encoding Q/K from a frozen backbone, training split only.
2. Train one shared encoder per attention layer/KV group. Keys are centered using
   their context mean and divided by saved training scale; queries are scaled
   without centering. K activations use a global positive budget, averaging at
   most eight memberships per token. Training Q codes use per-Q Top16.
3. After full prefill, encode keys and build feature -> original token-ID lists.
   A token may belong to multiple lists; original KV are stored once.
4. Cache count n, original-K mean mu and population diagonal variance nu per list.
5. At each decode step and for each Q head, score
   `log(n) + q@mu/sqrt(d) + (q*q)@nu/(2*d)`; mask empty lists.
6. Softmax each Q's bucket scores separately; average across Q heads sharing KV.
   Select Top16 buckets, OR memberships and deduplicate token IDs.
7. Compute attention separately for each Q over these original KV rows plus all
   tokens in the generated unindexed tail.
8. Every 4096 generated tokens, encode only the new block with the original
   center and merge its counts/moments/lists. Preserve older memberships.

The diagonal-variance score is an approximation of a bucket log exponential sum;
it is not an exact bound or a guarantee of selecting the true highest-mass lists.
Overlap also means normalized bucket scores are not disjoint token probabilities.

The optional `q_activation` route encodes current Q and aggregates its positive
TopK activations. It is a supplementary variant. Optional unique caps skip whole
buckets that overflow; these caps do not silently truncate an oracle-chosen list.
