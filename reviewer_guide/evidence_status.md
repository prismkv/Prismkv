# Evidence status

The submitted default remains F3000, mean K activation budget8, bucket Top16,
with count/mean/diagonal-variance routing. Supplementary Q-routing variants
are explicitly separate. No compact candidate was promoted.

## Executed and audited

- Ordinary reference:24CPU tests, tiny-backbone dense/sparse/cache/generation,
  saved-backend configuration, explicit eager causality, reference NLL and access conservation.
- [Public evaluation pipeline](evaluation.md): all 14,000 archived input-format
  checks,29 official-metric fixtures, and one actual preselected8K RULER case per
  model through Full/all-read/sparse generation and scoring. Not full quality reruns.
- All 288 8B and 64 hybrid 27B encoders packaged, with source/tensor/scales checks.
- Short pretrained 8B and hybrid 27B public-CLI execution witnesses; the latter
  uses all 64 standard KV groups and leaves48 linear layers untouched. See
  [hybrid scope/runtime limits](hybrid_reference.md).
- Standalone PPU 8B compile/correctness, six synthetic cases through 64K.
- Verified 27B RULER 65 task-length cells,880-case LongProc subset and 8B actual-read
  audits with explicit denominators/prefill discrepancies in [results](results/README.md).
- 58-case stronger organizers and seven score-term variants,29-case Squeezed
  reference port,16-case fixed-index dynamic/frozen LongProc control. These are
  bounded pilots with mixed outcomes, not universal SAE advantages.
- [Full 27B LongBench ablations](results/longbench27b_ablations/README.md) report
  all 3,750 examples per arm for width, fixed Key memberships, bucket cutoff
  and single-term scoring, with realized reads and explicit budget controls.
- Original 8B native32K/64K phase graphs and Torch allocator/index inventory;
  isolated phases are not additive, allocator memory is not physical-device peak.

## Incomplete

- Broader pretrained-scale quality/equivalence; native full-benchmark integration,
  SCBench reuse orchestration and hybrid clean dependency lock/upstream framework commit mapping.
- Complete checkpoint/training ancestry, near-duplicates and capture wall time.
- Independent all-group quality/read/latency Pareto, same-selector Quest curves,
  update-interval sensitivity, additional requested faithful baselines.
- Full physical-device/serving memory and othermodels/native paths.

New-card timing is deferred. Public unauthenticated repository access has not
been verified; a successful source push is not proof that a reviewer can open it.
No manuscript revision or OpenReview post is represented by these artifacts.
