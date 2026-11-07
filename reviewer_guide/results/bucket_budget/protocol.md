# Native Qwen3-8B bucket-count sensitivity

Use the same 26 input cases as the paired Quest pilot, with the same
dictionary, native Attention, prefill, dense prompt remainder and generation
caps. This is a shared development subset. It does not introduce new independent
test data or tune against the labels.

Compare Full, Top4, Top8 and Top16 on all 288 standard-attention KV groups.
The native fused router always computes Top16. Top4/Top8 zero validity flags
after the requested rank before the existing bitmap union. This preserves the
selected first B buckets and adds a device mask operation. Keep that operation
and the Top16 work in the measured cost. Do not claim optimal specialized
Top4/Top8 kernels.

Every arm shares one causal prefill. The first decode forward remains dense,
and all generated tokens stay dense. Report actual unique old-history reads
including that warmup, as well as routed-only rates. The index is static on
these tasks because the output caps are below the 4096-token update interval.

Reuse the 16-token Full replay and native module Q/K/V snapshots. Record eager
model time, eager module time and a separate CUDA Graph module measurement.
Graph setup is outside repeated timing and reported separately. Three model
repeats and four eager/Graph module repeats are paired by case.

Before accepting a case, verify the ranked buckets and their literal token
union for all 288 groups at all three B values. Compare AV to the selected-token
FP32 reference. Top16 must match the unchanged native route. Graph/eager outputs
and selected IDs must agree. Run one complete real-case smoke, then dispatch
the frozen 26 cases on verified idle resources.
