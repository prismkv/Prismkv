# Native8B phase latency and allocator-memory diagnostic

Two completed 32K/64K single-input runs. Each uses all 36 layers / 288 KV groups, F3000, mean8 key memberships, Top16 moments routing and a dense 8-token generated tail. Original BF16Q/K/V snapshots come from Full continuation. Same released pack/union/fused-normalization-Top16/tuned-AV source hashes and original compile manifests were checked.

Every layer passed exact feature-ID and deduplicated-union checks against the unfused native route. All 576 group attention-output checks had relativeL2 below 0.002182. Graph/eager output checks were below 1e-5. These are execution checks, not downstream quality scores.

## Latency: isolated stage graphs

Units:ms, summed over 36 layers. Each number is the median of six replays of a graph containing only that stage across layers. This removes Python launch gaps, but changes inter-stage cache/scheduling behavior. **Do not treat these values as additive shares of full-chain latency.** The CSV separately retains event-instrumented eager profiles; those include launch gaps.

|Stage|32K|64K|
|---|---:|---:|
|Q_pack|0.1082|0.0854|
|summary_GEMM|0.6378|0.6351|
|normalize_aggregate_Top16|1.1788|1.1667|
|bitmap_union_compaction|0.5843|0.6191|
|indexed_AV|1.0020|1.5009|

## Independently measured complete module graphs

|History|FullFlash|FullSDPA|Sparse incl.route|SparseAV only|Full-best / sparse|Unique old reads|
|---|---:|---:|---:|---:|---:|---:|
|32K|5.3491|5.5026|3.3770|1.0000|1.584x|7.073%|
|64K|10.0259|10.3112|3.9003|1.4954|2.571x|6.916%|

The whole-chain graph contains pack, scoring, normalize/Top16, union/compaction and original-KV attention. Full SDPA uses grouped dimensions without physical KV repetition; the independent Flash path also keeps native GQA. Preload, prefill, index construction, capture and first replay are excluded from steady-state module graph times. All 32K/64K runs use historical YaRN4 settings. This is a supplementary one-snapshot diagnostic, not the paper's 256-step natural-generation curve, end-to-end timing or a new-card/serving result.

## Torch allocator memory

GiB=2^30 bytes. This is PyTorch allocated memory, with separate peak/reserved counters in the CSV, **not complete physical device peak**. Full cache capacity is H+8. Sparse setup includes the same model/original KV, all encoders, original retained lists/moments, packed score matrices, membership bitmaps and workspaces.

|History|Full allocated|Sparse setup allocated|Extra overFull|Index-build peak allocated|Index-build peak reserved|
|---|---:|---:|---:|---:|---:|
|32K|19.768|25.943|6.175|26.244|26.342|
|64K|24.268|34.576|10.308|35.184|35.484|

The 64K Native-object inventory contains about 6.592 GiB of membership bitmaps plus 1.125 GiB of retained membership lists,0.824 GiB packed scoring matrices and0.827 GiB original-K moments/counts. The inventory deduplicates tensor storages reachable from Native objects; it does not equal every allocator/runtime allocation. The original timed implementation retains those lists/moments, while the standalone public binding can have different caller-owned lifetimes. Do not transfer this memory number unchanged to another binding or call it KV-memory compression.

The method saves KV access/computation here while adding substantial index memory. No NVIDIA, concurrency, p95, process-wide/device-wide memory or full training/capture cost claim follows. Index-build time in the JSON is a single setup observation, not an additional speed curve.

Files: latency.csv, torch_memory.csv, native_tensor_inventory.csv and summary.json. No prompts, model weights, host addresses, private paths or reviewer text are included.
