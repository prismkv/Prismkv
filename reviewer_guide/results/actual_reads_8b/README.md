# Qwen3-8B actual historical-read audit

Raw closure SHA and model/method/case/H/cap/budget/data/Full/Sparse identities checked for 31,250 baseline records, plus 7,000 reference identities. This is a counter audit, not official-method certification, metric replay, or a latency result.

|Suite|Method|Aligned cases|Actual old unique reads|Prism paired reads|First-prefill mismatches|
|---|---|---:|---:|---:|---:|
|longbench|PrismKV|3665|7.2198%|7.2198%|0|
|longbench|h2o|3665|6.5940%|7.2198%|0|
|longbench|quest|3665|7.0779%|7.2198%|0|
|longbench|snapkv|3665|7.2195%|7.2198%|0|
|longbench|pyramidkv|3665|7.2195%|7.2198%|0|
|ruler|PrismKV|3241|7.0392%|7.0392%|0|
|ruler|h2o|3241|6.8950%|7.0392%|50|
|ruler|quest|3241|6.9923%|7.0392%|50|
|ruler|snapkv|3241|7.0391%|7.0392%|50|
|ruler|pyramidkv|3241|7.0391%|7.0392%|50|
|ruler|kvzip|3241|7.0391%|7.0392%|50|

## Denominators and limitations

Read fractions are unique old-history positions divided by that case's old-history length, averaged over each method's own generation/group events. Cells are task (LongBench) or task x length (RULER), then macro-averaged. They are not identical per-step/group budgets or physical bytes read. Generated-tail access is separate.

LongBench: 3,750 quality cases; Prism has 3,665 nonzero-counter cases and 85 without sparse events. RULER: 3,250 quality cases; Prism has 3,241 nonzero-counter cases and 9 without sparse events. Zero-event cases remain in quality; they are not assigned a zero read. Baselines used full-history targets when Prism had no sparse measurement, so whole-cohort baseline read averages differ from the aligned subset. Both are retained in summary.json.

**New audit finding:** all five 8B RULER baselines disagree with the frozen Full reference on the first generated token for the same 50 cases (VT32, QA1 eight, QA2 eight, FWE one, multiquery one). They agree with each other on the new first token; exact input-token hashes match the reference on all50. The discrepancy precedes sparse selection. Its arithmetic/runtime cause and quality effect remain unresolved. Do not cite this audit as certification of strictly paired RULER quality, remove these cases, or attribute the discrepancy to retrieval. LongBench's four audited baselines have no recorded first-token mismatches.

Includes H2O, Quest, SnapKV, PyramidKV for LongBench; additionally KVzip for RULER. 8B KVzip LongBench is not covered by this audit. Sources are the frozen scale-out migration manifest and completed shards; different duplicate closures are rejected. All rows retain their input/reference SHA identity. No original result or score was modified.

8B uses 36 layers, 32Q/8KV, d128; extended64K/128K RULER uses the historical YaRN4 configuration, not native128K. GQA/budget adaptations are not unmodified official implementations.
