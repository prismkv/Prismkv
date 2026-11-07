# Additional archived historical reads

## Qwen3.8-27B SCBench

The read audit verifies 12,908 case result receipts, their pinned input rows and
the old-token counters. The archive contains 1,844 cases and 10,286 turns per
method, with 922 cases each for multi-request and self-generated multi-turn.
Quality values come from the archived score table. Read percentages average the
per-case fraction over cases in the relevant mode. Full reports the dense policy.

| Method | Archived quality macro | Multi-request reads | Multi-turn reads | All cases |
|---|---:|---:|---:|---:|
| Full | 59.796 | 100.0000% | 100.0000% | 100.0000% |
| PrismKV | 59.082 | 7.9394% | 7.9292% | 7.9343% |
| H2O | 31.149 | 7.8282% | 7.4750% | 7.6516% |
| Quest | 54.265 | 7.9300% | 7.9196% | 7.9248% |
| SnapKV | 32.894 | 7.9394% | 7.9292% | 7.9343% |
| PyramidKV | 27.218 | 7.9394% | 7.9292% | 7.9343% |
| KVzip | 27.238 | 7.9394% | 7.9292% | 7.9343% |

Each baseline uses the rounded and clamped mean from its paired PrismKV case.
The initial prompt tokens define old history. Multi-request restores the base
branch for each request. In multi-turn, each method extends its own generated
dialogue. H2O bounds its total retained attention set. The other sparse adapters
keep the generated tail eligible. H2O's largest absolute difference between a
case's actual mean and its target is 1,960.054 old tokens. Quest's corresponding
difference is 20.136. SnapKV, PyramidKV and KVzip hit the scalar targets in this
audit. Per-step/layer/group equivalence remains unmeasured.

[SCBench task/mode counters and provenance](scbench27b_reads.json)

## KVzip on LongBench and RULER

The 10,750 completed quality cases use query-visible reconstruction with uniform
GQA budgets. The stored per-case grades reproduce the displayed quality macros.
The read table uses paired cases with recorded sparse events for both methods.
Its read fractions average task means on LongBench and task/length cell means
on RULER. Short generations without sparse events remain in the quality macro.

| Model | Benchmark | Read / quality cases | PrismKV reads | KVzip reads | KVzip quality |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | LongBench | 3,664/3,750 | 7.2199% | 7.2196% | 42.274 |
| Qwen3.8-27B | LongBench | 3,546/3,750 | 8.6693% | 8.6692% | 45.886 |
| Qwen3.8-27B | RULER | 3,200/3,250 | 7.2236% | 7.2235% | 36.520 |

The dense warmup is outside the sparse counters. KVzip hits the rounded old-token
target on each of the 10,410 counted cases, and the measured tail means follow
the dense-tail count formula. The 27B LongBench cohort includes original and
repair receipts without duplicate completions. Seven same-run Full supplements
remain separate from the legacy Full reference.

[KVzip task/length counters and provenance](kvzip_reads.json)
