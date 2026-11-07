# Reviewer navigation

This guide distinguishes the submitted method from supplementary variants.

| Question | Read | Status |
|---|---|---|
| What uses SAE at decode? | `core.bucket_scores`; [algorithm](algorithm.md) | Main router uses raw Q and raw-K moments; Q activation is separately labeled |
| What are the loss terms? | `training.py`; [training](training.md) | Component reference; historical checkpoint/data provenance still being packaged |
| How are overlaps removed? | `kernels/ppu_8b/union.cu`; [systems](systems.md) | Bitwise union plus sorted token compaction |
| Does attention gather KV first? | `kernels/ppu_8b/av_tuned.cu` | Native path loads original KV through IDs; portable reference gathers for correctness |
| How are 4 Q heads shared? | `route_parallel.cu`, `av_tuned.cu` | Per-Q normalization then aggregation, shared IDs, separate per-Q attention weights |
| Are budgets exactly identical? | [budget protocol](budget_protocol.md) | Archived calibration is not a per-step/per-group equality guarantee |
| Where are hyperparameter results? | [evidence status](evidence_status.md) | New experiments in progress; no incomplete grid presented as final |
| How can I run quality evaluation? | [Evaluation guide](evaluation.md), `examples/evaluate_hf.py`, `prepare_benchmark.py`, `score_benchmark.py` | 24 CPU tests; 14,000 input-format checks; one actual 8K public-pipeline witness per model. Not a full quality rerun |
| Does the model integration preserve dense behavior? | `tests/test_hf_reference.py`; [validation](validation.md) | Tiny-backbone causality controls and short 8B/hybrid 27B CLI witnesses passed. See the separate hybrid dependency checks below. |
| Where are the trained encoders? | [`pretrained/`](../pretrained/README.md) | All 288/64 dictionaries, source-DONE/hash checked, exact tensor/scales export; not new downstream validation |
| Does the standalone native binding compute the selected attention correctly? | [`native8b_validation.json`](native8b_validation.json), `examples/check_native8b.py` | Six synthetic PPU cases through64K; exact IDs/features and36 AV checks. Not a model-quality or latency reproduction |
| Where are task/length results and actual baseline reads? | [Verified result tables](results/README.md) |27B RULER65cells and LongProc16subsets; own-trajectory mean-budget and adapter limits explicit |
| What inputs trained the27B index? | [Input inventory](training_contexts_27b.json), [training limits](training.md) |308input fingerprints across28/112/168pools;43inherited/21fresh split explicit; not full ancestry or near-duplicate certification |
| What were the offline costs? | [Training phase accounting](training.md#27b-recorded-offline-phases) | 324 timed captures and 226 optimization-stage records. Includes capture, mining and optimization subtotals with explicit omissions. |
| How does simpler key organization compare? | [Supplementary organizer pilot](results/organizer_pilot/README.md) |58case common-operator diagnostic, exact matched counts, mixed task outcomes; training histories differ and no full-benchmark/latency claim |
| Are count, mean and variance each necessary? | [Seven score-term controls](results/score_terms_pilot/README.md) |Same58cases and fixed SAE buckets, matched counts; mean-only is competitive here, so no necessity claim |
| How does a closely related retrieval selector compare? | [Squeezed reference-port pilot](results/squeezed_pilot/README.md) |29cases; source-style per-Q budget differs, shared-budget adaptation has14infeasible cases; paired denominators and port limits explicit |
| What are actual reads on the 8B benchmarks? | [8B counter audit](results/actual_reads_8b/README.md) |31,250 baseline records and 7,000 reference identities checked; 50 shared RULER prefill-first-token mismatches remain an explicit quality-comparison limitation |
| Where are route phases and memory costs? | [Native8B phase/memory diagnostic](results/native_stage_profile/README.md) |32K/64K, same released kernels, all36layers; isolated stage graphs and complete chains kept separate; Torch allocator peak and retained-index storage disclosed, not physical device peak |

The CPU tests verify math and state semantics. They do not establish downstream
quality, GPU latency, mainstream GPU support, or production-serving performance.

[LongProc rerouting control](results/longproc_rerouting_pilot/README.md): complete16-case fixed-index pilot, same generated-tail policy, dynamic/frozen comparison with explicit Full/HTML loss and offline-budget limits.

The SeerAttention-R evaluation uses the released learned gate on two Qwen3-8B RULER cohorts and reports reads under both source-style and full-tail selection. [Paired Seer quality control](results/seer_gqa_control/README.md)

[Hybrid 27B reference](hybrid_reference.md) records the original execution
witness and its framework fingerprints. The [locked hybrid dependency recipe](hybrid_install.md)
records a fresh install, the Torch 2.4 schema patch, CPU contracts and a
pretrained 27B 8K RULER execution with all 64 encoders.

## Bucket-budget quality and module latency

[Paired Top4/8/16 pilot](results/bucket_budget/README.md) reports official scores, actual reads, complete-module Graph latency and separate eager model timing on the same 26 RULER development inputs. Short-context slowdowns and unchanged Top16-compute overhead are retained.

## Expanded budget follow-up

[195 nonpilot RULER cases](results/bucket_budget_followup/README.md) extend the same budget test to five samples per task at8K/16K/32K. Top8 loses4.453points toTop16; the small-pilot trend does not support a new default. Actual reads and complete-module Graph/eager-model timings are separate.
