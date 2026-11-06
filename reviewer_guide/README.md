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
| How can I run this? | root README, `examples/replay.py`, `tests/` | Portable core runnable; full-backbone packaging pending |
| Does the model integration preserve dense behavior? | `tests/test_hf_reference.py`; [validation](validation.md) | Tiny Qwen3 dense-equivalence and greedy-generation controls passed; pretrained-scale checks pending |
| Where are the trained encoders? | [`pretrained/`](../pretrained/README.md) | All 288/64 dictionaries, source-DONE/hash checked, exact tensor/scales export; not new downstream validation |
| Does the standalone native binding compute the selected attention correctly? | [`native8b_validation.json`](native8b_validation.json), `examples/check_native8b.py` | Six synthetic PPU cases through64K; exact IDs/features and36 AV checks. Not a model-quality or latency reproduction |
| Where are task/length results and actual baseline reads? | [Verified result tables](results/README.md) |27B RULER65cells and LongProc16subsets; own-trajectory mean-budget and adapter limits explicit |
| What inputs trained the27B index? | [Input inventory](training_contexts_27b.json), [training limits](training.md) |308input fingerprints across28/112/168pools;43inherited/21fresh split explicit; not full ancestry or near-duplicate certification |
| How does simpler key organization compare? | [Supplementary organizer pilot](results/organizer_pilot/README.md) |58case common-operator diagnostic, exact matched counts, mixed task outcomes; training histories differ and no full-benchmark/latency claim |
| Are count, mean and variance each necessary? | [Seven score-term controls](results/score_terms_pilot/README.md) |Same58cases and fixed SAE buckets, matched counts; mean-only is competitive here, so no necessity claim |
| How does a closely related retrieval selector compare? | [Squeezed reference-port pilot](results/squeezed_pilot/README.md) |29cases; source-style per-Q budget differs, shared-budget adaptation has14infeasible cases; paired denominators and port limits explicit |

The CPU tests verify math and state semantics. They do not establish downstream
quality, GPU latency, mainstream GPU support, or production-serving performance.
