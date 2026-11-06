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

The CPU tests verify math and state semantics. They do not establish downstream
quality, GPU latency, mainstream GPU support, or production-serving performance.
