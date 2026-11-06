# Validation scope

Current results:

-16 CPU tests passed without skips, using Torch2.4.0 and Transformers4.57.1.
-Four of these execute a tiny randomly initialized two-layer Qwen3 backbone:
  all-read equivalence, backend restoration, actual greedy generation from a
  one-token prefix, offset chunked prefill, full-group sparse selection under a
  cap, and incremental tail indexing. They do not measure pretrained-model quality.
-Three private source-parity seeds compare the training objective with historical
  implementation definitions on a small F32 test shape. Largest loss difference
  is1.2e-7 and parameter-gradient difference1.7e-8. This is a component check,
  not training-data or full-checkpoint provenance certification.
-The four extracted PPU8B sources compile into a shared library. On PPU-ZW810,
  six standalone synthetic cases pass exact-feature/ID checks and36 AV checks
  across split/scheduling choices. The maximum relativeL2 error is0.001683 and
  maximum absolute error0.003902. Cases include ties, few/empty buckets, variable
  counts, overlapping lists and dense tails, through64K history. See
  [the numerical report](native8b_validation.json) and `examples/check_native8b.py`.
  This validates neither original SAE index-builder performance nor pretrained
  model integration or the paper's latency.
-Historical 8B/27B encoders were exported and reloaded as tensor-only bundles:
  288/64 dictionaries, exact tensors and scales, each source checked against
  its DONE hash. This does not establish new pretrained-adapter equivalence.

The initial environment had Torch2.4 and a much newer Transformers development
build that required a missing Torch API. A project-local pinned4.57.1 dependency
stack was used instead; existing experiment environments were not modified.

Still pending: pretrained-scale generation, hybrid27B integration, original
training/benchmark manifests, model-scale native comparisons, full-model peak memory
and full downstream benchmark launchers. No test is substituted for those.

The synthetic witness records582,458,880peak PyTorch-allocated bytes and
876,609,536reserved bytes for one8-KV-group layer with its fixture/index/build
temporaries. These numbers are not full-model peak memory or a production
request measurement. Live binding-owned buffers are listed separately percase.
