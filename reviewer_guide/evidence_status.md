# Evidence status

The submitted default is F=3000, average K activation budget 8, bucket Top16,
with count/mean/diagonal-variance routing. New compact SAE and Q-activation
experiments are supplementary and must be labeled by exact configuration.

Pending release items:

- [x] Tiny-backbone reference integration and pinned optional dependencies.
- [x] Tensor-only packaging of historical 288/64 model-matched SAE encoders.
- [ ] Pretrained-scale/hybrid integration and full evaluation launchers.
- [ ] Original checkpoint and training-context provenance, train/test audit.
- [x] Standalone PPU8B library compilation.
- [x] Standalone native-binding GPU correctness checks, six synthetic cases through64K.
- [x] Verified27B RULER task-length and LongProc subset quality/read tables in [results](results/README.md).
- [ ] New configuration quality/read/latency curves with all standard-attention groups sparse.
- [ ] Matched-retention/tail LongProc control and strong non-SAE organizers.
- [ ] Peak index/workspace memory measurements.

No incomplete experiment is presented as a final score. New-card timing and
production serving are not included in the present measurements.
