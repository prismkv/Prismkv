# Training reference and reproducibility boundary

The reference implements normalized reconstruction MSE, anti-dead-feature
residual reconstruction with weight 1/32, query-key cosine margin matching,
selected-evidence matching and a utilization regularizer weighted by 0.5.
The frozen backbone is not updated.

Ordinary training pairs use a teacher-attention-weighted positive from Top2% of
historical K and 16 sampled negatives outside Top10%. Raw attention vectors
determine these labels before SAE scaling/centering. Evidence pairs require
separate training-only evidence metadata; they must not be mined from test answers.
The temporary softplus matching weight is `0.25*max(0,1-step/1000)`.

K selection is batch-global; individual tokens can activate zero, fewer, or more
than eight features. Q selection is per row. No fixed-eight claim applies to K.

The utilization term is
`F * sum(EMA_frequency * mean_softmax(K_preactivation/0.5)) - 1`.
Its value may be negative. It does not by itself measure uniform bucket sizes.
See actual empty-bucket rates and load histograms when assessing balance.

Checkpoint continuation, scale estimation, number of training contexts, sampling
and intermediate evidence definitions are part of each historical recipe. This
portable component implementation does not erase those differences. A complete
checkpoint-level provenance manifest is still pending; do not assume all groups
were initialized or trained on an identical schedule.
