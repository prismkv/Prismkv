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

The packaged 8B checkpoints use 28 synthetic training contexts and 4000 steps
per dictionary, with a generic middle-20%-80% auxiliary positive sampler gated
by teacher mass >=2%. The inherited 27B dictionaries have a longer, different
continuation history. See the pretrained bundle notes. Calling both auxiliary
samplers simply "evidence matching" would hide a material recipe difference.

[The 8B training-context manifest](training_contexts_8b.json) records all 28
identities, lengths, generation caps and exact input fingerprints. The training
and 3750 LongBench input files were checked against their saved manifest hashes;
no identical wrapped token sequence was found across these sets. This is an
exact-duplicate check, not a semantic/near-duplicate audit or a universal
train-test separation guarantee.

The [27B input inventory](training_contexts_27b.json) now provides308distinct
full-input fingerprints from28-,112- and168-context synthetic pools. The later
280-context pool has112/112/56length targets at16K/32K/64K and9,153,497actual
input tokens. Both source row and completion/manifest hashes were checked.
There is no exact wrapped-token-sequence collision with the3750original27B
LongBench or3250RULER inputs. Within these local manifests, train and held-out
instance/template hashes are also disjoint.
This excludes the earlier4/12-context initialization ancestors, near duplicates,
other benchmark splits and a complete capture/cost audit. It is an inventory
of these three pools, not proof of complete historical data independence.
Training-use manifests for43inherited dictionaries request280contexts;21fresh
configs use28contexts/4000updates. Frozen initialization and continuation
histories still differ across these groups.

The original 8B capture records recover 142.069 seconds of prefill and 519.021
seconds of decode over 10312 generated tokens. These exclude model loading,
initial K serialization and other unrecorded overhead, so their sum is not the
complete collection cost. SAE optimizer time must also be reported separately.
