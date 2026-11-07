# Paired SeerAttention-R quality evaluation

Use the frozen 26 RULER prompts from the Quest pilot, with one prompt per task at 8K and 32K. All arms share one Qwen3-8B prefill, greedy decoding, stopping rules and the initial dense decode forward. Run Full, native PrismKV Top16, a source-style Seer selector, and a Seer selector restricted to historical blocks with a complete generated tail.

The common historical index ends at the last complete 64-token block. Calibrate one scalar per prompt from the mean PrismKV indexed reads across all groups on the existing 16-step Full-trajectory replay. Round down to complete blocks. The source-style selector uses the released learned ranking and forced last block over all currently available blocks. The tail-matched variant ranks only the indexed history using the same scores and includes every token after the index. This budget policy is mean calibration. Report actual reads on each generated trajectory, with separate old-history and generated-tail counts and warmup denominators.

Pin the released gate parameters and source. Use the validated layer cursor correction for both Seer arms. Add one return value to expose gate probabilities for the tail-matched arm. Capture post-normalization, pre-RoPE Q/K from Qwen3, and pass the model's base RoPE to the gate at the original position indices. Use the common indexed AV backend for all sparse arms. These are quality measurements on a reference port. Record all 36 layers and 288 KV groups. Keep the score slots blank until the run and verification finish.

# SeerAttention-R paired quality follow-up

Apply the 26-case paired protocol to the previously frozen 195-case RULER budget cohort. This includes five prompts per task at 8K, 16K and 32K. The input identities and disjointness audit come from that completed run. Use the identical four-arm worker, Gate source, weights and score replay, with no outcome-based changes to case inclusion, budgets or arithmetic.

The 195 inputs are disjoint from the 26 pilot prompts. The budget cohort has previously been evaluated with PrismKV, so the new study extends the baseline comparison on historical data. The four arms share the same model, prefill and AV policy described in the paired protocol. The old-history index boundary is rounded to 64 tokens in every sparse arm. The source-style Seer policy retains its forced last block and can route across generated blocks. The history-only Seer policy shares PrismKV's complete generated tail. Keep these two policies separate and report actual reads alongside scores.
