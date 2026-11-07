# Fixed-budget follow-up on nonpilot examples

Recorded after the 26-case pilot, before inspecting follow-up outcomes.
Use Qwen3-8B, unchanged 288 encoders (F3000), original key assignment,
native moment ranking and indexed AV. Copy the pilot worker byte-for-byte.
The only new scientific dimensions are input sample count and the 16K length.

The frozen 3,250-case RULER manifest contains 50 examples per task-length cell.
For each of 13 tasks at 8K, 16K and 32K, sort by lexical case ID, exclude every
input selected in the preceding 26-case pilot and take the first five remaining
IDs. This yields 195 cases. Verify disjoint input-ID hashes against the pilot.
Do not filter on dense or sparse scores. Do not inspect labels to choose cases.
These are held out from the small pilot, not a newly collected benchmark or
certified disjoint from all historical development. Data/train ancestry limits
remain unchanged. No retraining or tuning on the new outputs.

Full and B4/B8/B16 remain the four arms. Keep original generation caps, the
common first dense forward and complete generated tail. Smaller B retains
native Top16 calculation plus masking; charge both to timing. Keep the
0-15 prompt-token dense remainder. Record all-forward and routed-only read
rates separately, with zero-sparse and capped-case counts.

Same 16-step Full-prefix teacher replay and Full Q/K/V snapshots as the pilot.
Separate eager model timing, eager complete-module timing, Graph complete-module
timing and Graph setup. Never call module timing an end-to-end Graph gain.
Run three eager model repeats and four module repeats, retaining all cases.
Audit literal memberships, AV, unchanged Top16 and Graph/eager parity before
accepting results. Use official scorer postprocessing and replay every grade.

Launch only on currently idle seven verified idle PPU slots after its existing controller guard
and /proc/PPU checks pass. Smoke one preselected 8K case through every stage
before the seven formal shards. No old output or live claim is changed.
Follow-up outcomes stay unreported until the cohort and checks are complete.
