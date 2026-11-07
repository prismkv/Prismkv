# LongProc: query reranking with one fixed SAE index

Complete16-case diagnostic, one fixed index0 input from each historical task/length subset. All64 standard-attention KV groups are sparse; the other48 linear-attention layers and backbone weights are unchanged. This is not the880-case benchmark, a significance test or a latency result.

|Arm|Official subset macro(0-100)|Mean old unique reads|Read cases|Cap-terminated cases|
|---|---:|---:|---:|---:|
|full|62.3742|100% Full policy; no sparse counter|0|5|
|sae_native|56.2500|13.1826%|16|5|
|dynamic_matched|56.2500|12.9791%|16|5|
|frozen_matched|7.1324|12.4189%|16|9|

## Intervention and budget meaning

NativeSAE uses its unchanged Top16 rule. Dynamic-matched and frozen-matched use the same SAE dictionaries/memberships, original-K summaries, original-KV indexed attention, complete generated tail, output caps and seeded attention-independent boundary priority. They follow the same Full-derived per-step/layer/KV-group UNIQUE-count schedule. Every matched count was checked against that schedule. This is an offline control using teacher-derived counts, not an online deployment algorithm that knows future Full behavior.

The only routing-policy intervention between matched arms is bucket ranking: dynamic reranks from current Q each step; frozen holds the ranking from the first routed decode step. Frozen is not a claim that an identical token set is kept when the target count changes. If an arm outlives the Full trajectory, the last target count for that layer is repeated. Own output trajectories/lengths differ, so their FINAL mean read fractions need not be equal even though same-step targets match.

Full uses GQAFlash; the two matched sparse arms use the same indexed AV operator. Initial forced8-token all-read and repeated-Full checks passed, but do not certify whole-trajectory arithmetic equivalence between Flash and indexed AV. The strongest controlled inference is dynamic versus frozen, which share the operator.

## Findings and limits

Dynamic-matched scores56.250 versus frozen7.132, a49.118-point difference in this pilot. This supports a benefit from current-query reranking within the fixed SAE-index protocol while holding generated-tail eligibility fixed. **It is not a measurement of what fraction of the historical Prism-versus-KVzip gap comes from retrieval. The frozen-ranking arm is not KVzip, which uses a different reconstruction-based selection procedure.** It also does not establish that SAE is uniquely responsible or that all static compressors fail.

Full scores62.374. NativeSAE and dynamic-matched both score56.250, including a failure on the8K HTML-to-TSV case where Full scores97.987. Do not hide that loss or describe this pilot as full-quality preservation. Countdown8K, path-traversal cases and travel8K also expose Full/task-cap limitations; all cases remain included. Five native/dynamic outputs and nine frozen outputs hit the unchanged caps.

## Scoring/provenance repair

The original CPU executable-task scorer environment failed on two cases. A private pinned CPU runtime repair preserved official scorer functions, executable code section and sandbox restrictions;24host/task replay witnesses passed. ALLfive executable-family inputs(Countdown/Pseudo) were rerun in the repaired version, including already completed duplicates. The other11non-executable cases use the original protocol. This version selection is by task family, never by score; old failed/completed artifacts remain preserved. Source, result/official-score hashes, identity, target counts and tail-counter consistency were checked before this64-row export.

Historical task templates/development inputs are not an independent heldout benchmark. Raw old-history reads exclude the full generated tail and48unmodified linear layers. Full is an all-history policy; its missing sparse counters are not zeros. The CSV records output length, truncation, clamp counts, source version and closure hashes.
