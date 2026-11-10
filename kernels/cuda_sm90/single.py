"""True 4K continuation with a frozen prefix index and a fully read tail."""
import json
import statistics

import torch
from flash_attn import flash_attn_with_kvcache

from measure import allocator_peak, event_times, sync, wall
from model import Session, load_weights, sha
from native import cuda_empty
from run import LAYERS, OUTPUT_TOKENS, single_plan
from stage_cases import LONG_SHA, crop


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else (eos,))


def full_av(q, k, v, lengths):
    return flash_attn_with_kvcache(q, k, v, cache_seqlens=lengths, softmax_scale=1/16, causal=False)


def layout_probe(session, q, k, v, history, tail, repeat, flush, attend=None, graph=False):
    ops = session.ops
    attend = ops.attend if attend is None else attend
    counts = ops.counts.tolist()
    largest = max(counts)
    packed_k = cuda_empty(1, largest + tail, 4, 256, dtype=k.dtype)
    packed_v = torch.empty_like(packed_k)
    identity = torch.arange(ops.history, device='cuda', dtype=torch.int32).repeat(4, 1)
    positions = torch.tensor([largest], device='cuda', dtype=torch.int32)
    total = positions + tail
    lengths = session.histories + tail

    def gather():
        ops.kernels.call('gather_kv', k, v, ops.ids, ops.counts, packed_k, packed_v,
                         history, tail, ops.history, largest)

    gather()
    for g, count in enumerate(counts):
        expect = ops.ids[g, :count].long()
        if not torch.equal(packed_k[0, :count, g], k[0, expect, g]):
            raise RuntimeError('gathered keys differ from frozen selection')
        if not torch.equal(packed_v[0, :count, g], v[0, expect, g]):
            raise RuntimeError('gathered values differ from frozen selection')
        for original, packed in ((k, packed_k), (v, packed_v)):
            if not torch.equal(packed[0, largest:, g], original[0, history:history + tail, g]):
                raise RuntimeError('gather did not retain the complete generated tail')
    indexed = lambda: attend(q, k, v, session.histories, lengths)
    contiguous = lambda: attend(q, packed_k, packed_v, positions, total, ids=identity)
    left = indexed().clone()
    torch.testing.assert_close(left, contiguous(), rtol=0, atol=0)
    result = dict(counts=counts, indexed_av=event_times(indexed, repeat, flush),
                  prepacked_av=event_times(contiguous, repeat, flush), explicit_gather=event_times(gather, repeat, flush),
                  temporary_bytes=2 * packed_k.numel() * packed_k.element_size())
    if graph:
        from triton.testing import do_bench_cudagraph

        def gather_and_av():
            gather()
            contiguous()

        result['graph_ms'] = {name: do_bench_cudagraph(fn, rep=25) for name, fn in
                              (('indexed_av', indexed), ('prepacked_av', contiguous),
                               ('explicit_gather', gather), ('gather_and_av', gather_and_av))}
    return result


def profiles(session, emitted, seed, restore, repeats, gather_layer, tune_splits):
    H, decode_forwards = session.history, len(emitted) - 1
    chosen = {int((decode_forwards-1)*i/4) for i in range(5)}
    inputs = torch.tensor(emitted[:-1], device='cuda').view(-1, 1, 1)
    flush = torch.zeros(64 * 1024**2, dtype=torch.uint8, device='cuda')
    ops = session.ops
    restore(seed)
    session.method = 'full'
    rows = []
    for step, ids in enumerate(inputs):
        session.capture_queries = {} if step in chosen else None
        session.step(ids)
        queries = session.capture_queries
        session.capture_queries = None
        if queries is not None:
            tail = step + 1
            lengths = session.histories + tail

            def chain(kind, splits=None):
                for layer in LAYERS:
                    q, k, v = queries[layer], session.keys[layer], session.values[layer]
                    index = session.indices[layer]
                    if kind == 'full':
                        full_av(q, k, v, lengths)
                    elif kind == 'score':
                        ops.score(q, index)
                    else:
                        ops.route(q, index)
                        if kind == 'sparse':
                            ops.attend(q, k, v, session.histories, lengths, splits=splits)

            # A frozen probe per layer keeps AV-only timings independent of routing.
            counts = []
            frozen = []
            for layer in LAYERS:
                ops.route(queries[layer], session.indices[layer])
                counts.append(dict(layer=layer, unique=ops.counts.tolist(), raw=ops.raw.tolist()))
                frozen.append((ops.ids.clone(), ops.counts.clone(), ops.features.clone(), ops.selected.clone()))

            def av_only():
                for which, layer in enumerate(LAYERS):
                    ids, counts, _, _ = frozen[which]
                    ops.attend(queries[layer], session.keys[layer], session.values[layer],
                               session.histories, lengths, ids=ids, counts=counts)

            def merge_only(mode=None):
                for which, layer in enumerate(LAYERS):
                    _, _, features, selected = frozen[which]
                    ops.merge(1, session.indices[layer], features, selected, mode)

            modes = ('full', 'sparse') if len(rows) % 2 == 0 else ('sparse', 'full')
            row = dict(decode_forward=step + 1, tail_tokens=tail, group_counts=counts)
            for mode in modes:
                row[mode] = event_times(lambda: chain(mode), repeats, flush)
            row['score'] = event_times(lambda: chain('score'), repeats, flush)
            row['merge'] = event_times(merge_only, repeats, flush)
            row['av_only'] = event_times(av_only, repeats, flush)
            if tune_splits:
                row['split_diagnostic'] = {splits: event_times(lambda: chain('sparse', splits), repeats, flush)
                                           for splits in (32, 64, 128)}
                row['union_diagnostic'] = {mode: event_times(lambda: merge_only(mode), repeats, flush)
                                           for mode in (0, 1, 2, 3)}
            if gather_layer is not None:
                ops.route(queries[gather_layer], session.indices[gather_layer])
                row['layout_layer'] = gather_layer
                row['layout'] = layout_probe(session, queries[gather_layer], session.keys[gather_layer],
                                            session.values[gather_layer], H, tail, repeats, flush)
            rows.append(row)
        session.position.add_(1)
    return rows


def run(args, model, kernels, memory):
    plan = single_plan(args.history, args.output_tokens)
    H, steps = plan['prompt_tokens'], plan['decode_forwards']
    prompt_path = args.assets / 'long_source.json'
    if sha(prompt_path) != LONG_SHA:
        raise ValueError('expected the archived 256K speed prompt')
    ids = crop(json.loads(prompt_path.read_bytes())['ids'], H)
    prompt = torch.tensor([ids], device='cuda')
    memory.phase = 'cache_allocate'
    hopper = args.av_backend != 'legacy'
    splits = args.splits or (128 if hopper else None)
    session = Session(model, kernels, 1, H, args.output_tokens, splits=splits, union_mode=args.union_mode)
    memory.phase = 'prefill_validation'
    prefill_witness = session.prefill_witness(prompt[:, -80:])
    memory.phase = 'prefill'
    print(f'prefill: {H:,} prompt tokens', flush=True)
    torch.cuda.reset_peak_memory_stats()
    seed, prefill_s = wall(lambda: session.prefill(prompt, 0, args.chunk).argmax(-1).reshape(1, 1))
    prefill_memory = allocator_peak()
    session.extent_to(1)
    session.histories.fill_(H)
    initial = session.snapshot()
    token = torch.empty_like(seed)
    outputs = torch.empty(steps, device='cuda', dtype=torch.long)

    def restore(value):
        session.restore(initial)
        session.position.fill_(H)
        token.copy_(value)

    def step():
        logits = session.step(token)
        token.copy_(logits.argmax(-1).reshape(1, 1))
        session.position.add_(1)
        return logits

    def continuation(step_fn, count):
        for i in range(count):
            logits = step_fn()
            outputs[i:i + 1].copy_(token[0])
        return logits

    arms = {}
    index_s = None
    for method in ('full', 'sparse'):
        session.method = method
        if method == 'sparse':
            memory.phase = 'sae_load'
            session.enable_sparse(load_weights(args.assets / 'weights.pt'))
            if args.route_backend == 'hopper-warp':
                session.ops.enable_hopper_route(args.route_library)
            if hopper:
                session.ops.enable_hopper_av(pv_mode=2 if args.av_backend == 'hopper-compensated' else 1)
            memory.phase = 'index_build'
            print('building the prefix index once', flush=True)
            torch.cuda.reset_peak_memory_stats()
            _, index_s = wall(lambda: session.build_index(0, H))
            index_memory = allocator_peak()
        memory.phase = f'{method}_decode_warmup'
        restore(seed)
        count = min(8, steps)
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            eager = continuation(step, count).clone()
        torch.cuda.current_stream().wait_stream(warm_stream)
        expected = outputs[:count].tolist()
        step_fn = step
        capture_s = None
        if args.engine == 'graph':
            memory.phase = f'{method}_graph_capture'
            restore(seed)
            sync()
            graph = torch.cuda.CUDAGraph()

            def capture():
                with torch.cuda.graph(graph):
                    fixed = step()
                return fixed

            logits, capture_s = wall(capture)

            def graph_step():
                graph.replay()
                return logits

            restore(seed)
            got = continuation(graph_step, count)
            if outputs[:count].tolist() != expected:
                raise RuntimeError(f'{method} Graph/eager IDs disagree')
            torch.testing.assert_close(got, eager, rtol=0, atol=0)
            step_fn = graph_step
        memory.phase = f'{method}_decode'
        torch.cuda.reset_peak_memory_stats()
        times, emitted = [], None
        for _ in range(args.repeats):
            restore(seed)
            print(f'{method}: timing {steps:,} consecutive decode forwards', flush=True)
            _, seconds = wall(lambda: continuation(step_fn, steps))
            ids = [seed.item(), *outputs.tolist()]
            if emitted is not None and ids != emitted:
                raise RuntimeError(f'{method} output changed across repeats')
            emitted = ids
            times.append(seconds)
        arms[method] = dict(decode_seconds=times, decode_ms_per_forward=statistics.median(times)*1000/steps,
                            emitted=emitted, eos_positions=[i+1 for i,t in enumerate(emitted) if t in eos_ids(model)],
                            graph_capture_s=capture_s, allocator_peak=allocator_peak(), inventory=session.inventory())
        if args.engine == 'graph':
            del graph_step, step_fn, graph, got, logits
    memory.phase = 'frozen_attention_diagnostics'
    if not args.skip_profile:
        print('replaying Full tokens for matched Attention diagnostics', flush=True)
    module = profiles(session, arms['full']['emitted'], seed, restore, args.profile_repeats,
                      args.gather_layer, args.split_diagnostic) if not args.skip_profile else []
    session.close()
    full, sparse = arms['full']['decode_ms_per_forward'], arms['sparse']['decode_ms_per_forward']
    amort = index_s * 1000 / plan['output_tokens']
    result = dict(**plan, status='complete', engine=args.engine, fixed_output_even_after_eos=True,
                  protocol='formal_4k' if args.output_tokens == OUTPUT_TOKENS else 'diagnostic_horizon',
                  arm_order=['full', 'sparse'], recurrent_snapshot_bytes=sum(
                      v.numel()*v.element_size() for row in initial.values() for v in row.values()),
                  prefill_s=prefill_s, prefill_allocator_peak=prefill_memory, prefill_witness=prefill_witness,
                  index_s=index_s, index_allocator_peak=index_memory,
                  index_amortized_ms_per_output_token=amort,
                  decode_speedup=full/sparse,
                  decode_speedup_with_amortized_index=(full*steps/1000)/(index_s+sparse*steps/1000),
                  total_decode_seconds_with_index=index_s+sparse*steps/1000,
                  decode_amortized_ms_per_output_token=(sparse*steps/1000+index_s)*1000/plan['output_tokens'],
                  request_s_excluding_startup={'full': prefill_s+full*steps/1000,
                                               'sparse': prefill_s+index_s+sparse*steps/1000},
                  arms=arms, attention=module)
    if module:
        full_attention = statistics.mean(row['full']['ms'] for row in module)
        sparse_attention = statistics.mean(row['sparse']['ms'] for row in module)
        result['attention_sample_mean_ms'] = {'full': full_attention, 'sparse': sparse_attention}
        result['attention_speedup'] = full_attention/sparse_attention
        result['attention_speedup_with_amortized_index'] = full_attention/(sparse_attention+amort)
        result['mean_sampled_old_history_read_fraction'] = statistics.mean(
            sum(sum(layer['unique']) for layer in row['group_counts'])/(H*64) for row in module)
    result['request_speedup_excluding_startup'] = (
        result['request_s_excluding_startup']['full']/result['request_s_excluding_startup']['sparse'])
    return result
