"""Open-loop FIFO serving with dynamically admitted requests and native batched decode."""
from collections import deque
import gzip
import hashlib
import json
import random
import statistics
import time

import torch

from measure import allocator_peak, quantile, sync, wall
from pack import PINNED, sha
from stage_cases import MANIFEST_SHA
from run import CONTEXT_LIMIT


def extend_requests(rows, history, batch, repeat, output_cap=64):
    """Use disjoint source panels within each request and retain its final question."""
    size = min(history * 1024, CONTEXT_LIMIT - output_cap)
    result = []
    for i, target in enumerate(rows[:batch+2]):
        rng = random.Random(20261110 + 10000*repeat + 100*history + i)
        fillers = [row for row in rows if row['source'] != target['source']]
        rng.shuffle(fillers)
        ids, panels = [], []
        remaining = size - len(target['ids'])
        for row in fillers:
            if remaining <= 0:
                break
            panel = row['ids'][:remaining]
            ids.extend(panel)
            panels.append(dict(source=row['source'], source_sha=row['source_sha'], tokens=len(panel)))
            remaining -= len(panel)
        if remaining:
            raise ValueError('frozen panels cannot supply the requested context length')
        ids.extend(target['ids'])
        panels.append(dict(source=target['source'], source_sha=target['source_sha'], tokens=len(target['ids'])))
        cap = output_cap if i == 0 else target['cap'] * output_cap // 64
        result.append(dict(target, ids=ids, panels=panels, cap=cap))
    return result


def load_requests(path, history, batch, repeat, smoke=False, output_cap=64):
    if sha(path) != PINNED['serving.json.gz']:
        raise ValueError('serving bundle does not match the frozen artifact')
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    if bundle['manifest_sha'] != MANIFEST_SHA:
        raise ValueError('invalid serving corpus identity')
    long_context = history > 32 or output_cap > 64
    if smoke and long_context:
        raise ValueError('long contexts use the full-length serving pilot')
    rows = bundle['cohorts'][f'{min(history, 32)}:{repeat}']
    if long_context:
        rows = extend_requests(rows, history, batch, repeat, output_cap)
    limit = batch+2 if smoke or long_context else max(24, 3*batch)
    if limit > len(rows):
        raise ValueError('not enough frozen requests')
    result = []
    for i, row in enumerate(rows[:limit]):
        ids = row['ids'][-(512 + 32*i):] if smoke else row['ids']
        result.append(dict(i=i, source=row['source'], source_sha=row['source_sha'], ids=ids,
                           input_sha=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                           history=len(ids), cap=4+i%3 if smoke else row['cap'],
                           arrival=0. if smoke else .025*i/max(1, batch//2)))
        if long_context:
            result[-1]['panels'] = row['panels']
    return result


class GraphDecode:
    """Capture fixed extents over shared, preallocated KV and recurrent storage."""

    def __init__(self, session, warm, chunk, memory):
        memory.phase = 'serve_graph_warmup'
        self.session, self.graphs, self.capture_s = session, {}, {}
        seeds = []
        for slot in range(session.slots):
            logits = session.prefill(warm, slot, chunk)
            seeds.append(logits.argmax(-1))
            if session.method == 'sparse':
                session.build_index(slot, warm.shape[1])
        seed = torch.stack(seeds).reshape(-1, 1)
        initial = session.snapshot()
        session.histories.fill_(warm.shape[1])
        session.position.fill_(warm.shape[1])
        for extent in range(1, session.slots+1):
            session.extent_to(extent)
            inputs = seed[:extent].clone()
            session.restore(initial)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                reference = session.step(inputs).clone()
            torch.cuda.current_stream().wait_stream(side)
            session.restore(initial)
            graph = torch.cuda.CUDAGraph()
            memory.phase = 'serve_graph_capture'

            def capture():
                with torch.cuda.graph(graph):
                    logits = session.step(inputs)
                    tokens = logits.argmax(-1)
                return logits, tokens

            (logits, tokens), seconds = wall(capture)
            session.restore(initial)
            graph.replay()
            torch.testing.assert_close(logits, reference, rtol=0, atol=0)
            self.graphs[extent] = graph, inputs, tokens
            self.capture_s[extent] = seconds
        session.restore(initial)

    def __call__(self, token, position, histories):
        extent = len(token)
        graph, inputs, output = self.graphs[extent]
        # The scheduler already needs the emitted IDs on the host. A single
        # packed copy supplies all per-step inputs without recapturing graphs.
        host = torch.tensor([position, histories, [row[0] for row in token]], dtype=torch.int32)
        panel = host.to(self.session.position.device)
        self.session.position[:extent].copy_(panel[0])
        self.session.histories[:extent].copy_(panel[1])
        inputs[:, 0].copy_(panel[2])
        graph.replay()
        return output.tolist()


def simulate(session, requests, chunk, memory, phase='serve', progress=False, graph_decode=None):
    pending = deque(requests)
    active = [None] * session.slots
    finished, steps = [], []
    old_opportunities = dense_tail = 0
    started = time.perf_counter()
    while pending or any(active):
        for slot in range(session.slots):
            if active[slot] is not None or not pending or pending[0]['arrival'] > time.perf_counter()-started:
                continue
            request = pending.popleft()
            admitted = time.perf_counter()-started
            ids = torch.tensor([request['ids']], device=session.position.device)
            memory.phase = f'{phase}_prefill'
            logits, prefill_s = wall(lambda: session.prefill(ids, slot, chunk))
            first = logits.argmax(-1).item()
            index_s = 0.
            if session.method == 'sparse':
                memory.phase = f'{phase}_index'
                _, index_s = wall(lambda: session.build_index(slot, request['history']))
            active[slot] = dict(request, tokens=[first], emission=[time.perf_counter()-started],
                                admission_s=admitted, prefill_s=prefill_s, index_s=index_s)
            if progress:
                print(f'admitted request {request["i"]+1}/{len(requests)}: prefill {prefill_s:.2f}s, index {index_s:.2f}s', flush=True)
        live = [i for i,row in enumerate(active) if row is not None]
        if not live:
            if pending:
                time.sleep(max(0., pending[0]['arrival']-(time.perf_counter()-started)))
            continue
        extent = live[-1]+1
        session.extent_to(extent)
        rows = active[:extent]
        token = [[row['tokens'][-1] if row else 0] for row in rows]
        position = [row['history']+len(row['tokens'])-1 if row else -1 for row in rows]
        histories = [row['history'] if row else 0 for row in rows]
        memory.phase = f'{phase}_decode'
        t0 = time.perf_counter()
        if graph_decode is None:
            session.position[:extent].copy_(torch.tensor(position, device=session.position.device, dtype=torch.int32))
            session.histories[:extent].copy_(torch.tensor(histories, device=session.position.device, dtype=torch.int32))
            tokens = session.step(torch.tensor(token, device=session.position.device)).argmax(-1).tolist()
        else:
            tokens = graph_decode(token, position, histories)
        emitted = time.perf_counter()
        steps.append((len(live), extent, emitted-t0))
        for slot in live:
            row = active[slot]
            old_opportunities += row['history'] * 64
            dense_tail += len(row['tokens']) * 64
            row['tokens'].append(tokens[slot])
            row['emission'].append(emitted-started)
            if len(row['tokens']) == row['cap']:
                del row['ids']
                row['ttft_s'] = row['emission'][0]-row['arrival']
                row['e2e_s'] = row['emission'][-1]-row['arrival']
                row['slot'] = slot
                finished.append(row)
                active[slot] = None
    elapsed = time.perf_counter()-started
    total = sum(len(row['tokens']) for row in finished)
    if len(finished) != len(requests) or total != sum(row['cap'] for row in requests):
        raise RuntimeError('not all fixed output horizons completed')
    if sum(n for n, _, _ in steps) != total - len(finished):
        raise RuntimeError('active decode counts do not match output lengths')
    inter = [right-left for row in finished for left,right in zip(row['emission'], row['emission'][1:])]
    return dict(wall_s=elapsed, request_count=len(finished), output_tokens=total,
                request_per_s=len(finished)/elapsed, output_tokens_per_s=total/elapsed,
                decode_tokens_per_s=(total-len(finished))/sum(x[2] for x in steps),
                ttft_p50_s=quantile([x['ttft_s'] for x in finished], .5),
                ttft_p95_s=quantile([x['ttft_s'] for x in finished], .95),
                itl_p95_ms=1000*quantile(inter, .95), e2e_p95_s=quantile([x['e2e_s'] for x in finished], .95),
                mean_active=statistics.mean(x[0] for x in steps), max_active=max(x[0] for x in steps),
                mean_physical_lanes=statistics.mean(x[1] for x in steps),
                active_batch_histogram={size: sum(n == size for n, _, _ in steps) for size in range(1, session.slots+1)},
                index_s_per_request=statistics.mean(x['index_s'] for x in finished),
                old_group_token_opportunities=old_opportunities, dense_tail_group_tokens=dense_tail,
                old_group_tokens=old_opportunities if session.method == 'full' else
                None if session.reads is None else session.reads.sum().item(),
                measured_with_read_audit=session.reads is not None, **session.inventory(),
                allocator_peak=allocator_peak(), request_stats=finished, steps=steps)


def run(args, model, kernels, memory):
    from model import Session, load_weights
    from single import eos_ids

    rows = load_requests(args.assets/'serving.json.gz', args.history, args.batch, args.repeat,
                         args.smoke, args.output_cap)
    long_pilot = args.history > 32 or args.output_cap > 64
    H = max(row['history'] for row in rows) if args.smoke or long_pilot else args.history*1024
    horizon = max(row['cap'] for row in rows)
    memory.phase = 'serve_allocate'
    weights = load_weights(args.assets/'weights.pt') if args.method == 'sparse' else None

    def create_session(slots):
        session = Session(model, kernels, slots, H, horizon, weights, args.splits, args.union_mode)
        session.method = args.method
        if args.method == 'sparse':
            if args.route_backend == 'hopper-warp':
                session.ops.enable_hopper_route(args.route_library)
            if args.av_backend != 'legacy':
                session.ops.enable_hopper_av(pv_mode=2 if args.av_backend == 'hopper-compensated' else 1)
        return session

    session = create_session(args.batch)
    try:
        graph_decode = None
        if args.engine == 'graph':
            warm_ids = torch.tensor([rows[0]['ids'][-min(H, 512):]], device='cuda')
            graph_decode = GraphDecode(session, warm_ids, args.chunk, memory)
        elif not args.smoke:
            warm = dict(rows[0], history=512, ids=rows[0]['ids'][-512:], cap=3, arrival=0.)
            simulate(session, [warm], args.chunk, memory, phase='serve_warmup')
            sync()
        if args.audit_reads and args.method == 'sparse':
            session.reads = torch.zeros(args.batch, device='cuda', dtype=torch.int64)
        torch.cuda.reset_peak_memory_stats()
        result = simulate(session, rows, args.chunk, memory, progress=long_pilot, graph_decode=graph_decode)
        if graph_decode is not None:
            result['graph_capture_s_by_extent'] = graph_decode.capture_s
    finally:
        session.close()
    if args.smoke:
        candidates = {row['i']: row['tokens'] for row in result['request_stats']}
        serial = create_session(1)
        try:
            for row in rows:
                individual = simulate(serial, [row], args.chunk, memory, phase='serve_serial_check')
                if individual['request_stats'][0]['tokens'] != candidates[row['i']]:
                    raise RuntimeError(f"batch/serial output mismatch for request {row['i']}")
        finally:
            serial.close()
        if result['max_active'] != args.batch or len({row['slot'] for row in result['request_stats']}) != args.batch:
            raise RuntimeError('smoke did not fill its scheduled batch')
        result['serial_exact'] = len(rows)
    eos = eos_ids(model)
    for row in result['request_stats']:
        row['eos_positions'] = [i+1 for i,t in enumerate(row['tokens']) if t in eos]
    result['post_eos_tokens'] = sum(len(row['tokens'])-row['eos_positions'][0]
                                    for row in result['request_stats'] if row['eos_positions'])
    return dict(status='complete', method=args.method, nominal_prompt_tokens=H,
                requested_concurrency=args.batch, repeat=args.repeat, smoke=args.smoke,
                workload='concatenated_frozen_longbench_pilot' if long_pilot else 'frozen_longbench',
                first_token_release='after_prefill_and_index', fixed_cap_even_after_eos=True, **result)
