"""Paired 128K replay and short real generation for candidate CUDA AV kernels."""
import argparse
import json
from pathlib import Path
import statistics

import torch
import triton
from triton.testing import do_bench_cudagraph

from hopper_av import HopperAV
from measure import event_times, wall
from model import Session, load_model, load_weights, sha
from native import Kernels
from run import LAYERS, write_json
from single import full_av
from stage_cases import LONG_SHA, crop


ROOT = Path(__file__).resolve().parent
HISTORY, CAPACITY = 131072, 135168
TAILS = (1, 1024, 2048, 3071, 4095)


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a-b).norm()/b.norm().clamp_min(1e-12)), float((a-b).abs().max())


def prefill_tail(session, emitted, first, end):
    session.decode = False
    session.slot = 0
    for offset in range(first, end, 1024):
        session.start = session.history + offset
        batch = emitted[:, offset:min(offset+1024, end)]
        positions = torch.arange(session.start, session.start + batch.shape[1], device='cuda')[None]
        session.forward(batch, positions, session.decode_cache)


def probe(session, opt, query, lengths, settings, flush, pv_mode):
    ops = session.ops
    idx = session.indices
    k, v = session.keys, session.values
    histories = session.histories
    old = ops.attend
    fn = {
        'legacy_64': lambda *a, **kw: old(*a, **kw, splits=64),
        'legacy_128': lambda *a, **kw: old(*a, **kw, splits=128),
    }
    for t, w, s in settings:
        fn[f'tc_{t}_{w}_{s}'] = lambda *a, t=t, w=w, s=s, **kw: opt.attend(
            *a, **kw, tile=t, warps=w, splits=s, pv_mode=pv_mode)
    selected = {}
    for layer in LAYERS:
        ops.route(query[layer], idx[layer])
        selected[layer] = ops.ids.clone(), ops.counts.clone()

    errors = {}
    for name, candidate in fn.items():
        if name == 'legacy_64':
            continue
        checks = []
        for layer in LAYERS:
            indices, counts = selected[layer]
            args = query[layer], k[layer], v[layer], histories, lengths
            expected = old(*args, ids=indices, counts=counts, splits=64).clone()
            observed = candidate(*args, ids=indices, counts=counts)
            checks.append(error(observed, expected))
        errors[name] = dict(max_relative_l2=max(x[0] for x in checks), max_absolute=max(x[1] for x in checks))
        if errors[name]['max_relative_l2'] >= .005:
            raise RuntimeError(f'AV accuracy failed for {name}: {errors[name]}')

    def chain(name, with_route):
        for layer in LAYERS:
            q = query[layer]
            if name == 'full':
                full_av(q, k[layer], v[layer], lengths)
            elif with_route:
                ops.route(q, idx[layer])
                fn[name](q, k[layer], v[layer], histories, lengths)
            else:
                ids, counts = selected[layer]
                fn[name](q, k[layer], v[layer], histories, lengths, ids=ids, counts=counts)

    times = {}
    for name in ['full', *fn]:
        times[name] = {'attention_eager': event_times(lambda: chain(name, True), repeats=7, flush=flush)}
        if name != 'full':
            times[name]['av_eager'] = event_times(lambda: chain(name, False), repeats=7, flush=flush)
            times[name]['av_graph_hot_ms'] = do_bench_cudagraph(lambda: chain(name, False), rep=15)
    return dict(times=times, errors=errors, old_tokens_mean=float(torch.cat([v[1] for v in selected.values()]).float().mean()))


def generate(session, seed, initial, opt, setting, steps, engines, pv_mode):
    ops = session.ops
    old = ops.attend
    t, w, s = setting
    methods = ['full', 'legacy', 'tensor_core']
    series = {engine: {method: [] for method in methods} for engine in engines}
    outputs = {}
    for order in (methods, methods[::-1]):
        for name in order:
            session.method = 'full' if name == 'full' else 'sparse'
            ops.attend = (lambda *a, **kw: opt.attend(
                *a, **kw, tile=t, warps=w, splits=s, pv_mode=pv_mode)) if name == 'tensor_core' else old
            token = seed.clone()
            emitted = torch.empty(steps, device='cuda', dtype=torch.long)

            def restore():
                session.restore(initial)
                session.position.fill_(HISTORY)
                token.copy_(seed)

            def step():
                token.copy_(session.step(token).argmax(-1).reshape_as(token))
                session.position.add_(1)

            restore()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(8):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            restore()
            torch.cuda.synchronize()
            operations = {'eager': step}
            if 'graph' in engines:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
                operations['graph'] = graph.replay

            for engine in engines:
                operation = operations[engine]
                def loop():
                    for i in range(steps):
                        operation()
                        emitted[i].copy_(token[0, 0])

                restore()
                _, seconds = wall(loop)
                series[engine][name].append(seconds*1000/steps)
                observed = emitted.tolist()
                previous = outputs.setdefault(name, {}).setdefault(engine, observed)
                if observed != previous:
                    raise RuntimeError(f'non-reproducible output for {name} {engine}')
    ops.attend = old
    first_difference = {engine: next((i+1 for i,(a,b) in enumerate(zip(outputs['legacy'][engine],
                                      outputs['tensor_core'][engine])) if a != b), None) for engine in series}
    return dict(decode_ms_per_forward={engine: {k:statistics.median(v) for k,v in methods.items()}
                                       for engine, methods in series.items()}, repeats=series,
                decode_forwards=steps, first_difference=first_difference,
                engine_exact={name: ids['eager']==ids['graph'] for name, ids in outputs.items()}
                if len(engines) == 2 else None,
                output_tokens=outputs)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assets = ROOT/'staged'
    prompt = assets/'long_source.json'
    if sha(prompt) != LONG_SHA:
        raise ValueError('wrong archived prompt')
    ids = crop(json.loads(prompt.read_bytes())['ids'], HISTORY)
    source = json.loads(args.receipt.read_bytes())
    if source.get('status') != 'complete' or source.get('prompt_tokens') != HISTORY:
        raise ValueError('the fixed-token replay expects a completed 128K receipt')
    emitted = torch.tensor([source['arms']['full']['emitted']], device='cuda')
    kernels = Kernels(ROOT/'build/prism.so')
    print('loading archived weights', flush=True)
    model = load_model(args.model, assets/'model.safetensors.index.json')
    session = Session(model, kernels, 1, HISTORY, CAPACITY-HISTORY)
    seed, prefill_s = wall(lambda: session.prefill(torch.tensor([ids], device='cuda'),0,2048).argmax(-1).reshape(1,1))
    session.extent_to(1)
    session.histories.fill_(HISTORY)
    initial = session.snapshot()
    session.enable_sparse(load_weights(assets/'weights.pt'))
    _, index_s = wall(lambda: session.build_index(0,HISTORY))
    opt = HopperAV(session.ops)
    flush = torch.zeros(64*1024**2, dtype=torch.uint8, device='cuda')
    settings = [tuple(map(int, x.split(','))) for x in args.config]
    result = dict(status='running', gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), triton=triton.__version__,
                  original_receipt_sha=sha(args.receipt), old_prompt_tokens=HISTORY, prefill_s=prefill_s,
                  index_s=index_s, pv_mode=args.pv_mode, probes=[],
                  notes='Same archived greedy tokens replayed via chunked prefill between snapshot queries.')
    offset = 0
    for tail in TAILS:
        print(f'replay tail={tail}', flush=True)
        prefill_tail(session, emitted, offset, tail-1)
        session.position.fill_(HISTORY+tail-1)
        session.capture_queries = {}
        logits = session.step(emitted[:,tail-1:tail])
        query = session.capture_queries
        session.capture_queries = None
        row = probe(session, opt, query, session.histories+tail, settings, flush, args.pv_mode)
        row['tail'] = tail
        row['full_replay_next_token_match'] = bool(logits.argmax()==emitted[0,tail])
        result['probes'].append(row)
        write_json(args.out,result)
        offset = tail
    totals = {n: statistics.mean(row['times'][n]['attention_eager']['ms'] for row in result['probes'])
              for n in result['probes'][0]['times']}
    winner = min((n for n in totals if n.startswith('tc_')), key=totals.get)
    result['mean_attention_eager_ms'] = totals
    result['tensor_core_winner'] = winner
    if args.decode_steps:
        print(f'{winner}: paired short decode', flush=True)
        result['decode'] = generate(session, seed, initial, opt, tuple(map(int, winner.split('_')[1:])),
                                    args.decode_steps, args.engines, args.pv_mode)
    session.close()
    result['status'] = 'complete'
    write_json(args.out, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', action='append', required=True, help='tile,warps,splits')
    parser.add_argument('--decode-steps', type=int, default=128)
    parser.add_argument('--engines', nargs='+', choices=('eager', 'graph'), default=('eager', 'graph'))
    parser.add_argument('--pv-mode', type=int, choices=(0, 1, 2), default=1,
                        help='BF16, scaled FP16, or scaled FP16 with a residual correction')
    run(parser.parse_args())
