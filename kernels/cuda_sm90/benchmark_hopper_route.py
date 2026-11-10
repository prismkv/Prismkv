"""Compare Top16 routes and paired Attention paths on archived H200 cases."""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from triton.testing import do_bench_cudagraph

from benchmark_hopper_av import ROOT, TAILS, prefill_tail
from hopper_av import HopperAV
from hopper_route import HopperTop16
from measure import event_times, wall
from model import Session, load_model, load_weights, sha
from native import DecodeOps, FEATURES, Kernels
from run import LAYERS, OUTPUT_TOKENS, write_json
from single import full_av, layout_probe
from stage_cases import LONG_SHA, crop


def baseline(ops, scores, index, groups):
    ops.kernels.call('fused_norm_top', scores, index.logn, index.counts,
                     ops.features, ops.selected, ops.raw, groups)


def score_route(ops, candidates, q, index, mode):
    groups = 4 * q.shape[0]
    ops.kernels.call('pack', q, ops.packed, groups)
    gemm(ops, index, groups)
    select(ops, candidates, index, groups, mode)


def gemm(ops, index, groups):
    torch.bmm(ops.packed[:groups], index.matrix.view(-1, 512, FEATURES)[:groups], out=ops.scores[:groups])


def select(ops, candidates, index, groups, mode):
    if mode:
        candidates[mode](ops.scores, index, ops, groups)
    else:
        baseline(ops, ops.scores, index, groups)


def selection(ops):
    return [x.clone() for x in (ops.features, ops.selected, ops.raw)]


def assert_selection(ops, expected, label):
    for field, actual, reference in zip(('features', 'selected', 'raw'), selection(ops), expected):
        if not torch.equal(actual, reference):
            raise RuntimeError(f'{label} changed {field}: {(actual != reference).sum().item()} elements')


def check_cases(kernels, library, modes):
    ops = DecodeOps(kernels, 2, 128, 160)
    groups = ops.scores.shape[0]
    candidates = {mode: HopperTop16(ops, library, mode) for mode in modes}
    index = SimpleNamespace(counts=torch.zeros(groups, FEATURES, device='cuda', dtype=torch.int32),
                            logn=torch.zeros(groups, FEATURES, device='cuda'))
    cases = []
    for name in ('empty', 'fewer_than_16', 'all_equal', 'near_tie', 'masked_random', 'large_range'):
        index.counts.fill_(1)
        ops.scores.zero_()
        if name == 'empty':
            index.counts.zero_()
        elif name == 'fewer_than_16':
            index.counts[:, 13:] = 0
        elif name == 'near_tie':
            ops.scores.normal_(std=1e-6)
        elif name in ('masked_random', 'large_range'):
            index.counts.random_(0, 150)
            index.counts[:, ::3] = 0
            ops.scores.normal_(std=100 if name == 'large_range' else 2)
        index.logn.copy_(index.counts.float().clamp_min(1).log())
        select(ops, candidates, index, groups, 0)
        expected = selection(ops)
        for mode in candidates:
            select(ops, candidates, index, groups, mode)
            assert_selection(ops, expected, f'{name} mode={mode}')
        cases.append(name)
    return cases


def probe(session, candidates, queries, lengths):
    ops, indices = session.ops, session.indices
    frozen = {}
    groups = 4
    positions = torch.arange(ops.history, device='cuda')[None, :]
    for layer in LAYERS:
        index = indices[layer]
        q = queries[layer]
        score_route(ops, candidates, q, index, 0)
        expected = selection(ops)
        ops.merge(1, index)
        ids, counts = ops.ids.clone(), ops.counts.clone()
        for mode in candidates:
            score_route(ops, candidates, q, index, mode)
            assert_selection(ops, expected, f'layer={layer} mode={mode}')
            ops.merge(1, index)
            if not torch.equal(ops.counts, counts) or not torch.equal(
                    torch.where(positions < counts[:, None], ops.ids, 0),
                    torch.where(positions < counts[:, None], ids, 0)):
                raise RuntimeError(f'layer={layer} mode={mode} changed token IDs')
        frozen[layer] = ops.scores.clone(), expected

    def chain(phase, mode=0):
        for layer in LAYERS:
            q, idx = queries[layer], indices[layer]
            scores, (features, active, _) = frozen[layer]
            if phase == 'pack':
                ops.kernels.call('pack', q, ops.packed, groups)
            elif phase == 'gemm':
                gemm(ops, idx, groups)
            elif phase == 'top16':
                if mode:
                    candidates[mode](scores, idx, ops, groups)
                else:
                    baseline(ops, scores, idx, groups)
            elif phase == 'union':
                ops.merge(1, idx, features, active)
            else:
                score_route(ops, candidates, q, idx, mode)
                ops.merge(1, idx)
                if phase == 'attention':
                    ops.attend(q, session.keys[layer], session.values[layer], session.histories, lengths)

    def bench(phase, mode=0):
        return do_bench_cudagraph(lambda: chain(phase, mode), rep=25)

    return dict(fixed={phase: bench(phase) for phase in ('pack', 'gemm', 'union')},
                modes={mode: {phase: bench(phase, mode) for phase in ('top16', 'route', 'attention')}
                       for mode in (0, *candidates)})


def attention_probe(session, candidates, hopper, queries, lengths, flush, reverse, diagnostics=False):
    ops = session.ops
    names = ('full', 'legacy', 'tensor_core', 'combined')
    positions = torch.arange(ops.history, device='cuda')[None, :]
    frozen = {}

    def forward(layer, name):
        q, k, v = queries[layer], session.keys[layer], session.values[layer]
        if name == 'full':
            return full_av(q, k, v, lengths)
        index = session.indices[layer]
        if name == 'combined':
            score_route(ops, candidates, q, index, 5)
            ops.merge(1, index)
        else:
            ops.route(q, index)
        args = q, k, v, session.histories, lengths
        if name == 'legacy':
            return ops.attend(*args)
        return hopper.attend(*args, tile=64, warps=4, splits=128, pv_mode=1)

    for layer in LAYERS:
        expected = forward(layer, 'tensor_core').clone()
        expected_selection = selection(ops)
        counts = ops.counts.clone()
        expected_ids = torch.where(positions < counts[:, None], ops.ids, 0)
        actual = forward(layer, 'combined')
        assert_selection(ops, expected_selection, f'layer={layer} combined')
        if not torch.equal(ops.counts, counts) or not torch.equal(
                torch.where(positions < counts[:, None], ops.ids, 0), expected_ids):
            raise RuntimeError(f'layer={layer} combined changed token IDs')
        if not torch.equal(actual, expected):
            raise RuntimeError(f'layer={layer} combined changed Tensor Core output')
        if diagnostics:
            frozen[layer] = ops.ids.clone(), counts, expected_selection
            for mode in (0, 1, 2, 3):
                ops.merge(1, session.indices[layer], *expected_selection[:2], mode)
                if not torch.equal(ops.counts, counts) or not torch.equal(
                        torch.where(positions < counts[:, None], ops.ids, 0), expected_ids):
                    raise RuntimeError(f'layer={layer} union mode {mode} changed token IDs')

    def chain(name):
        for layer in LAYERS:
            forward(layer, name)

    times = {}
    for name in (names[::-1] if reverse else names):
        times[name] = dict(eager=event_times(lambda: chain(name), repeats=9, flush=flush),
                           graph_hot_ms=do_bench_cudagraph(lambda: chain(name), rep=35))
    result = dict(times=times, exact_tensor_core_output_layers=len(LAYERS))
    if diagnostics:
        def stage_chain(stage, option):
            for layer in LAYERS:
                q, index = queries[layer], session.indices[layer]
                ids, counts, (features, active, _) = frozen[layer]
                if stage == 'score':
                    score_route(ops, candidates, q, index, 5)
                elif stage == 'union':
                    ops.merge(1, index, features, active, option)
                else:
                    hopper.attend(q, session.keys[layer], session.values[layer], session.histories, lengths,
                                  ids=ids, counts=counts, tile=64, warps=4, splits=option)

        def bench(stage, option=None):
            return do_bench_cudagraph(lambda: stage_chain(stage, option), rep=25)

        gather_layer = 31
        tail = int((lengths - session.histories).item())
        ops.route(queries[gather_layer], session.indices[gather_layer])
        layout = layout_probe(session, queries[gather_layer], session.keys[gather_layer], session.values[gather_layer],
                              session.history, tail, 7, flush, graph=True,
                              attend=lambda *a, **kw: hopper.attend(*a, **kw, tile=64, warps=4, splits=128))
        result['diagnostics'] = dict(score_graph_ms=bench('score'),
            union_graph_ms={mode: bench('union', mode) for mode in (0, 1, 2, 3)},
            av_graph_ms={splits: bench('av', splits) for splits in (32, 64, 128)},
            old_unique_fraction=sum(int(x[1].sum()) for x in frozen.values()) / (len(LAYERS)*4*session.history),
            gather_layer=gather_layer, layout=layout)
    return result


def decode(session, seed, initial, candidates, mode, steps):
    ops = session.ops
    original_score, original_av = ops.score, ops.attend
    hopper = HopperAV(ops)
    arms = ('full', 'old', 'route', 'tensor_core', 'combined')
    times, outputs = {arm: [] for arm in arms}, {}
    for order in (arms, arms[::-1]):
        for arm in order:
            session.method = 'full' if arm == 'full' else 'sparse'
            ops.score = (lambda q, index: score_route(ops, candidates, q, index, mode)) if arm in (
                'route', 'combined') else original_score
            ops.attend = (lambda *a: hopper.attend(*a, tile=64, splits=128)) if arm in (
                'tensor_core', 'combined') else original_av
            token = torch.empty_like(seed)
            emitted = torch.empty(steps, dtype=torch.long, device='cuda')

            def restore():
                session.restore(initial)
                session.position.fill_(session.history)
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
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()

            def loop():
                for i in range(steps):
                    graph.replay()
                    emitted[i].copy_(token[0, 0])

            restore()
            _, seconds = wall(loop)
            times[arm].append(seconds * 1000 / steps)
            values = emitted.tolist()
            if outputs.setdefault(arm, values) != values:
                raise RuntimeError(f'{arm} generation changed across repeats')
            print(f'{arm}: {times[arm][-1]:.4f} ms/token', flush=True)
    ops.score, ops.attend = original_score, original_av
    exact = {arm: outputs[arm] == outputs[reference] for arm, reference in (
        ('route', 'old'), ('combined', 'tensor_core'))}
    return dict(steps=steps, ms_per_forward={arm: statistics.median(v) for arm, v in times.items()},
                repeats_ms=times, exact_route_tokens=exact, output_tokens=outputs)


@torch.inference_mode()
def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(20261110)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    kernels = Kernels(ROOT/'build/prism.so')
    result = dict(status='running', gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                  backend_sha=sha(args.library), original_cases=check_cases(kernels, args.library, args.modes))
    write_json(args.out, result)
    if not args.check_only:
        assets = ROOT/'staged'
        prompt = assets/'long_source.json'
        if sha(prompt) != LONG_SHA:
            raise ValueError('wrong archived prompt')
        history = args.history * 1024
        ids = crop(json.loads(prompt.read_bytes())['ids'], history)
        source = json.loads(args.receipt.read_bytes())
        if (source.get('status') != 'complete' or source.get('prompt_tokens') != history
                or source.get('output_tokens') != OUTPUT_TOKENS):
            raise ValueError('expected a completed receipt for the selected history and a 4096-token continuation')
        emitted = torch.tensor([source['arms']['full']['emitted']], device='cuda')
        if emitted.shape[1] != OUTPUT_TOKENS:
            raise ValueError('the Full receipt has an incomplete trajectory')
        model = load_model(args.model, assets/'model.safetensors.index.json')
        session = Session(model, kernels, 1, history, OUTPUT_TOKENS)
        seed, prefill_s = wall(lambda: session.prefill(torch.tensor([ids], device='cuda'), 0, 2048).argmax(-1).reshape(1,1))
        session.extent_to(1)
        session.histories.fill_(history)
        initial = session.snapshot()
        session.enable_sparse(load_weights(assets/'weights.pt'))
        _, index_s = wall(lambda: session.build_index(0, history))
        result.update(prompt_tokens=history, output_tokens=OUTPUT_TOKENS, prefill_s=prefill_s, index_s=index_s,
                      receipt_sha=sha(args.receipt), probes=[])
        candidates = {mode: HopperTop16(session.ops, args.library, mode) for mode in args.modes}
        if args.attention_only:
            hopper = HopperAV(session.ops)
            flush = torch.zeros(64 * 1024**2, dtype=torch.uint8, device='cuda')
            result['protocol'] = 'attention_only_paired_16_layers'
        offset = 0
        for i, tail in enumerate(TAILS):
            print(f'replay tail={tail}', flush=True)
            prefill_tail(session, emitted, offset, tail-1)
            session.position.fill_(history+tail-1)
            session.capture_queries = {}
            logits = session.step(emitted[:, tail-1:tail])
            queries = session.capture_queries
            session.capture_queries = None
            if args.attention_only:
                row = attention_probe(session, candidates, hopper, queries, session.histories+tail, flush,
                                      i % 2, diagnostics=args.stage_diagnostics)
                row['full_replay_next_token_match'] = bool(logits.argmax() == emitted[0, tail])
            else:
                row = probe(session, candidates, queries, session.histories+tail)
            result['probes'].append(dict(tail=tail, **row))
            write_json(args.out, result)
            offset = tail
        if args.attention_only:
            rows = [p['times'] for p in result['probes']]
            result['mean_attention_ms'] = {
                name: dict(eager=statistics.mean(row[name]['eager']['ms'] for row in rows),
                           graph_hot=statistics.mean(row[name]['graph_hot_ms'] for row in rows))
                for name in rows[0]}
        else:
            averages = {mode: statistics.mean(p['modes'][mode]['route'] for p in result['probes'])
                        for mode in (0, *candidates)}
            winner = min(candidates, key=averages.get)
            result.update(route_mean_graph_ms=averages, candidate=winner)
            result['decode'] = decode(session, seed, initial, candidates, winner, args.steps)
        session.close()
    result['status'] = 'complete'
    write_json(args.out, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--history', type=int, choices=(32, 64, 128, 252), default=128)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--modes', nargs='+', type=int, choices=(1, 2, 3, 4, 5), default=(1, 2, 3, 4, 5))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--attention-only', action='store_true', help='time Full and all combined Attention paths')
    parser.add_argument('--stage-diagnostics', action='store_true', help='also time stages and fixed-ID KV layouts')
    args = parser.parse_args()
    if not args.check_only and (args.model is None or args.receipt is None or not 8 <= args.steps < OUTPUT_TOKENS):
        parser.error('model, receipt and a decode horizon from 8 to 4095 are required')
    if args.attention_only and (args.check_only or 5 not in args.modes):
        parser.error('paired Attention timing requires route mode 5 and a model')
    if args.stage_diagnostics and not args.attention_only:
        parser.error('stage diagnostics require paired Attention timing')
    run(args)
