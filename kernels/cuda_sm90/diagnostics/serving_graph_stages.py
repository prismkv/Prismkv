"""Profile the two-slot Attention path on the first two real Full requests."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from triton.testing import do_bench_cudagraph

from model import Session, load_model, load_weights, sha
from native import Kernels
from run import LAYERS, write_json
from serving import load_requests
from single import full_av


def profile(session, queries, lengths):
    ops = session.ops
    frozen = {}
    for layer in LAYERS:
        ops.route(queries[layer], session.indices[layer])
        frozen[layer] = tuple(x.clone() for x in (ops.features, ops.selected, ops.ids, ops.counts))

    def chain(stage):
        for layer in LAYERS:
            q = queries[layer]
            k, v = session.keys[layer], session.values[layer]
            index = session.indices[layer]
            features, active, ids, counts = frozen[layer]
            if stage == 'full_attention':
                full_av(q, k, v, lengths)
            elif stage == 'score':
                ops.score(q, index)
            elif stage == 'union':
                ops.merge(2, index, features, active)
            elif stage == 'route':
                ops.route(q, index)
            elif stage == 'indexed_attention':
                ops.attend(q, k, v, session.histories, lengths, ids=ids, counts=counts)
            else:
                ops.route(q, index)
                ops.attend(q, k, v, session.histories, lengths)

    names = ('full_attention', 'score', 'union', 'route', 'indexed_attention', 'combined_attention')
    ms = {name: do_bench_cudagraph(lambda: chain(name), rep=35) for name in names}
    reads = sum(int(row[3].sum()) for row in frozen.values()) / (len(LAYERS)*2*4*session.history)
    return dict(ms_per_batch_step=ms, old_unique_fraction=reads)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assets = ROOT/'staged'
    source = json.loads(args.receipt.read_text())
    if source['status'] != 'complete' or source['method'] != 'full' or source['settings']['engine'] != 'graph':
        raise ValueError('expected the completed Full Graph service receipt')
    rows = load_requests(assets/'serving.json.gz', 256, 2, 0)[:2]
    records = sorted(source['request_stats'], key=lambda r: r['i'])[:2]
    if [(r['input_sha'], r['cap']) for r in rows] != [(r['input_sha'], r['cap']) for r in records]:
        raise ValueError('the two requests differ from the timing receipt')
    history = rows[0]['history']
    tokens = torch.tensor([r['tokens'][:17] for r in records], device='cuda')
    model = load_model(args.model, assets/'model.safetensors.index.json')
    kernels = Kernels(ROOT/'build/prism.so')
    session = Session(model, kernels, 2, history, 64, load_weights(assets/'weights.pt'))
    try:
        session.ops.enable_hopper_route(args.route_library)
        session.ops.enable_hopper_av(pv_mode=1)
        for slot, row in enumerate(rows):
            print(f'prefill slot {slot}', flush=True)
            logits = session.prefill(torch.tensor([row['ids']], device='cuda'), slot, 2048)
            if logits.argmax().item() != records[slot]['tokens'][0]:
                raise RuntimeError('prefill output differs from the timing receipt')
            session.build_index(slot, history)
        session.extent_to(2)
        session.histories.fill_(history)
        result = dict(status='running', prompt_tokens=history, batch=2, receipt_sha=sha(args.receipt),
                      route_library_sha=sha(args.route_library), probes=[])
        for step in range(1, 17):
            sample = step in (1, 8, 16)
            session.position.fill_(history + step - 1)
            session.capture_queries = {} if sample else None
            logits = session.step(tokens[:, step-1:step].contiguous())
            queries = session.capture_queries
            session.capture_queries = None
            if sample:
                print(f'profile tail {step}', flush=True)
                result['probes'].append(dict(tail=step, full_next_tokens_match=torch.equal(logits.argmax(-1), tokens[:, step]),
                                             **profile(session, queries, session.lengths)))
                write_json(args.out, result)
        result['mean_ms_per_batch_step'] = {
            name: statistics.mean(p['ms_per_batch_step'][name] for p in result['probes'])
            for name in result['probes'][0]['ms_per_batch_step']}
        result['status'] = 'complete'
        write_json(args.out, result)
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--route-library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
