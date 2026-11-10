"""Entry point for the NVIDIA latency suite. Planning needs only Python."""
import argparse
import json
import os
from pathlib import Path
import time

LENGTHS = (32, 128, 252)
LAYERS = tuple(range(3, 64, 4))
OUTPUT_TOKENS = 4096
CONTEXT_LIMIT = 262144
HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def single_plan(band, output_tokens=OUTPUT_TOKENS):
    history = band * 1024
    if not 0 < history or not 2 <= output_tokens or history + output_tokens > CONTEXT_LIMIT:
        raise ValueError('prompt plus generation must fit the native 256K window')
    return dict(prompt_tokens=history, output_tokens=output_tokens,
                decode_forwards=output_tokens - 1, context_tokens=history + output_tokens,
                final_attention_tail=output_tokens - 1, index_updates=0)


def make_plan():
    return dict(status='planned', sparse_layers=LAYERS, single=[single_plan(k) for k in LENGTHS],
                serving=[dict(history_k=k, concurrency=b) for k, batches in
                         ((32, (2, 4, 8, 16)), (8, (2, 4, 8))) for b in batches],
                methods=['full', 'sparse'])


def gpu(args):
    import torch

    from measure import DeviceMemory, allocator_peak, sync
    from native import Kernels
    from pack import sha as artifact_sha

    if args.out.exists():
        raise FileExistsError(f'preserve existing receipt {args.out}')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError('select exactly one NVIDIA Hopper GPU with CUDA_VISIBLE_DEVICES')
    if args.action != 'check' and not args.allow_other_hopper and 'H200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('this formal run is labeled H200. Use an H200 or explicitly label other Hopper hardware')
    torch.set_num_threads(2)
    torch.manual_seed(20261109)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    kernels = Kernels(args.library)
    identity = dict(device=torch.cuda.get_device_name(0), capability=[9, 0], torch=str(torch.__version__),
                    cuda=torch.version.cuda, pid=os.getpid(), library_sha=artifact_sha(args.library))
    if args.action == 'check':
        from validate import run
        write_json(args.out, dict(**identity, **run(kernels, not args.short)))
        return
    from model import load_model, sha
    import transformers
    import flash_attn
    import importlib.metadata
    import platform
    import subprocess

    identity.update(transformers=transformers.__version__, flash_attn=flash_attn.__version__,
                    driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                    '--format=csv,noheader'], text=True).splitlines(),
                    python=platform.python_version(), source_sha={p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
                    model_config_sha=sha(args.model/'config.json'),
                    model_index_sha=sha(args.model/'model.safetensors.index.json'))
    for name in ('flash-linear-attention', 'causal-conv1d'):
        try:
            identity[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            identity[name] = None
    if args.action == 'single':
        from single import run
    else:
        from serving import run
    if args.route_backend == 'hopper-warp':
        identity['route_library_sha'] = artifact_sha(args.route_library)
    try:
        with DeviceMemory() as memory, torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            model = load_model(args.model, args.assets/'model.safetensors.index.json', args.allow_slow)
            sync()
            identity.update(model_load_s=time.perf_counter()-start, model_load_allocator_peak=allocator_peak())
            result = run(args, model, kernels, memory)
        write_json(args.out, dict(**identity, **result, device_memory=memory.report(),
                                  settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}))
    except RuntimeError as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()
        write_json(args.out, dict(**identity, status='oom' if oom else 'failed',
                                  phase=getattr(locals().get('memory'), 'phase', None), error=str(error)))
        raise


def gpu_options(parser, model=True):
    parser.add_argument('--library', type=Path, default=HERE/'build/prism.so')
    parser.add_argument('--out', type=Path, required=True)
    if model:
        parser.add_argument('--model', type=Path, required=True)
        parser.add_argument('--assets', type=Path, default=HERE/'staged')
        parser.add_argument('--chunk', type=int, default=2048)
        parser.add_argument('--splits', type=int, choices=(32, 64, 128))
        parser.add_argument('--union-mode', type=int, choices=(0, 1, 2, 3), default=2)
        parser.add_argument('--allow-slow', action='store_true', help='diagnostics only')
        parser.add_argument('--allow-other-hopper', action='store_true', help='record actual device label')


def decode_options(parser):
    parser.add_argument('--av-backend', choices=('legacy', 'hopper', 'hopper-compensated'), default='legacy')
    parser.add_argument('--route-backend', choices=('legacy', 'hopper-warp'), default='legacy')
    parser.add_argument('--route-library', type=Path, default=HERE/'build/hopper_top16_v3.so')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--out', type=Path, required=True)
    check = sub.add_parser('check')
    gpu_options(check, model=False)
    check.add_argument('--short', action='store_true')
    single = sub.add_parser('single')
    gpu_options(single)
    single.add_argument('--history', type=int, choices=sorted((*LENGTHS, 64)), required=True)
    single.add_argument('--output-tokens', type=int, default=OUTPUT_TOKENS)
    single.add_argument('--repeats', type=int, default=2)
    single.add_argument('--engine', choices=('eager', 'graph'), default='eager')
    decode_options(single)
    single.add_argument('--profile-repeats', type=int, default=5)
    single.add_argument('--gather-layer', type=int, choices=LAYERS, default=31)
    single.add_argument('--split-diagnostic', action='store_true')
    single.add_argument('--skip-profile', action='store_true')
    serve = sub.add_parser('serve')
    gpu_options(serve)
    decode_options(serve)
    serve.add_argument('--history', type=int, choices=(8, 32, 64, 128, 256), required=True)
    serve.add_argument('--batch', type=int, choices=(2, 4, 8, 16), required=True)
    serve.add_argument('--repeat', type=int, choices=(0, 1), default=0)
    serve.add_argument('--method', choices=('full', 'sparse'), required=True)
    serve.add_argument('--engine', choices=('eager', 'graph'), default='eager')
    serve.add_argument('--output-cap', type=int, choices=(64, 4096), default=64)
    serve.add_argument('--smoke', action='store_true')
    serve.add_argument('--audit-reads', action='store_true', help='use a separate run for instrumented reads')
    args = parser.parse_args()
    if args.action == 'plan':
        write_json(args.out, make_plan())
    else:
        if args.action == 'single':
            single_plan(args.history, args.output_tokens)
            if min(args.repeats, args.profile_repeats) < 1:
                parser.error('repeats must be positive')
        if args.action == 'serve' and args.engine == 'graph' and args.audit_reads:
            parser.error('run read instrumentation in a separate eager service trace')
        if args.action == 'serve' and args.output_cap != 64 and (args.history < 32 or args.smoke):
            parser.error('the long-output trace requires at least 32K history and no smoke truncation')
        if args.action != 'check' and args.chunk < 1:
            parser.error('chunk size must be positive')
        gpu(args)


if __name__ == '__main__':
    main()
