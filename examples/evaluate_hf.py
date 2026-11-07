"""Evaluate fixed tokenized inputs; formatting/scoring belongs to the data protocol."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from prismkv.evaluation import evaluate_jsonl,sha_file
from prismkv.hf_reference import load_reference_model
from prismkv.weights import load_encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True, help='JSONL with id and final input_ids')
    parser.add_argument('--output-dir', required=True, help='new directory; never overwritten')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode',choices=['full','all_read','sparse'],default='full')
    parser.add_argument('--encoders',help='required for sparse/all_read; model-matched bundle')
    parser.add_argument('--router',choices=['moments','q_activation'],default='moments')
    parser.add_argument('--selected-buckets',type=int,default=16)
    parser.add_argument('--unique-cap',type=float,default=None,help='separate experimental policy')
    parser.add_argument('--update-interval',type=int,default=4096)
    parser.add_argument('--attention-backend', choices=['sdpa','eager'], default='sdpa')
    parser.add_argument('--config-overrides-json',help='explicit Qwen3 RoPE/context overrides; saved config unchanged')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--prefill-chunk-size', type=int, default=2048)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.mode!='full' and not args.encoders:
        parser.error('--encoders is required for sparse/all_read')
    encoders,manifest=None,None
    if args.encoders:
        manifest=json.loads((Path(args.encoders)/'manifest.json').read_text())
        for name,digest in manifest.get('metadata',{}).get('model_fingerprints',{}).items():
            if sha_file(Path(args.model)/name)!=digest:
                raise ValueError(f'checkpoint/bundle fingerprint mismatch: {name}')
        if args.mode!='full':
            encoders,manifest=load_encoders(args.encoders)
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    overrides=json.loads(Path(args.config_overrides_json).read_text()) if args.config_overrides_json else None
    model = load_reference_model(args.model, dtype=dtype, local_files_only=args.local_files_only,
                                 backend=args.attention_backend,config_overrides=overrides).to(args.device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    evaluate_jsonl(model, args.input, args.output_dir, tokenizer=tokenizer,
                   max_new_tokens=args.max_new_tokens, prefill_chunk_size=args.prefill_chunk_size,
                   mode=args.mode,encoders=encoders,router=args.router,
                   selected_buckets=args.selected_buckets,unique_cap=args.unique_cap,
                   update_interval=args.update_interval,
                   metadata=dict(attention_backend=args.attention_backend,
                                 dtype=str(dtype),no_latency_claim=True,
                                 config_overrides=overrides,
                                 encoder_manifest_SHA=sha_file(Path(args.encoders)/'manifest.json') if manifest else None))


if __name__ == '__main__':
    main()
