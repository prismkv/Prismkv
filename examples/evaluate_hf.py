"""Evaluate fixed tokenized inputs; formatting/scoring belongs to the data protocol."""
import argparse
import torch
from transformers import AutoTokenizer
from prismkv.evaluation import evaluate_jsonl
from prismkv.hf_reference import load_reference_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True, help='JSONL with id and final input_ids')
    parser.add_argument('--output-dir', required=True, help='new directory; never overwritten')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--attention-backend', choices=['sdpa','eager'], default='sdpa')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--prefill-chunk-size', type=int, default=2048)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model = load_reference_model(args.model, dtype=dtype, local_files_only=args.local_files_only,
                                 backend=args.attention_backend).to(args.device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    evaluate_jsonl(model, args.input, args.output_dir, tokenizer=tokenizer,
                   max_new_tokens=args.max_new_tokens, prefill_chunk_size=args.prefill_chunk_size,
                   metadata=dict(attention_backend=args.attention_backend,
                                 dtype=str(dtype), no_latency_claim=True))


if __name__ == '__main__':
    main()
