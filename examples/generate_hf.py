"""Reference generation using an explicit, model-matched encoder bundle."""
import argparse
import json
import torch
from transformers import AutoTokenizer
from prismkv.hf_reference import HFReference,load_reference_model
from prismkv.weights import load_encoders


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--encoders',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prompt',default='Explain why the sky looks blue in two sentences.')
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--all-read',action='store_true',help='dense-equivalence control')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--enable-thinking',action='store_true',help='default follows non-thinking evaluation protocol')
    a=p.parse_args()
    dtype=torch.float32 if a.device=='cpu' else torch.bfloat16
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=a.local_files_only)
    model=load_reference_model(a.model,dtype=dtype,
        local_files_only=a.local_files_only).to(a.device).eval().requires_grad_(False)
    encoders,manifest=load_encoders(a.encoders)
    label=manifest['metadata'].get('model_label')
    print(json.dumps(dict(encoder_model_label=label,note='caller must verify exact checkpoint/tokenizer identity')))
    session=HFReference(model,encoders,mode='all_read' if a.all_read else 'sparse')
    ids=tok.apply_chat_template([dict(role='user',content=a.prompt)],tokenize=True,
        add_generation_prompt=True,return_tensors='pt',enable_thinking=a.enable_thinking).to(a.device)
    try:
        with torch.inference_mode():
            result=session.generate(input_ids=ids,max_new_tokens=a.max_new_tokens,do_sample=False)
        print(tok.decode(result[0,ids.shape[-1]:],skip_special_tokens=True))
        print(json.dumps(dict(group_steps=len(session.records),
            selected_old_tokens=sum(r['unique'] for r in session.records),
            old_history_tokens=sum(r['history'] for r in session.records),
            dense_tail_tokens=sum(r['tail'] for r in session.records),
            note='reference implementation; no latency claim')))
    finally:
        session.close()


if __name__=='__main__':main()
