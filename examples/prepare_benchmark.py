"""Convert local benchmark rows with an explicitly named paper input protocol."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from prismkv.benchmarks import longbench_input,ruler_input,longbench_root
from prismkv.evaluation import sha_file,_write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',choices=['longbench','ruler'],required=True)
    parser.add_argument('--protocol',choices=['qwen8_paper','qwen27_paper'],required=True)
    parser.add_argument('--model',required=True,help='local matching checkpoint/tokenizer')
    parser.add_argument('--input',required=True,help='one raw task JSONL; datasets not downloaded')
    parser.add_argument('--task',required=True)
    parser.add_argument('--output-dir',required=True,help='new directory only')
    parser.add_argument('--official-root',help='official classic LongBench checkout for templates/caps')
    parser.add_argument('--context-length',type=int,help='RULER length-band label; never truncates input')
    parser.add_argument('--max-new-tokens',type=int,help='explicit RULER output cap')
    parser.add_argument('--prompt-field',help='explicit RULER field, e.g. input or prompt')
    parser.add_argument('--answers-field',help='explicit RULER field, e.g. outputs or answers')
    parser.add_argument('--answer-only',action='store_true',help='historical VT/CWE variant, explicitly labeled')
    parser.add_argument('--limit',type=int,help='explicit diagnostic subset, not a complete task')
    args=parser.parse_args()
    if args.benchmark=='longbench' and not args.official_root:
        parser.error('--official-root required for LongBench')
    if args.benchmark=='ruler' and any(getattr(args,k) is None for k in ['context_length','max_new_tokens','prompt_field','answers_field']):
        parser.error('RULER requires context length, output cap and explicit prompt/answer fields')
    if args.limit is not None and args.limit<=0:
        parser.error('--limit must be positive')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    generation=json.loads((Path(args.model)/'generation_config.json').read_text())
    stops=generation['eos_token_id'];stops=[stops] if isinstance(stops,int) else stops
    templates=caps=None;official_files={}
    if args.benchmark=='longbench':
        official=longbench_root(args.official_root)
        templates=json.loads((official/'config/dataset2prompt.json').read_text())
        caps=json.loads((official/'config/dataset2maxlen.json').read_text())
        official_files={str(p.relative_to(official)):sha_file(p) for p in [official/'config/dataset2prompt.json',official/'config/dataset2maxlen.json']}
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=False)
    count=0;overrides=None
    try:
        with Path(args.input).open(encoding='utf8') as source,(root/'rows.jsonl').open('x',encoding='utf8') as output:
            for line in source:
                if not line.strip():continue
                if args.limit is not None and count>=args.limit:break
                raw=json.loads(line)
                shared=dict(task=args.task,index=count,tokenizer=tokenizer,protocol=args.protocol,eos_token_ids=stops)
                if args.benchmark=='longbench':
                    row=longbench_input(raw,templates=templates,caps=caps,**shared)
                else:
                    row=ruler_input(raw,context_length=args.context_length,max_new_tokens=args.max_new_tokens,
                        prompt_field=args.prompt_field,answers_field=args.answers_field,answer_only=args.answer_only,**shared)
                if overrides is not None and overrides!=row['required_model_overrides']:
                    raise ValueError('split different effective model configurations into separate files')
                overrides=row['required_model_overrides'];output.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
        if not count:raise ValueError('no input rows')
        _write_json(root/'model_overrides.json',overrides)
        _write_json(root/'protocol.json',dict(benchmark=args.benchmark,protocol=args.protocol,task=args.task,
            requests=count,diagnostic_limit=args.limit,source_SHA=sha_file(args.input),official_files=official_files,
            model_files={name:sha_file(Path(args.model)/name) for name in ['config.json','tokenizer_config.json','generation_config.json']},
            context_length=args.context_length,answer_only=args.answer_only,
            notice='Named historical input protocol, not untouched official defaults or full native-backend reproduction'))
        _write_json(root/'DONE.json',dict(complete=True,requests=count,hashes={name:sha_file(root/name) for name in ['rows.jsonl','model_overrides.json','protocol.json']}))
    except BaseException as error:
        _write_json(root/'FAILED.json',dict(error=repr(error),prepared_rows=count));raise


if __name__=='__main__':main()
