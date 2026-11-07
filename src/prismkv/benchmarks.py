"""Named historical input protocols and wrappers around caller-owned official metrics.

No dataset, official scorer implementation or implicit download is bundled here.
Input-format parity and model/arithmetic parity are different checks.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from .evaluation import sha_file

LONG_BENCH_TASKS=('narrativeqa','qasper','multifieldqa_en','hotpotqa','2wikimqa','musique',
    'gov_report','qmsum','multi_news','trec','triviaqa','samsum','passage_count',
    'passage_retrieval_en','lcc','repobench-p')
RAW_FIVE={'trec','triviaqa','samsum','lcc','repobench-p'}
RULER_TASKS=tuple(f'niah_single_{i}' for i in range(1,4))+tuple(f'niah_multikey_{i}' for i in range(1,4))+( 'niah_multivalue','niah_multiquery','vt','cwe','fwe','qa_1','qa_2')
ANSWER_ONLY='\nOutput only the requested final answer. Do not repeat the question or the answer introduction. Do not explain your reasoning.'


def _ids_sha(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def _chat(tokenizer,text):
    return tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False)


def longbench_input(raw, *, task, index, tokenizer, templates, caps, protocol, eos_token_ids):
    if task not in LONG_BENCH_TASKS or protocol not in ('qwen8_paper','qwen27_paper'):
        raise ValueError('unsupported LongBench task/protocol')
    prompt=templates[task].format(**raw);cap=caps[task]
    if protocol=='qwen27_paper':
        encoded=tokenizer.encode(prompt,add_special_tokens=True);original=len(encoded)
        truncated=original>65536
        if truncated:
            prompt=tokenizer.decode(encoded[:32768],skip_special_tokens=True)+tokenizer.decode(encoded[-32768:],skip_special_tokens=True)
        text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=True)
    else:
        encoded=tokenizer.encode(prompt,add_special_tokens=False);original=len(encoded)
        raw_task=task in RAW_FIVE
        encode=(lambda text:tokenizer(text,add_special_tokens=True)['input_ids']) if raw_task else (lambda text:_chat(tokenizer,text))
        ids=encode(prompt);limit=32768-cap;take=limit-64;truncated=len(ids)>limit
        while len(ids)>limit:
            if take<=0:
                raise ValueError('tokenizer wrapper cannot fit the explicit context budget')
            text=tokenizer.decode(encoded[:take//2],skip_special_tokens=True)+tokenizer.decode(encoded[-(take-take//2):],skip_special_tokens=True)
            ids=encode(text);take-=max(32,len(ids)-limit)
    stops=list(eos_token_ids);banned=[]
    if task=='samsum':
        stops=list(dict.fromkeys(stops+[tokenizer.encode('\n',add_special_tokens=False)[-1]]))
        banned=list(stops)
    return dict(id=f'{task}_{index:04d}',benchmark='longbench',task=task,protocol=protocol,
        input_ids=ids,max_new_tokens=cap,eos_token_ids=stops,first_token_banned_ids=banned,
        dense_decode_steps=1,required_model_overrides={'rope_scaling':None} if protocol=='qwen8_paper' else {},
        answers=raw['answers'],all_classes=raw.get('all_classes',[]),length=raw.get('length'),
        original_prompt_tokens=original,truncated=truncated,input_ids_SHA=_ids_sha(ids))


def ruler_base_task(task):
    if task not in RULER_TASKS:
        raise ValueError('unsupported RULER task')
    if task.startswith('niah_'):
        return 'niah'
    names={'vt':'variable_tracking','cwe':'common_words_extraction',
           'fwe':'freq_words_extraction','qa_1':'qa','qa_2':'qa'}
    if task not in names:
        raise ValueError('unsupported RULER task')
    return names[task]


def ruler_input(raw, *, task, index, tokenizer, protocol, eos_token_ids,
                context_length, max_new_tokens, prompt_field, answers_field, answer_only=False):
    base=ruler_base_task(task)
    if protocol not in ('qwen8_paper','qwen27_paper') or context_length<=0 or max_new_tokens<=0:
        raise ValueError('invalid RULER protocol/length/cap')
    prompt=raw[prompt_field];answers=raw[answers_field]
    if not isinstance(prompt,str) or not isinstance(answers,list) or not answers or not all(isinstance(x,str) for x in answers):
        raise ValueError('RULER requires a prompt string and nonempty string references')
    text=tokenizer.apply_chat_template([dict(role='user',content=prompt+(ANSWER_ONLY if answer_only else ''))],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    extended=protocol=='qwen8_paper' and context_length>32768
    overrides=(dict(rope_scaling=dict(rope_type='yarn',factor=4.,original_max_position_embeddings=32768),
                    max_position_embeddings=131072) if extended else dict(rope_scaling=None)) if protocol=='qwen8_paper' else {}
    limit=(131072 if extended else 32768) if protocol=='qwen8_paper' else 262144
    if len(ids)+max_new_tokens>limit:
        raise ValueError('RULER prompt/output exceeds this explicit protocol; no truncation')
    return dict(id=f'{context_length}_{task}_{index:04d}',benchmark='ruler',task=task,base_task=base,
        context_length=context_length,protocol=protocol,input_ids=ids,max_new_tokens=max_new_tokens,
        eos_token_ids=list(eos_token_ids),dense_decode_steps=1,required_model_overrides=overrides,
        answer_only=answer_only,truncated=False,answers=answers,input_ids_SHA=_ids_sha(ids))


def longbench_root(path):
    root=Path(path)
    if not (root/'eval.py').exists() and (root/'LongBench/eval.py').exists():
        root=root/'LongBench'
    if not (root/'eval.py').is_file() or not (root/'metrics.py').is_file():
        raise ValueError('point to the official classic LongBench checkout')
    return root


def load_official_longbench(path):
    root=longbench_root(path)
    spec=importlib.util.spec_from_file_location('prismkv_official_lb_metrics',root/'metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=vars(module).copy()
    tree=ast.parse((root/'eval.py').read_text())
    nodes=[node for node in tree.body if
        isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dataset2metric' for t in node.targets)
        or isinstance(node,ast.FunctionDef) and node.name=='scorer']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/'eval.py'),'exec'),namespace)
    return namespace,{name:sha_file(root/name) for name in ['eval.py','metrics.py']}


def load_official_ruler(path):
    file=Path(path)/'scripts/eval/synthetic/constants.py'
    spec=importlib.util.spec_from_file_location('prismkv_official_ruler_metrics',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,{'scripts/eval/synthetic/constants.py':sha_file(file)}


def score_records(inputs, predictions, *, official_root, benchmark):
    """Require exactly paired IDs/inputs; return official rounded cells plus coverage."""
    rows={r['id']:r for r in inputs};preds={r['id']:r for r in predictions}
    if len(rows)!=len(inputs) or len(preds)!=len(predictions) or set(rows)!=set(preds) or not rows:
        raise ValueError('scoring requires exactly paired unique request IDs')
    groups={}
    for key,row in rows.items():
        pred=preds[key]
        if row['benchmark']!=benchmark or pred['input_ids_SHA']!=_ids_sha(row['input_ids']):
            raise ValueError('benchmark/input identity mismatch')
        if not isinstance(pred.get('prediction'),str):
            raise ValueError('prediction text must be decoded with the matching tokenizer')
        cell=(row['task'],row.get('context_length'))
        groups.setdefault(cell,[]).append((row,pred['prediction']))
    cells=[]
    if benchmark=='longbench':
        official,hashes=load_official_longbench(official_root)
        for (task,band),pairs in sorted(groups.items()):
            classes=pairs[0][0].get('all_classes',[])
            if any(r.get('all_classes',[])!=classes for r,_ in pairs):
                raise ValueError('LongBench class lists differ within a task')
            score=official['scorer'](task,[p for _,p in pairs],[r['answers'] for r,_ in pairs],classes)
            cells.append(dict(task=task,context_length=band,cases=len(pairs),official_score=score))
    elif benchmark=='ruler':
        official,hashes=load_official_ruler(official_root)
        for (task,band),pairs in sorted(groups.items()):
            base=ruler_base_task(task)
            if any(r.get('base_task',base)!=base for r,_ in pairs):
                raise ValueError('RULER task mapping mismatch')
            score=official.TASKS[base]['metric_fn']([p for _,p in pairs],[r['answers'] for r,_ in pairs])
            cells.append(dict(task=task,context_length=band,cases=len(pairs),official_score=score))
    else:
        raise ValueError('supported benchmarks: longbench/ruler')
    return dict(benchmark=benchmark,requests=len(rows),cells=cells,
        macro_of_official_rounded_cells=statistics.mean(r['official_score'] for r in cells),
        official_source_SHA=hashes,
        scope='official scorer on supplied paired subset; coverage listed explicitly, not automatically a full benchmark')
