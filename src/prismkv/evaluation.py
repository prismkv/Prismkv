"""Fixed-token quality replay, not native latency or an official benchmark scorer."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import math
import torch
from .hf_reference import HFReference


def token_sequence(values, *, name, vocabulary):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of token IDs")
    if any(type(x) is not int or not 0 <= x < vocabulary for x in values):
        raise ValueError(f"{name} contains an invalid token ID")
    return values


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def vocabulary_size(model):
    config = getattr(model.config, 'text_config', model.config)
    return config.vocab_size


def eos_ids(model, row):
    value = row.get('eos_token_ids', model.generation_config.eos_token_id)
    if value is None:
        return []
    if type(value) is int:
        value = [value]
    if not isinstance(value, list) or any(type(x) is not int or not 0 <= x < vocabulary_size(model) for x in value):
        raise ValueError('invalid EOS token IDs')
    return value


def standard_groups(model):
    groups = sum(module.config.num_key_value_heads for module in model.modules()
                 if type(module).__name__ in ('Qwen3Attention','Qwen3_5Attention'))
    if not groups:
        raise ValueError('no supported standard-attention groups found')
    return groups


def _ppl(nll):
    mean = math.fsum(nll)/len(nll)
    return dict(mean_nll=mean, ppl=math.exp(mean) if mean < 709 else None)


def access_summary(records, *, mode, groups, initial_history, output_tokens):
    """Counts exclude prefill and unmodified linear layers; raw may exceed unique."""
    if mode == 'full':
        events = max(0, output_tokens-1)*groups
        old = events*initial_history
        return dict(source='derived_full_policy', group_steps=events,
            old_history_opportunities=old, selected_old_unique=old, selected_old_raw=old,
            old_unique_read_fraction=1. if events else None,
            indexed_prefix_opportunities=old, selected_indexed_unique=old,
            generated_indexed_unique=0, generated_indexed_raw=0,
            dense_generated_tail=groups*output_tokens*(output_tokens-1)//2,
            scope='standard-attention KV groups only; Full counts are policy-derived')
    expected=max(0, output_tokens-1)*groups
    if len(records)!=expected:
        raise RuntimeError(f'expected {expected} group events, observed {len(records)}')
    old=sum(r['old_unique'] for r in records)
    opportunities=sum(r['initial_history'] for r in records)
    return dict(source='instrumented_reference',group_steps=len(records),
        old_history_opportunities=opportunities,selected_old_unique=old,
        selected_old_raw=sum(r['old_raw'] for r in records),
        old_unique_read_fraction=old/opportunities if opportunities else None,
        indexed_prefix_opportunities=sum(r['history'] for r in records),
        selected_indexed_unique=sum(r['unique'] for r in records),
        generated_indexed_unique=sum(r['generated_indexed_unique'] for r in records),
        generated_indexed_raw=sum(r['generated_indexed_raw'] for r in records),
        dense_generated_tail=sum(r['tail'] for r in records),
        scope='standard-attention KV groups only; initial prompt separated from indexed generation')


@torch.inference_mode()
def _trajectory(model, ids, *, cap, stop, prefill_chunk_size, session=None, forced=None):
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device, dtype=torch.long)
    cache = None
    if session is not None:
        session.begin_request()
    with session.prefill() if session is not None else nullcontext():
        for start in range(0, len(ids), prefill_chunk_size):
            chunk = inputs[:, start:start+prefill_chunk_size]
            positions = torch.arange(start, start+chunk.shape[1], device=device)[None]
            result = model(input_ids=chunk, position_ids=positions, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = result.past_key_values
    logits = result.logits[:, -1].float()
    generated, nll = [], []
    for step in range(cap):
        token = int(logits.argmax(-1)) if forced is None else forced[step]
        generated.append(token)
        nll.append(float(-logits.log_softmax(-1)[0, token]))
        if forced is None and token in stop:
            break
        if step+1 < cap:
            positions = torch.tensor([[len(ids)+step]], device=device)
            result = model(input_ids=torch.tensor([[token]], device=device), position_ids=positions,
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache, logits = result.past_key_values, result.logits[:, -1].float()
    return generated,nll,[] if session is None else list(session.records)


def run_request(model, row, *, mode='full', encoders=None, max_new_tokens=128,
                prefill_chunk_size=2048, router='moments', selected_buckets=16,
                unique_cap=None, update_interval=4096):
    """Free generation first; independent new-cache reference replay afterwards."""
    if model.training:
        raise ValueError('evaluation requires model.eval()')
    if mode not in ('full','all_read','sparse'):
        raise ValueError('mode must be full, all_read or sparse')
    ids=token_sequence(row['input_ids'],name='input_ids',vocabulary=vocabulary_size(model))
    cap=row.get('max_new_tokens',max_new_tokens)
    if type(cap) is not int or cap<=0 or type(prefill_chunk_size) is not int or prefill_chunk_size<=0:
        raise ValueError('positive generation cap and prefill chunk size required')
    reference=row.get('reference_token_ids')
    if reference is not None:
        token_sequence(reference,name='reference_token_ids',vocabulary=vocabulary_size(model))
        if row.get('reference_kind') not in ('full_generated','dataset_target'):
            raise ValueError('reference_kind must label full_generated or dataset_target')
    elif 'reference_kind' in row:
        raise ValueError('reference_kind requires reference_token_ids')
    stop=eos_ids(model,row);groups=standard_groups(model)
    session=None
    try:
        if mode!='full':
            session=HFReference(model,encoders or {},mode=mode,router=router,
                selected_buckets=selected_buckets,unique_cap=unique_cap,
                update_interval=update_interval,detailed_records=True)
        generated,nll,records=_trajectory(model,ids,cap=cap,stop=stop,
            prefill_chunk_size=prefill_chunk_size,session=session)
        values=_ppl(nll)
        result=dict(id=row['id'],mode=mode,input_tokens=len(ids),max_new_tokens=cap,
            input_ids_SHA=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            eos_token_ids=stop,generated_token_ids=generated,generated_tokens=len(generated),
            termination='eos' if generated[-1] in stop else 'cap',
            self_generated_token_nll=nll,self_generated_mean_nll=values['mean_nll'],
            self_generated_ppl=values['ppl'],
            access=access_summary(records,mode=mode,groups=groups,
                                  initial_history=len(ids),output_tokens=len(generated)),
            access_events=records,note='quality/reference replay; no latency or official task score')
        if reference is not None:
            _,teacher_nll,teacher_records=_trajectory(model,ids,cap=len(reference),stop=stop,
                prefill_chunk_size=prefill_chunk_size,session=session,forced=reference)
            result['reference']=dict(kind=row['reference_kind'],token_ids=reference,
                token_nll=teacher_nll,termination='reference_length',**_ppl(teacher_nll),
                access=access_summary(teacher_records,mode=mode,groups=groups,
                    initial_history=len(ids),output_tokens=len(reference)),
                access_events=teacher_records)
        return result
    finally:
        if session is not None:
            session.close()


def full_request(model, row, *, max_new_tokens=128, prefill_chunk_size=2048):
    """Original-backbone baseline, retaining the F0 call interface."""
    return run_request(model,row,max_new_tokens=max_new_tokens,prefill_chunk_size=prefill_chunk_size)


def evaluate_jsonl(model, input_file, output_dir, *, tokenizer=None,
                   max_new_tokens=128, prefill_chunk_size=2048, metadata=None,
                   mode='full',encoders=None,router='moments',selected_buckets=16,
                   unique_cap=None,update_interval=4096):
    """Exclusive output directory; preserve partial results on any failure."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    seen,completed = set(),0
    try:
        config = dict(mode=mode, input_SHA=sha_file(input_file), max_new_tokens=max_new_tokens,
                      prefill_chunk_size=prefill_chunk_size, metadata=metadata or {},
                      router=router,selected_buckets=selected_buckets,unique_cap=unique_cap,
                      update_interval=update_interval,
                      protocol='pretokenized inputs; no prompt formatting/truncation; fresh cache per request')
        _write_json(root/'run.json',config)
        with Path(input_file).open(encoding='utf8') as source, (root/'predictions.jsonl').open('x', encoding='utf8') as stream:
            for line in source:
                row = json.loads(line)
                if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
                    raise ValueError('each request requires a unique nonempty string id')
                seen.add(row['id'])
                result = run_request(model,row,mode=mode,encoders=encoders,
                    max_new_tokens=max_new_tokens,prefill_chunk_size=prefill_chunk_size,
                    router=router,selected_buckets=selected_buckets,
                    unique_cap=unique_cap,update_interval=update_interval)
                if tokenizer is not None:
                    result['prediction'] = tokenizer.decode(result['generated_token_ids'], skip_special_tokens=True)
                stream.write(json.dumps(result, allow_nan=False)+'\n')
                stream.flush()
                completed+=1
        if not seen:
            raise ValueError('input file has no requests')
        done = dict(complete=True, requests=completed, predictions_SHA=sha_file(root/'predictions.jsonl'),
                    run_SHA=sha_file(root/'run.json'))
        _write_json(root/'DONE.json',done)
        return done
    except BaseException as error:
        _write_json(root/'FAILED.json',dict(error=repr(error),completed_requests=completed))
        raise
