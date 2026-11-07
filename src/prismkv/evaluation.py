"""Fixed-token quality replay, not native latency or an official benchmark scorer."""
from pathlib import Path
import hashlib
import json
import math
import torch


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


@torch.inference_mode()
def full_request(model, row, *, max_new_tokens=128, prefill_chunk_size=2048):
    """Greedy baseline with the original backbone/cache, no PrismKV dispatch."""
    ids = token_sequence(row['input_ids'], name='input_ids', vocabulary=vocabulary_size(model))
    cap = row.get('max_new_tokens', max_new_tokens)
    if type(cap) is not int or cap <= 0 or type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
        raise ValueError('positive generation cap and prefill chunk size required')
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device, dtype=torch.long)
    cache = None
    for start in range(0, len(ids), prefill_chunk_size):
        chunk = inputs[:, start:start+prefill_chunk_size]
        positions = torch.arange(start, start+chunk.shape[1], device=device)[None]
        result = model(input_ids=chunk, position_ids=positions, past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
    logits = result.logits[:, -1].float()
    generated, nll = [], []
    stop = eos_ids(model, row)
    for step in range(cap):
        token = int(logits.argmax(-1))
        generated.append(token)
        nll.append(float(-logits.log_softmax(-1)[0, token]))
        if token in stop:
            break
        if step+1 < cap:
            positions = torch.tensor([[len(ids)+step]], device=device)
            result = model(input_ids=torch.tensor([[token]], device=device), position_ids=positions,
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache, logits = result.past_key_values, result.logits[:, -1].float()
    mean = math.fsum(nll)/len(nll)
    return dict(id=row['id'], mode='full', input_tokens=len(ids), max_new_tokens=cap,
                eos_token_ids=stop, generated_token_ids=generated, generated_tokens=len(generated),
                termination='eos' if generated[-1] in stop else 'cap',
                self_generated_token_nll=nll, self_generated_mean_nll=mean,
                self_generated_ppl=math.exp(mean) if mean < 709 else None,
                note='original-backbone quality reference; no latency or official task score')


def evaluate_jsonl(model, input_file, output_dir, *, tokenizer=None,
                   max_new_tokens=128, prefill_chunk_size=2048, metadata=None):
    """Exclusive output directory; preserve partial results on any failure."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    config = dict(mode='full', input_SHA=sha_file(input_file), max_new_tokens=max_new_tokens,
                  prefill_chunk_size=prefill_chunk_size, metadata=metadata or {},
                  protocol='pretokenized inputs; no prompt formatting/truncation; fresh cache per request')
    (root/'run.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    seen = set()
    try:
        with Path(input_file).open(encoding='utf8') as source, (root/'predictions.jsonl').open('x', encoding='utf8') as stream:
            for line in source:
                row = json.loads(line)
                if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
                    raise ValueError('each request requires a unique nonempty string id')
                seen.add(row['id'])
                result = full_request(model, row, max_new_tokens=max_new_tokens,
                                      prefill_chunk_size=prefill_chunk_size)
                if tokenizer is not None:
                    result['prediction'] = tokenizer.decode(result['generated_token_ids'], skip_special_tokens=True)
                stream.write(json.dumps(result, allow_nan=False)+'\n')
                stream.flush()
        if not seen:
            raise ValueError('input file has no requests')
        done = dict(complete=True, requests=len(seen), predictions_SHA=sha_file(root/'predictions.jsonl'),
                    run_SHA=sha_file(root/'run.json'))
        (root/'DONE.json').write_text(json.dumps(done, indent=2), encoding='utf8')
        return done
    except BaseException as error:
        (root/'FAILED.json').write_text(json.dumps(dict(error=repr(error), completed_requests=len(seen)), indent=2), encoding='utf8')
        raise
