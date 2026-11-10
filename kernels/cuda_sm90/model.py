"""Pinned hybrid model runner with native GQA and preallocated slot-major KV."""
import hashlib
import types

import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from transformers import AttentionInterface, AutoConfig, DynamicCache, __version__ as hf_version
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from native import DecodeOps, Index, IndexBuilder, WIDTH, cuda_empty
from run import LAYERS

CONFIG_SHA = '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'
SAE_SHA = '8420f99119b8166f8917dc060503280b94f0faca07dc82dbf591e27bc356d8a8'
LINEAR = tuple(i for i in range(64) if i not in LAYERS)
STATES = ('conv_states', 'recurrent_states')


def sha(path):
    with path.open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
        return digest.hexdigest()


def load_weights(path):
    if sha(path) != SAE_SHA:
        raise ValueError('SAE checkpoint hash differs from the paper archive')
    rows = torch.load(path, map_location='cpu', weights_only=False)['weights']
    if set(rows) != {(layer, g) for layer in LAYERS for g in range(4)}:
        raise ValueError('expected exactly 64 SAE dictionaries')
    return {key: (row['W_enc'].squeeze(0).cuda().float().contiguous(),
                  row['b_enc'].squeeze(0).cuda().float().contiguous(), float(row['scale']['k']))
            for key, row in rows.items()}


def load_model(path, staged_index, allow_slow=False):
    if hf_version != '5.8.0':
        raise RuntimeError(f'expected Transformers 5.8.0, got {hf_version}')
    if sha(path / 'config.json') != CONFIG_SHA or sha(path / 'model.safetensors.index.json') != sha(staged_index):
        raise ValueError('model config or shard index differs from the archived checkpoint')
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    cfg._attn_implementation = 'eager'
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, config=cfg, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, output_loading_info=True, attn_implementation='eager')
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if info.get(key):
            raise RuntimeError(f'checkpoint load reported {key}: {info[key][:6]}')
    text = model.model.language_model
    if (text.config.num_hidden_layers, text.config.num_attention_heads,
            text.config.num_key_value_heads, text.config.head_dim) != (64, 24, 4, WIDTH):
        raise ValueError('unexpected hybrid model geometry')
    if tuple(i for i, value in enumerate(text.config.layer_types) if value == 'full_attention') != LAYERS:
        raise ValueError('unexpected standard Attention layer IDs')
    first = text.layers[LINEAR[0]].linear_attn
    if not allow_slow and (first.causal_conv1d_fn is None or first.recurrent_gated_delta_rule.__name__.startswith('torch_')
                           or first.chunk_gated_delta_rule.__name__.startswith('torch_')):
        raise RuntimeError('install NVIDIA flash-linear-attention and causal-conv1d before formal timing')
    return model.eval().requires_grad_(False)


def dispatch(module, q, k, v, attention_mask, **kwargs):
    if attention_mask is not None:
        raise ValueError('the timing runner accepts unpadded requests only')
    return module._h200_session.attention(module.layer_idx, q, k, v), None


class Session:
    def __init__(self, model, kernels, slots, history, horizon, weights=None, splits=None, union_mode=2):
        self.model, self.text = model, model.model.language_model
        self.history, self.capacity, self.slots = history, history + horizon, slots
        self.kernels, self.splits, self.union_mode = kernels, splits, union_mode
        self.ops = self.builder = None
        if weights is not None:
            self.enable_sparse(weights)
        self.indices, self.states = {}, {}
        self.keys = {layer: cuda_empty(slots, self.capacity, 4, WIDTH, dtype=torch.bfloat16) for layer in LAYERS}
        self.values = {layer: torch.empty_like(key) for layer, key in self.keys.items()}
        for layer in LAYERS:
            self.keys[layer][:, 0].zero_()
            self.values[layer][:, 0].zero_()
        self.position = torch.zeros(slots, dtype=torch.int32, device='cuda')
        self.histories = torch.zeros_like(self.position)
        self.capture_queries = None
        self.decode, self.extent, self.method = False, 0, 'full'
        self.reads = None
        self.decode_cache = self.cache()
        self.original = self.text.config._attn_implementation
        AttentionInterface.register('prismkv_h200_native', dispatch)
        for layer in LAYERS:
            self.text.layers[layer].self_attn._h200_session = self
        self.text.config._attn_implementation = 'prismkv_h200_native'

    def close(self):
        self.text.config._attn_implementation = self.original
        for layer in LAYERS:
            del self.text.layers[layer].self_attn._h200_session

    def cache(self):
        cache = DynamicCache(config=self.text.config)
        for layer in LAYERS:
            target = cache.layers[layer]
            target.update = types.MethodType(self.append, target)
            target.h200_layer = layer
        return cache

    def append(self, target, k, v, *unused, **unused_kw):
        layer = target.h200_layer
        keys, values = self.keys[layer], self.values[layer]
        if self.decode:
            keys, values = keys[:self.extent], values[:self.extent]
            self.kernels.call('batch_append', k.contiguous(), v.contiguous(), keys, values,
                              self.position[:self.extent], self.extent, self.capacity)
            return keys, values
        end = self.start + k.shape[-2]
        keys, values = keys[self.slot:self.slot + 1], values[self.slot:self.slot + 1]
        keys[:, self.start:end].copy_(k.transpose(1, 2))
        values[:, self.start:end].copy_(v.transpose(1, 2))
        return keys[:, :end], values[:, :end]

    def attention(self, layer, q, k, v):
        q = q.transpose(1, 2).contiguous()
        if not self.decode:
            return flash_attn_func(q, k, v, dropout_p=0., softmax_scale=1/16, causal=True)
        if self.capture_queries is not None:
            self.capture_queries[layer] = q.clone()
        if self.method == 'full':
            return flash_attn_with_kvcache(q, k, v, cache_seqlens=self.flash_lengths,
                                          softmax_scale=1/16, causal=False)
        index = self.indices[layer]
        self.ops.route(q, index)
        if self.reads is not None:
            counts = self.ops.counts[:4*self.extent].view(-1, 4).sum(-1)
            self.reads[:self.extent].add_(counts * (self.lengths > 0))
        return self.ops.attend(q, k, v, self.histories[:self.extent], self.lengths)

    def forward(self, ids, positions, cache):
        # Same text-only RoPE and decoder order as Transformers 5.8.0. Native
        # Flash handles the causal mask, so no dense [query, history] mask is allocated.
        hidden = self.text.embed_tokens(ids)
        embeddings = self.text.rotary_emb(hidden, positions.unsqueeze(0).expand(3, -1, -1))
        for layer in self.text.layers:
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=None,
                           position_ids=positions, past_key_values=cache, use_cache=True)
        return self.model.lm_head(self.text.norm(hidden[:, -1:]))[:, 0]

    def prefill(self, ids, slot, chunk):
        if not 0 <= slot < self.slots or not 0 < ids.shape[-1] <= self.history:
            raise ValueError('slot or prompt length exceeds reserved capacity')
        self.decode, self.slot = False, slot
        cache = self.cache()
        for self.start in range(0, ids.shape[-1], chunk):
            panel = ids[:, self.start:self.start + chunk]
            pos = torch.arange(self.start, self.start + panel.shape[-1], device='cuda')[None]
            logits = self.forward(panel, pos, cache)
        for layer in LINEAR:
            source = cache.layers[layer]
            target = self.decode_cache.layers[layer]
            if layer not in self.states:
                target.lazy_initialization(conv_states=source.conv_states.expand(self.slots, -1, -1))
                target.lazy_initialization(recurrent_states=source.recurrent_states.expand(self.slots, *source.recurrent_states.shape[1:]))
                self.states[layer] = {name: getattr(target, name) for name in STATES}
            for name, state in self.states[layer].items():
                state[slot:slot + 1].copy_(getattr(source, name))
            target.has_previous_state = True
        return logits

    def build_index(self, slot, length):
        if self.builder is None:
            raise ValueError('sparse index requires an SAE checkpoint')
        for layer in LAYERS:
            if layer not in self.indices:
                self.indices[layer] = Index(self.slots, self.history)
            self.indices[layer].set_slot(slot, self.keys[layer][slot, :length], layer, self.builder)

    def enable_sparse(self, weights):
        self.builder = IndexBuilder(self.kernels, weights)
        self.ops = DecodeOps(self.kernels, self.slots, self.history, self.capacity, self.splits, self.union_mode)

    def prefill_witness(self, ids):
        # Test masking, RoPE and chunked recurrent continuation against the
        # unmodified eager forward on exactly the same small input.
        self.text.config._attn_implementation = self.original
        try:
            reference = self.text(input_ids=ids, use_cache=True)
            expected = [self.model.lm_head(reference.last_hidden_state[:, -1:]).float()[:, 0]]
            forced = []
            for _ in range(3):
                token = expected[-1].argmax(-1, keepdim=True)
                forced.append(token)
                reference = self.text(input_ids=token, use_cache=True, past_key_values=reference.past_key_values)
                expected.append(self.model.lm_head(reference.last_hidden_state).float()[:, 0])
        finally:
            self.text.config._attn_implementation = 'prismkv_h200_native'
        observed = [self.prefill(ids, 0, 32).float()]
        self.extent_to(1)
        self.position.fill_(ids.shape[-1])
        for token in forced:
            observed.append(self.step(token).float())
            self.position.add_(1)
        rows = []
        for actual, reference in zip(observed, expected):
            relative = ((actual-reference).norm()/reference.norm()).item()
            ref_log = reference.log_softmax(-1)
            kl = (ref_log.exp()*(ref_log-actual.log_softmax(-1))).sum().item()
            if relative >= .03 or kl >= .02:
                raise RuntimeError(f'Flash/eager model mismatch: relative L2={relative}, KL={kl}')
            rows.append(dict(relative_l2=relative, kl=kl,
                             top1_exact=actual.argmax().item()==reference.argmax().item()))
        return dict(prefill=rows[0], full_decode=rows[1:])

    def extent_to(self, n):
        if n != self.extent:
            for layer, storage in self.states.items():
                for name, state in storage.items():
                    setattr(self.decode_cache.layers[layer], name, state[:n])
            self.extent = n

    def step(self, ids):
        self.decode = True
        pos = self.position[:self.extent]
        self.lengths = pos + 1
        self.flash_lengths = self.lengths.clamp_min(1)
        return self.forward(ids, pos.clamp_min(0).long()[:, None], self.decode_cache)

    def snapshot(self):
        return {layer: {name: state.clone() for name, state in values.items()}
                for layer, values in self.states.items()}

    def restore(self, snapshot):
        for layer, values in snapshot.items():
            for name, value in values.items():
                self.states[layer][name].copy_(value)

    def inventory(self):
        def nbytes(tensor):
            return tensor.numel() * tensor.element_size()
        index = {name: sum(item.inventory()[name] for item in self.indices.values())
                 for name in ('bits', 'matrix', 'logn', 'counts')}
        return dict(kv_bytes=sum(nbytes(x) for x in [*self.keys.values(), *self.values.values()]),
                    recurrent_bytes=sum(nbytes(x) for row in self.states.values() for x in row.values()),
                    index_bytes=sum(index.values()), index_breakdown=index,
                    route_workspace_bytes=sum(nbytes(x) for x in vars(self.ops).values() if torch.is_tensor(x))
                    if self.ops else 0,
                    sae_weights_bytes=sum(nbytes(t) for row in self.builder.weights.values() for t in row[:2])
                    if self.builder else 0, index_build_workspace_bytes=nbytes(self.builder.moment_scratch)
                    if self.builder else 0)
