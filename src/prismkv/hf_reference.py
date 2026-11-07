"""Inspectable Hugging Face integration for unpadded, batch-one causal inference.

This is the correctness/quality path, NOT the measured native performance path.
It uses the model's own prefill backend and gathers selected original KV during
decode. Supported attention modules: Qwen3Attention / Qwen3_5Attention, with
all of their KV groups supplied explicitly. No silent dense-head fallback.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import importlib
import torch
from .core import build_index, append_index, select_buckets, attend


@dataclass
class Encoder:
    sae: object
    key_scale: float
    query_scale: float


def load_reference_model(model_path, *, dtype=torch.float32, local_files_only=True,
                         backend="sdpa"):
    """Load an explicit reference backend even if a checkpoint saved FA2.

    Some checkpoints serialize a private attention-backend flag. Under the
    pinned Transformers version it can survive a from_pretrained keyword.
    Change only the in-memory config; never rewrite the checkpoint or weights.
    This helper is a reference loader, not the native performance backend.
    """
    if backend not in ("sdpa", "eager"):
        raise ValueError("reference backend must be sdpa or eager")
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    config._attn_implementation = backend
    return AutoModelForCausalLM.from_pretrained(model_path, config=config,
        torch_dtype=dtype, attn_implementation=backend, local_files_only=local_files_only)


def _dispatch(module, q, k, v, attention_mask, **kwargs):
    session = getattr(module, "_prismkv_session", None)
    if session is None:
        raise RuntimeError("PrismKV backend selected without an attached session")
    return session.forward(module, q, k, v, attention_mask, **kwargs)


class HFReference:
    """A stateful, all-group adapter; one session per model, one request at a time.

    Set mode='all_read' only for a dense-equivalence control. The main method
    uses mode='sparse', router='moments', selected_buckets=16, unique_cap=None.
    Q-feature routing and strict caps are explicitly supplementary policies.
    Call begin_request() when starting a new KV cache; close() restores backends.
    """
    def __init__(self, model, encoders, *, mode="sparse", router="moments",
                 selected_buckets=16, unique_cap=None, update_interval=4096):
        if mode not in ("sparse", "all_read"):
            raise ValueError("mode must be sparse or all_read")
        if router not in ("moments", "q_activation") or selected_buckets <= 0:
            raise ValueError("unsupported router/bucket count")
        if unique_cap is not None and not 0 <= unique_cap <= 1:
            raise ValueError("unique cap must be a fraction")
        if update_interval <= 0:
            raise ValueError("positive update interval required")
        self.model, self.encoders = model, encoders
        self.mode, self.router = mode, router
        self.selected_buckets, self.unique_cap = selected_buckets, unique_cap
        self.update_interval, self.force_prefill = update_interval, False
        self.modules, self.original, self.configs = {}, {}, {}
        expected = set()
        for module in model.modules():
            if type(module).__name__ not in ("Qwen3Attention", "Qwen3_5Attention"):
                continue
            if hasattr(module, "_prismkv_session"):
                raise ValueError("model already has a reference session")
            layer = module.layer_idx
            if layer in self.modules:
                raise ValueError("duplicate attention-layer IDs")
            cfg = module.config
            expected.update((layer, g) for g in range(cfg.num_key_value_heads))
            self.modules[layer] = module
            implementation = cfg._attn_implementation
            if implementation not in ("sdpa", "eager"):
                raise ValueError("reference adapter is validated for SDPA/eager prefill only")
            # Decorators may put forward.__globals__ in a deprecation utility.
            namespace = vars(importlib.import_module(type(module).__module__))
            registry = namespace["ALL_ATTENTION_FUNCTIONS"]
            self.original[layer] = (namespace["eager_attention_forward"] if implementation == "eager"
                                    else registry[implementation])
            self.configs[id(cfg)] = (cfg, implementation)
        if not self.modules or set(encoders) != expected:
            raise ValueError(f"provide exactly all standard-attention KV encoders; missing={expected-set(encoders)}, extra={set(encoders)-expected}")
        # Register only after all non-mutating validation succeeds.
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask, eager_mask
        AttentionInterface.register("prismkv_reference", _dispatch)
        AttentionMaskInterface.register("prismkv_reference", sdpa_mask)
        # Eager attention does not infer causality from a missing mask. Using
        # SDPA's mask elision here would silently turn full prefill bidirectional.
        AttentionInterface.register("prismkv_reference_eager", _dispatch)
        AttentionMaskInterface.register("prismkv_reference_eager", eager_mask)
        for module in self.modules.values():
            module._prismkv_session = self
        for cfg, old in self.configs.values():
            cfg._attn_implementation = ("prismkv_reference_eager" if old == "eager"
                                         else "prismkv_reference")
        self.begin_request()

    def begin_request(self):
        self.prefix_lengths, self.indices, self.records = {}, {}, []

    @contextmanager
    def prefill(self):
        """Explicitly mark chunked prefill, including one-token chunks."""
        previous = self.force_prefill
        self.force_prefill = True
        try:
            yield self
        finally:
            self.force_prefill = previous

    def close(self):
        for module in self.modules.values():
            if getattr(module, "_prismkv_session", None) is self:
                del module._prismkv_session
        for cfg, implementation in self.configs.values():
            cfg._attn_implementation = implementation
        self.indices.clear()

    @torch.no_grad()
    def forward(self, module, q, k, v, attention_mask, **kwargs):
        if q.shape[0] != 1 or k.shape[0] != 1:
            raise NotImplementedError("reference adapter supports unpadded batch-one requests")
        if q.shape[-2] < 1 or q.shape[-1] != k.shape[-1] or k.shape != v.shape:
            raise ValueError("invalid QKV shapes")
        d = q.shape[-1]
        if not math.isclose(kwargs.get("scaling", 1/math.sqrt(d)), 1/math.sqrt(d)):
            raise NotImplementedError("nonstandard attention scale")
        if kwargs.get("sliding_window") is not None or kwargs.get("softcap") is not None:
            raise NotImplementedError("sliding-window/soft-capped attention is outside this adapter")
        if kwargs.get("dropout", 0) != 0:
            raise ValueError("inference only: disable dropout")
        layer, total, steps = module.layer_idx, k.shape[-2], q.shape[-2]
        # On an unpadded causal request, the last row can see every actual KV.
        if attention_mask is not None:
            if attention_mask.ndim != 4:
                raise NotImplementedError("padding/arbitrary attention masks are unsupported")
            last = attention_mask[..., -1, :total]
            invalid = ~last if last.dtype == torch.bool else last != 0
            if bool(invalid.any()):
                raise NotImplementedError("padding/arbitrary attention masks are unsupported")
        is_prefill = self.force_prefill or layer not in self.prefix_lengths or steps > 1
        if is_prefill:
            self.prefix_lengths[layer] = total
            for unit in [u for u in self.indices if u[0] == layer]:
                del self.indices[unit]
            # A custom registry name may not have a framework mask constructor.
            # For cached chunks supply the correctly offset causal mask ourselves.
            if attention_mask is None and 1 < steps < total:
                positions = torch.arange(total-steps, total, device=q.device)
                visible = torch.arange(total, device=q.device)[None] <= positions[:, None]
                attention_mask = torch.zeros((1, 1, steps, total), device=q.device, dtype=q.dtype)
                attention_mask.masked_fill_(~visible[None, None], -torch.inf)
            return self.original[layer](module, q, k, v, attention_mask, **kwargs)
        if steps != 1:
            raise NotImplementedError("decode reference is one token per forward")
        groups, queries = k.shape[1], q.shape[1]
        if queries % groups:
            raise ValueError("Q heads must form equal-sized KV groups")
        per_group = queries // groups
        answer = torch.empty((1, 1, queries, d), dtype=q.dtype, device=q.device)
        for group in range(groups):
            unit = (layer, group)
            bundle = self.encoders[unit]
            Q, K, V = q[0, group*per_group:(group+1)*per_group, 0], k[0, group], v[0, group]
            if self.mode == "all_read":
                history = self.prefix_lengths[layer]
                ids = torch.arange(history, device=q.device)
                meta = dict(raw=history, unique=history, features=[])
            else:
                if unit not in self.indices:
                    h = self.prefix_lengths[layer]
                    bundle.sae.to(q.device).eval().requires_grad_(False)
                    self.indices[unit] = build_index(K[:h], bundle.sae, bundle.key_scale)
                index = self.indices[unit]
                # Index complete old tail blocks, retaining the CURRENT token in
                # dense tail. Old memberships and original center stay unchanged.
                while total - index.history > self.update_interval:
                    new = K[index.history:index.history+self.update_interval]
                    index = append_index(index, new, bundle.sae, bundle.key_scale)
                self.indices[unit] = index
                history = index.history
                ids, meta = select_buckets(Q, index, router=self.router,
                    sae=bundle.sae, query_scale=bundle.query_scale,
                    query_topk=self.selected_buckets, unique_cap=self.unique_cap)
            answer[0, 0, group*per_group:(group+1)*per_group] = attend(Q, K, V, ids, history)
            self.records.append(dict(layer=layer, group=group, history=history,
                unique=len(ids), raw=meta['raw'], buckets=len(meta['features']), tail=total-history))
        return answer, None

    def generate(self, *args, **kwargs):
        """Reset for a new model.generate call/cache. Existing cache reuse is manual."""
        if kwargs.get("past_key_values") is not None:
            raise ValueError("use manual model calls and session state for an existing cache")
        self.begin_request()
        return self.model.generate(*args, **kwargs)
