"""Auditable reference operations; Python routing is NOT the timed native backend.

All q/k inputs here are the actual post-position-encoding attention vectors.
Original KV vectors remain single-copy; lists contain original token positions.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


def positive_global_topk(a: torch.Tensor, count: int):
    """Global positive activation budget, deterministic flattened-index tie break."""
    if a.ndim != 2 or count < 0:
        raise ValueError("expected a matrix and nonnegative activation budget")
    flat = a.flatten()
    count = min(count, flat.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=a.device)
    threshold = flat.topk(count, sorted=False).values.min()
    strict = ((flat > threshold) & (flat > 0)).nonzero().flatten()
    if threshold > 0:
        ties = (flat == threshold).nonzero().flatten()[: count - len(strict)]
        return torch.cat((strict, ties))
    return strict


class SharedSAE(nn.Module):
    def __init__(self, head_dim=128, features=3000, key_mean=8, query_topk=16):
        super().__init__()
        if not 1 <= key_mean <= features or not 1 <= query_topk <= features:
            raise ValueError("invalid sparsity")
        self.head_dim, self.features = head_dim, features
        self.key_mean, self.query_topk = key_mean, query_topk
        self.W_enc = nn.Parameter(torch.empty(features, head_dim))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        self.b_enc = nn.Parameter(torch.zeros(features))
        # Preserve the historical 8B parameter initialization. Unit decoder
        # directions are formed in the loss, not by changing parameter norms.
        self.W_dec = nn.Parameter(self.W_enc.detach().T.clone())
        self.b_dec = nn.Parameter(torch.zeros(head_dim))
        self.register_buffer("seen", torch.tensor(0))
        self.register_buffer("last", torch.zeros(features, dtype=torch.long))
        self.register_buffer("counts", torch.zeros(features, dtype=torch.long))

    def dense(self, normalized):
        return F.linear(normalized.float(), self.W_enc.float(), self.b_enc.float()).relu()

    def encode(self, normalized, nq):
        """Queries use per-row TopK; keys share one batch-wide activation budget."""
        pre = F.linear(normalized, self.W_enc, self.b_enc)
        a = pre.relu()
        qi = a[:nq].argsort(dim=-1, descending=True, stable=True)[:, : self.query_topk]
        q = torch.zeros_like(a[:nq]).scatter(-1, qi, a[:nq].gather(1, qi))
        ka = a[nq:]
        ids = positive_global_topk(ka, len(ka) * self.key_mean)
        k = torch.zeros_like(ka).flatten().scatter(0, ids, ka.flatten()[ids]).reshape_as(ka)
        return torch.cat((q, k)), pre


@dataclass
class FeatureIndex:
    history: int
    center: torch.Tensor
    counts: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    lists: list

    @property
    def memberships(self):
        return sum(len(x) for x in self.lists)


@torch.no_grad()
def summarize(keys, lists, center=None):
    """Population diagonal moments of ORIGINAL keys, not SAE reconstructions."""
    n, d = keys.shape
    center = keys.float().mean(0) if center is None else center
    counts = torch.tensor([len(x) for x in lists], device=keys.device, dtype=torch.long)
    mean = keys.new_zeros((len(lists), d), dtype=torch.float32)
    var = torch.zeros_like(mean)
    for b, ids in enumerate(lists):
        if len(ids):
            x = keys[ids].float()
            mean[b] = x.mean(0)
            var[b] = (x - mean[b]).square().mean(0)
    return FeatureIndex(n, center, counts, mean, var, lists)


@torch.no_grad()
def build_index(keys, sae, key_scale, *, center=None):
    if keys.ndim != 2 or len(keys) == 0 or key_scale <= 0:
        raise ValueError("nonempty [history,head_dim] keys and positive scale required")
    if keys.shape[-1] != sae.head_dim:
        raise ValueError("SAE and attention head dimensions differ")
    center = keys.float().mean(0) if center is None else center.to(keys.device)
    activation = sae.dense((keys.float() - center) / key_scale)
    flat = positive_global_topk(activation, len(keys) * sae.key_mean)
    token_ids, feature_ids = flat // sae.features, flat % sae.features
    lists = [token_ids[feature_ids == b].sort().values for b in range(sae.features)]
    return summarize(keys, lists, center)


@torch.no_grad()
def bucket_scores(q, index, *, router="moments", sae=None, query_scale=None,
                  query_topk=16, terms=("count", "mean", "variance")):
    """One GQA group's Q heads -> shared bucket scores and eligible candidates.

    moments: submitted count/mean/variance estimator; TopK shared buckets.
    q_activation: new ablation, per-Q TopK then positive weighted aggregation.
    """
    q = q.float()
    n, d = len(index.counts), q.shape[-1]
    live = index.counts > 0
    if router == "moments":
        if not terms or set(terms) - {"count", "mean", "variance"}:
            raise ValueError("specify exact estimator components")
        logits = q.new_zeros((len(q), n))
        if "count" in terms:
            logits += index.counts.clamp_min(1).float().log()
        if "mean" in terms:
            logits += q @ index.mean.T / math.sqrt(d)
        if "variance" in terms:
            logits += q.square() @ index.variance.T / (2 * d)
        logits[:, ~live] = -torch.inf
        score = logits.softmax(-1).mean(0) if live.any() else logits.new_zeros(n)
        order = score.argsort(descending=True, stable=True)
        return score, order[live[order]][:query_topk]
    if router != "q_activation" or sae is None or query_scale is None or query_scale <= 0:
        raise ValueError("q_activation requires SAE and saved positive query scale")
    a = sae.dense(q / query_scale)
    ids = a.argsort(dim=-1, descending=True, stable=True)[:, :query_topk]
    val = a.gather(1, ids)
    val = val / val.sum(-1, keepdim=True).clamp_min(1e-20)
    score = q.new_zeros(n).scatter_add_(0, ids.flatten(), (val / len(q)).flatten())
    order = score.argsort(descending=True, stable=True)
    return score, order[(score[order] > 0) & live[order]]


@torch.no_grad()
def select_buckets(q, index, *, unique_cap=None, **kwargs):
    """Whole buckets; skip overflow; no oracle clipping or hidden fallback.

    unique_cap=None is the submitted native bucket-count rule. A fractional cap
    is a distinct, explicitly budget-capped experimental selection policy.
    """
    if unique_cap is not None and not 0 <= unique_cap <= 1:
        raise ValueError("unique_cap must be a fraction in [0,1]")
    scores, order = bucket_scores(q, index, **kwargs)
    chosen = torch.zeros(index.history, device=q.device, dtype=torch.bool)
    limit = index.history if unique_cap is None else math.floor(index.history * unique_cap)
    features, raw, count = [], 0, 0
    for b in order.tolist():
        ids = index.lists[b]
        new = int((~chosen[ids]).sum())
        if count + new > limit:
            continue
        chosen[ids] = True
        count += new
        raw += len(ids)
        features.append(b)
    ids = chosen.nonzero().flatten()
    assert len(ids) <= limit
    return ids, dict(features=features, raw=raw, unique=len(ids), limit=limit,
                     candidate_buckets=len(order), scores=scores)


@torch.no_grad()
def attend(q, keys, values, old_ids, history):
    """Reference exact softmax over selected ORIGINAL KV plus complete dense tail."""
    if keys.shape != values.shape or not 0 <= history <= len(keys):
        raise ValueError("invalid KV/history")
    if len(old_ids) and (old_ids.min() < 0 or old_ids.max() >= history):
        raise ValueError("old_ids must address old history only")
    if len(old_ids.unique()) != len(old_ids):
        raise ValueError("deduplicate history before attention")
    ids = torch.cat((old_ids, torch.arange(history, len(keys), device=keys.device)))
    if len(ids) == 0:
        return q.new_zeros(q.shape)
    logits = q.float() @ keys[ids].float().T / math.sqrt(q.shape[-1])
    return (logits.softmax(-1) @ values[ids].float()).to(q.dtype)


@torch.no_grad()
def append_index(index, new_keys, sae, key_scale):
    """Index only an appended block, preserving old memberships and initial center.

    BatchTopK runs over the NEW block; this intentionally differs from rebuilding
    the global index over old+new history. Merge population moments algebraically.
    """
    nxt = build_index(new_keys, sae, key_scale, center=index.center)
    a, b = index.counts.float(), nxt.counts.float()
    total = a + b
    den = total.clamp_min(1)
    mean = (a[:, None] * index.mean + b[:, None] * nxt.mean) / den[:, None]
    var = (a[:, None] * index.variance + b[:, None] * nxt.variance) / den[:, None]
    var += ((a * b) / den.square())[:, None] * (index.mean - nxt.mean).square()
    lists = [torch.cat((old, new + index.history)) for old, new in zip(index.lists, nxt.lists)]
    return FeatureIndex(index.history + len(new_keys), index.center, total.long(), mean, var, lists)
