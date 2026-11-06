"""Single-dictionary reference for the published SAE loss components.

Mining requires full-attention training trajectories, not downstream test labels.
The caller supplies normalized reconstruction examples and separately mined
ordinary/evidence pairs. Dataset generation and checkpoint histories must be
reported separately; this module does not turn a new run into an old checkpoint.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class TrainingBatch:
    reconstruction: torch.Tensor
    reconstruction_queries: int
    pairs: torch.Tensor | None = None
    pair_queries: int = 32
    evidence_pairs: torch.Tensor | None = None


def cosine_margin(z, nq, margin=0.1):
    if len(z) != 18 * nq:
        raise ValueError("pairs are [Q, positive K, 16 negatives per Q]")
    q = F.normalize(z[:nq], dim=-1)
    pos = F.normalize(z[nq:2 * nq], dim=-1)
    neg = F.normalize(z[2 * nq:].reshape(nq, 16, -1), dim=-1)
    return (margin + (q[:, None] * neg).sum(-1).max(-1).values
            - (q * pos).sum(-1)).relu().mean()


def reconstruction_loss(sae, x, nq, *, update=True):
    z, pre = sae.encode(x, nq)
    decoder = F.normalize(sae.W_dec, dim=0, eps=1e-8)
    residual = x - (z @ decoder.T + sae.b_dec)
    rec = residual.square().mean() / x.square().mean().clamp_min(1e-8)
    # Inactivity is measured in examples observed, as in the released recipe.
    dead = (sae.seen - sae.last) >= 4096
    activation = pre.relu().masked_fill(~dead[None], 0)
    values, ids = activation.topk(min(64, sae.features), dim=-1)
    aux_z = torch.zeros_like(pre).scatter(-1, ids, values)
    target = residual.detach()
    aux = ((aux_z @ decoder.T - target).square().mean()
           / target.square().mean().clamp_min(1e-8)) * dead.any()
    if update:
        with torch.no_grad():
            sae.seen += len(x)
            count = (z > 0).sum(0).long()
            sae.counts += count
            sae.last[count > 0] = sae.seen
    return rec, aux, z, pre


def objective(sae, batch, ema, step, *, match=True, balance_weight=0.5):
    """Reconstruction + matching + anti-dead/32 + utilization regularizer.

    Balance is F*dot(EMA use, mean softmax(K preactivations/0.5))-1;
    it can be negative. Its absolute value is not a uniformity certificate.
    EMA and hard-selection membership are detached; the softmax factor carries
    its gradient. Matching's softplus warm-start decays over 1000 steps.
    """
    if ema.shape != (sae.features,):
        raise ValueError("EMA must have one value per feature")
    x, nq = batch.reconstruction, batch.reconstruction_queries
    if not 0 < nq < len(x):
        raise ValueError("reconstruction needs both queries and keys")
    rec, dead, z, pre = reconstruction_loss(sae, x, nq)
    frequency = (z[nq:] > 0).float().mean(0)
    with torch.no_grad():
        ema.mul_(0.99).add_(frequency / frequency.sum().clamp_min(1e-8), alpha=0.01)
    balance = sae.features * (ema.detach() * (pre[nq:] / 0.5).softmax(-1).mean(0)).sum() - 1
    ordinary = rec * 0
    evidence = rec * 0
    weight = 0.25 * max(0.0, 1 - step / 1000)
    if match:
        if batch.pairs is None:
            raise ValueError("matching enabled without mined pairs")
        for pairs, name in [(batch.pairs, "ordinary"), (batch.evidence_pairs, "evidence")]:
            if pairs is None:
                continue
            code, pair_pre = sae.encode(pairs, batch.pair_queries)
            loss = cosine_margin(code, batch.pair_queries)
            loss = loss + weight * cosine_margin(F.softplus(pair_pre, beta=5), batch.pair_queries)
            if name == "ordinary":
                ordinary = loss
            else:
                evidence = loss
    total = rec + dead / 32 + ordinary + evidence + balance_weight * balance
    metrics = dict(loss=total, reconstruction=rec, dead=dead, ordinary=ordinary,
                   evidence=evidence, balance=balance,
                   mean_key_activations=(z[nq:] > 0).sum(-1).float().mean())
    return total, {k: v.detach() for k, v in metrics.items()}


@torch.no_grad()
def mine_pairs(q, k, *, queries=32, seed=0):
    """Teacher-weighted Top2% positive, 16 negatives outside Top10%.

    Raw post-RoPE Q/K enter here, BEFORE SAE centering/scaling. Returns indices,
    allowing the caller to apply saved Q/K scales and context-specific K center.
    This is offline training work and is never the deployment router.
    """
    import math
    if len(k) < 2:
        raise ValueError("need at least two historical keys")
    gen = torch.Generator(device=q.device).manual_seed(seed)
    qi = torch.randint(len(q), (queries,), generator=gen, device=q.device)
    logits = q[qi].float() @ k.float().T / math.sqrt(q.shape[-1])
    order = logits.argsort(dim=-1, descending=True, stable=True)
    npos = max(1, math.ceil(0.02 * len(k)))
    positive_ids = order[:, :npos]
    probability = logits.gather(1, positive_ids).softmax(-1)
    pi = torch.multinomial(probability, 1, generator=gen)
    positive = positive_ids.gather(1, pi).squeeze(1)
    start = min(len(k) - 1, math.ceil(0.10 * len(k)))
    negative_ranks = torch.randint(start, len(k), (queries, 16), generator=gen, device=q.device)
    return qi, positive, order.gather(1, negative_ranks)
