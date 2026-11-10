"""Split-K indexed GQA with Tensor Core tiles on NVIDIA Hopper."""
import triton
import triton.language as tl


@triton.jit
def _indexed_tile(Q, K, V, IDS, COUNTS, HISTORIES, LENGTHS, PARTIAL, META,
                  STRIDE: tl.constexpr, CAPACITY: tl.constexpr, SPLITS: tl.constexpr,
                  TILE: tl.constexpr, PV_MODE: tl.constexpr):
    group = tl.program_id(0)
    split = tl.program_id(1)
    slot = group // 4
    kv_head = group % 4
    history = tl.load(HISTORIES + slot)
    tail = tl.maximum(tl.load(LENGTHS + slot) - history, 0)
    old = tl.where(tail > 0, tl.load(COUNTS + group), 0)
    total = old + tail
    span = tl.cdiv(total, SPLITS * TILE) * TILE
    begin = split * span
    end = tl.minimum(total, begin + span)

    heads = tl.arange(0, 16)
    dims = tl.arange(0, 256)
    cols = tl.arange(0, TILE)
    q = tl.load(Q + (group * 6 + heads[:, None]) * 256 + dims[None, :], heads[:, None] < 6, 0)
    running_max = tl.full((16,), -float('inf'), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, 256), 0, tl.float32)
    for start in range(begin, end, TILE):
        ordinal = start + cols
        old_id = tl.load(IDS + group * STRIDE + ordinal, ordinal < tl.minimum(end, old), 0)
        position = tl.where(ordinal < old, old_id, history + ordinal - old)
        token_base = ((slot * CAPACITY + position) * 4 + kv_head) * 256
        k = tl.load(K + token_base[None, :] + dims[:, None], ordinal[None, :] < end, 0)
        scores = tl.dot(q, k) * 0.09016844005556022  # log2(e) / sqrt(256)
        scores = tl.where(ordinal[None, :] < end, scores, -float('inf'))
        next_max = tl.maximum(running_max, tl.max(scores, 1))
        alpha = tl.exp2(running_max - next_max)
        p = tl.exp2(scores - next_max[:, None])
        v = tl.load(V + token_base[:, None] + dims[None, :], ordinal[:, None] < end, 0)
        if PV_MODE != 0:
            # Scale before FP16 conversion to retain small probabilities. K/V
            # storage and QK stay BF16. PV accumulation stays FP32.
            p_scaled = p * 256.
            p_hi = p_scaled.to(tl.float16)
            v_half = v.to(tl.float16)
            acc = tl.dot(p_hi, v_half, acc * alpha[:, None])
            if PV_MODE == 2:
                p_lo = (p_scaled - p_hi.to(tl.float32)).to(tl.float16)
                acc = tl.dot(p_lo, v_half, acc)
        else:
            acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        running_max = next_max

    partial_row = (group * 6 + heads) * SPLITS + split
    if PV_MODE != 0:
        acc = acc / 256.
    tl.store(PARTIAL + partial_row[:, None] * 256 + dims[None, :], acc, heads[:, None] < 6)
    tl.store(META + partial_row * 2, running_max, heads < 6)
    tl.store(META + partial_row * 2 + 1, denominator, heads < 6)


@triton.jit
def _merge_tiles(PARTIAL, META, OUT, SPLITS: tl.constexpr, WIDTH: tl.constexpr,
                 SPLIT_BLOCK: tl.constexpr):
    head = tl.program_id(0)
    dims = tl.program_id(1) * WIDTH + tl.arange(0, WIDTH)
    splits = tl.arange(0, SPLIT_BLOCK)
    row = head * SPLITS + splits
    m = tl.load(META + row * 2, splits < SPLITS, -float('inf'))
    z = tl.load(META + row * 2 + 1, splits < SPLITS, 0)
    max_all = tl.max(m, 0)
    safe_max = tl.where(max_all == -float('inf'), 0.0, max_all)
    factors = tl.exp2(m - safe_max)
    denom = tl.sum(factors * z, 0)
    partial = tl.load(PARTIAL + row[:, None] * 256 + dims[None, :], splits[:, None] < SPLITS, 0)
    result = tl.sum(partial * factors[:, None], 0) / tl.where(denom > 0, denom, 1.)
    tl.store(OUT + head * 256 + dims, result)


class HopperAV:
    def __init__(self, ops):
        self.ids, self.counts, self.output = ops.ids, ops.counts, ops.output
        self.history = ops.history
        heads = ops.output.shape[0] * 24
        # Both backends need exactly 258 FP32 values per split/head. The
        # views below share the legacy scratch allocation on the same stream.
        workspace = ops.state.view(-1)
        self.partial = workspace[:heads * 128 * 256]
        self.meta = workspace[heads * 128 * 256:]

    def attend(self, q, k, v, histories, lengths, ids=None, counts=None, splits=64, tile=32, warps=4,
               pv_mode=1):
        slots = q.shape[0]
        ids = self.ids if ids is None else ids
        counts = self.counts if counts is None else counts
        _indexed_tile[(slots * 4, splits)](
            q, k, v, ids, counts, histories, lengths, self.partial, self.meta,
            self.history, k.shape[1], splits, tile, pv_mode, num_warps=warps, num_stages=2)
        _merge_tiles[(slots * 24, 4)](self.partial, self.meta, self.output, splits, 64,
                                      triton.next_power_of_2(splits), num_warps=4)
        return self.output[:slots]
