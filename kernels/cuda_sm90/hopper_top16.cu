#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>

namespace {
constexpr int threads = 256;
constexpr int buckets = 3000;
constexpr int heads = 6;
constexpr int items = (buckets + threads - 1) / threads;
constexpr int selected = 16;
constexpr int id_bits = 12;
constexpr uint64_t id_mask = (1 << id_bits) - 1;
using Sort = cub::BlockRadixSort<uint64_t, threads, items>;
constexpr int shards = 8;
constexpr int shard_threads = 64;
constexpr int shard_items = (buckets + shards * shard_threads - 1) / (shards * shard_threads);

// Match the archived route's lane order, CTA reduction and FP32 arithmetic.
template<bool Max>
__device__ float reduce(float value, float* shared) {
    const int tid = threadIdx.x;
    for (int delta = 16; delta; delta /= 2) {
        const float other = __shfl_down_sync(0xffffffffu, value, delta);
        value = Max ? fmaxf(value, other) : value + other;
    }
    if (tid % 32 == 0) shared[tid / 32] = value;
    __syncthreads();
    if (tid < 32) {
        value = tid < 8 ? shared[tid] : (Max ? -INFINITY : 0.f);
        for (int delta = 16; delta; delta /= 2) {
            const float other = __shfl_down_sync(0xffffffffu, value, delta);
            value = Max ? fmaxf(value, other) : value + other;
        }
        if (tid == 0) shared[0] = value;
    }
    __syncthreads();
    value = shared[0];
    __syncthreads();
    return value;
}

__device__ void probabilities(const float* score, const float* log_count, const bool (&valid)[items],
                              float (&prob)[items], float* shared) {
    const int tid = threadIdx.x;
    float maximum = -INFINITY;
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        const int f = tid + j * threads;
        prob[j] = valid[j] ? score[f] + log_count[f] : -INFINITY;
        maximum = fmaxf(maximum, prob[j]);
    }
    maximum = reduce<true>(maximum, shared);
    float denominator = 0.f;
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        prob[j] = isfinite(maximum) ? expf(prob[j] - maximum) : 0.f;
        denominator += prob[j];
    }
    denominator = reduce<false>(denominator, shared);
    #pragma unroll
    for (int j = 0; j < items; ++j) prob[j] = denominator > 0 ? prob[j] / denominator : 0.f;
}

__global__ void norm_heads(const float* scores, const float* logs, const int* counts, float* normalized) {
    const int group = blockIdx.x;
    const int start = (group * heads + blockIdx.y) * buckets;
    __shared__ float shared[8];
    bool valid[items];
    float prob[items];
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        const int f = threadIdx.x + j * threads;
        valid[j] = f < buckets && counts[group * buckets + f] > 0;
    }
    probabilities(scores + start, logs + group * buckets, valid, prob, shared);
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        const int f = threadIdx.x + j * threads;
        if (f < buckets) normalized[start + f] = prob[j];
    }
}

__device__ uint64_t order_key(float aggregate, int feature) {
    // Probabilities are nonnegative. Prefer smaller bucket IDs on ties.
    return (uint64_t(__float_as_uint(aggregate)) << id_bits) | (id_mask - feature);
}

__device__ void write_top(uint64_t key, const int* counts, int64_t* ids, float* active, int* raw) {
    const int lane = threadIdx.x;
    const int group = blockIdx.x;
    int count = 0;
    if (lane < selected) {
        const bool live = key != 0;
        const int feature = live ? int(id_mask - (key & id_mask)) : 0;
        ids[group * selected + lane] = feature;
        active[group * selected + lane] = live ? 1.f : 0.f;
        count = live ? counts[feature] : 0;
    }
    for (int delta = 16; delta; delta /= 2) count += __shfl_down_sync(0xffffffffu, count, delta);
    if (lane == 0) raw[group] = count;
}

__device__ uint64_t warp_best(uint64_t key) {
    const unsigned score = key >> id_bits;
    const unsigned best_score = __reduce_max_sync(0xffffffffu, score);
    const unsigned best_id = __reduce_max_sync(0xffffffffu, score == best_score ? unsigned(key & id_mask) : 0u);
    return (uint64_t(best_score) << id_bits) | best_id;
}

__device__ uint64_t thread_best(const uint64_t (&order)[items]) {
    uint64_t best = 0;
    #pragma unroll
    for (int j = 0; j < items; ++j) best = max(best, order[j]);
    return best;
}

__device__ uint64_t select_warp(uint64_t (&order)[items], uint64_t* shared) {
    const int tid = threadIdx.x;
    uint64_t local = thread_best(order);
    uint64_t result = 0;
    for (int rank = 0; rank < selected; ++rank) {
        const uint64_t warp = warp_best(local);
        if (tid % 32 == 0) shared[tid / 32] = warp;
        __syncthreads();
        if (tid < 32) {
            const uint64_t winner = warp_best(tid < 8 ? shared[tid] : 0);
            if (tid == 0) shared[8] = winner;
        }
        __syncthreads();
        const uint64_t winner = shared[8];
        if (tid == rank) result = winner;
        if (local == winner && winner != 0) {
            #pragma unroll
            for (int j = 0; j < items; ++j) if (order[j] == winner) order[j] = 0;
            local = thread_best(order);
        }
    }
    return result;
}

template<bool Prenormalized, bool WarpSelect = false>
__global__ void radix_top(const float* scores, const float* logs, const int* counts,
                          int64_t* ids, float* active, int* raw) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    counts += group * buckets;
    scores += group * heads * buckets;
    __shared__ float shared[8];
    bool valid[items];
    float aggregate[items] = {};
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        const int f = tid + j * threads;
        valid[j] = f < buckets && counts[f] > 0;
    }
    for (int head = 0; head < heads; ++head) {
        if constexpr (Prenormalized) {
            #pragma unroll
            for (int j = 0; j < items; ++j) {
                if (valid[j]) aggregate[j] += scores[head * buckets + tid + j * threads];
            }
        } else {
            float prob[items];
            probabilities(scores + head * buckets, logs + group * buckets, valid, prob, shared);
            #pragma unroll
            for (int j = 0; j < items; ++j) aggregate[j] += prob[j];
        }
    }
    uint64_t order[items];
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        const int f = tid + j * threads;
        order[j] = valid[j] ? order_key(aggregate[j], f) : 0;
    }
    if constexpr (WarpSelect) {
        __shared__ uint64_t warp_keys[9];
        const auto key = select_warp(order, warp_keys);
        if (tid < 32) write_top(key, counts, ids, active, raw);
    } else {
        __shared__ Sort::TempStorage storage;
        Sort(storage).SortDescendingBlockedToStriped(order, 0, 31 + id_bits);
        if (tid < 32) write_top(order[0], counts, ids, active, raw);
    }
}

__global__ void shard_top(const float* normalized, const int* counts, uint64_t* shortlist) {
    using Local = cub::BlockRadixSort<uint64_t, shard_threads, shard_items>;
    __shared__ Local::TempStorage storage;
    const int group = blockIdx.x;
    const int shard = blockIdx.y;
    uint64_t order[shard_items];
    #pragma unroll
    for (int j = 0; j < shard_items; ++j) {
        const int feature = (shard * shard_items + j) * shard_threads + threadIdx.x;
        order[j] = 0;
        if (feature < buckets && counts[group * buckets + feature] > 0) {
            float aggregate = 0;
            #pragma unroll
            for (int head = 0; head < heads; ++head)
                aggregate += normalized[(group * heads + head) * buckets + feature];
            order[j] = order_key(aggregate, feature);
        }
    }
    Local(storage).SortDescendingBlockedToStriped(order, 0, 31 + id_bits);
    if (threadIdx.x < selected) shortlist[(group * shards + shard) * selected + threadIdx.x] = order[0];
}

__global__ void merge_top(const uint64_t* shortlist, const int* counts, int64_t* ids, float* active, int* raw) {
    constexpr int local_items = shards * selected / 32;
    using Merge = cub::BlockRadixSort<uint64_t, 32, local_items>;
    __shared__ Merge::TempStorage storage;
    const int group = blockIdx.x;
    uint64_t order[local_items];
    #pragma unroll
    for (int j = 0; j < local_items; ++j) order[j] = shortlist[group * shards * selected + j * 32 + threadIdx.x];
    Merge(storage).SortDescendingBlockedToStriped(order, 0, 31 + id_bits);
    write_top(order[0], counts + group * buckets, ids, active, raw);
}
}  // namespace

extern "C" int hopper_top16(const float* scores, const float* logs, const int* counts,
                            int64_t* ids, float* active, int* raw, float* normalized,
                            uint64_t* shortlist, int groups, int mode, cudaStream_t stream) {
    if (mode == 1) {
        radix_top<false><<<groups, threads, 0, stream>>>(scores, logs, counts, ids, active, raw);
    } else if (mode >= 2 && mode <= 5) {
        if (mode == 4) {
            radix_top<false, true><<<groups, threads, 0, stream>>>(scores, logs, counts, ids, active, raw);
            return cudaGetLastError();
        }
        norm_heads<<<dim3(groups, heads), threads, 0, stream>>>(scores, logs, counts, normalized);
        auto status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        if (mode == 2) {
            radix_top<true><<<groups, threads, 0, stream>>>(normalized, logs, counts, ids, active, raw);
        } else if (mode == 3) {
            shard_top<<<dim3(groups, shards), shard_threads, 0, stream>>>(normalized, counts, shortlist);
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            merge_top<<<groups, 32, 0, stream>>>(shortlist, counts, ids, active, raw);
        } else {
            radix_top<true, true><<<groups, threads, 0, stream>>>(normalized, logs, counts, ids, active, raw);
        }
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
