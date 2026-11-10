#include <cuda_runtime.h>
#include <stdint.h>

// Four warps copy four positions. Each lane owns one aligned 16-byte vector.
__global__ void gather_candidates(const uint4* keys, const uint4* values,
                                  const int* indices, const int* counts,
                                  uint4* packed_keys, uint4* packed_values,
                                  int history, int tail, int stride, int packed_history) {
    constexpr int groups = 4, vectors = 32;
    int group = blockIdx.y;
    int row = blockIdx.x * 4 + threadIdx.x / vectors;
    int lane = threadIdx.x % vectors;
    int old = counts[group];
    if (row >= old && (row < packed_history || row >= packed_history + tail)) return;
    int pos = row < old ? indices[group * stride + row] : history + row - packed_history;
    int64_t src = (int64_t(pos) * groups + group) * vectors + lane;
    int64_t dst = (int64_t(row) * groups + group) * vectors + lane;
    packed_keys[dst] = keys[src];
    packed_values[dst] = values[src];
}

extern "C" int gather_kv(uint64_t k, uint64_t v, uint64_t ids, uint64_t counts,
                          uint64_t out_k, uint64_t out_v, int history, int tail,
                          int stride, int packed_history, uint64_t stream) {
    if (packed_history + tail <= 0) return 0;
    gather_candidates<<<dim3((packed_history + tail + 3) / 4, 4), 128, 0,
                         (cudaStream_t)stream>>>(
        (const uint4*)k, (const uint4*)v, (const int*)ids, (const int*)counts,
        (uint4*)out_k, (uint4*)out_v, history, tail, stride, packed_history);
    return (int)cudaGetLastError();
}
