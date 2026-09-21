/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernels.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace trtmc::llama::attention_state {
namespace {
__global__ void indexed_update(half* cache, const half* rows, const int* slots, int batch,
                               int heads, int queries, int pages, int page_size, int dim) {
    const int64_t count = int64_t(batch) * heads * queries * dim;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        const int d = i % dim;
        const int q = i / dim % queries;
        const int h = i / (dim * queries) % heads;
        const int b = i / (int64_t(dim) * queries * heads);
        const int slot = slots[b * queries + q];
        // Runtime validates uniqueness/bounds. Bounds checks also protect
        // builder profiling, whose execution tensors need not be meaningful.
        if (slot >= 0 && int64_t(slot) < int64_t(pages) * page_size) {
            const int64_t offset =
                ((int64_t(slot / page_size) * heads + h) * page_size + slot % page_size) * dim;
            cache[offset + d] = rows[i];
        }
    }
}

// Correctness-oriented padded FP16 GQA. Q is already scaled. FP16 score and
// probability rounding matches the primitive graph's exposed tensor types.
// Each CTA owns one query/head; masked or out-of-length rows are never loaded.
__global__ void paged_attention(const half* query, const half* keys, const half* values,
                                const int* key_pages, const int* value_pages, const int* lengths,
                                const bool* mask, half* output, int heads, int kv_heads,
                                int queries, int key_count, int value_count, int page_size,
                                int logical_pages, int dim, int mask_heads) {
    extern __shared__ float scores[];
    __shared__ float reduction[256];
    const int q = blockIdx.x % queries;
    const int h = blockIdx.x / queries % heads;
    const int b = blockIdx.x / (queries * heads);
    const int kh = h / (heads / kv_heads);
    const int capacity = logical_pages * page_size;
    const int length = max(0, min(lengths[b], capacity));
    const int64_t query_offset = ((int64_t(b) * heads + h) * queries + q) * dim;
    const int64_t mask_offset =
        ((int64_t(b) * mask_heads + (mask_heads == 1 ? 0 : h)) * queries + q) * capacity;
    float local_max = -CUDART_INF_F;
    for (int j = threadIdx.x; j < capacity; j += blockDim.x) {
        float score = -CUDART_INF_F;
        if (j < length && mask[mask_offset + j]) {
            const int page = key_pages[b * logical_pages + j / page_size];
            if (page >= 0 && page < key_count) {
                const int64_t offset =
                    ((int64_t(page) * kv_heads + kh) * page_size + j % page_size) * dim;
                float dot = 0.0F;
                for (int d = 0; d < dim; ++d)
                    dot = fmaf(__half2float(query[query_offset + d]),
                               __half2float(keys[offset + d]), dot);
                score = __half2float(__float2half_rn(dot));
            }
        }
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];
    float total = 0.0F;
    for (int j = threadIdx.x; j < capacity; j += blockDim.x) {
        const float probability = scores[j] == -CUDART_INF_F ? 0.0F : expf(scores[j] - maximum);
        scores[j] = probability;
        total += probability;
    }
    reduction[threadIdx.x] = total;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = reduction[0];
    for (int j = threadIdx.x; j < capacity; j += blockDim.x)
        scores[j] = denominator > 0 ? __half2float(__float2half_rn(scores[j] / denominator)) : 0;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0F;
        for (int j = 0; j < length; ++j) {
            // Avoid 0*NaN from invisible/stale cache rows.
            if (scores[j] == 0)
                continue;
            const int page = value_pages[b * logical_pages + j / page_size];
            if (page >= 0 && page < value_count) {
                const int64_t offset =
                    ((int64_t(page) * kv_heads + kh) * page_size + j % page_size) * dim;
                sum = fmaf(scores[j], __half2float(values[offset + d]), sum);
            }
        }
        output[query_offset + d] = __float2half_rn(sum);
    }
}
} // namespace

cudaError_t update(void* cache, const void* rows, const int* slots, int batch, int heads,
                   int queries, int pages, int page_size, int dim, cudaStream_t stream) {
    indexed_update<<<256, 256, 0, stream>>>(static_cast<half*>(cache),
                                            static_cast<const half*>(rows), slots, batch, heads,
                                            queries, pages, page_size, dim);
    return cudaGetLastError();
}

cudaError_t attention(const void* query, const void* keys, const void* values, const int* key_pages,
                      const int* value_pages, const int* lengths, const bool* mask, void* output,
                      int batch, int heads, int kv_heads, int queries, int key_count,
                      int value_count, int page_size, int logical_pages, int dim, int mask_heads,
                      cudaStream_t stream) {
#ifdef TRTMC_LLAMA_EDGE_XQA
    const auto addresses =
        reinterpret_cast<std::uintptr_t>(query) | reinterpret_cast<std::uintptr_t>(keys) |
        reinterpret_cast<std::uintptr_t>(values) | reinterpret_cast<std::uintptr_t>(output);
    if (dim == 128 && page_size == 64 && queries <= 64 && addresses % 16 == 0)
        return xqa_attention(query, keys, values, key_pages, value_pages, lengths, mask, output,
                             batch, heads, kv_heads, queries, key_count, value_count, logical_pages,
                             mask_heads, stream);
#endif
    paged_attention<<<batch * heads * queries, 256, logical_pages * page_size * sizeof(float),
                      stream>>>(static_cast<const half*>(query), static_cast<const half*>(keys),
                                static_cast<const half*>(values), key_pages, value_pages, lengths,
                                mask, static_cast<half*>(output), heads, kv_heads, queries,
                                key_count, value_count, page_size, logical_pages, dim, mask_heads);
    return cudaGetLastError();
}
} // namespace trtmc::llama::attention_state
