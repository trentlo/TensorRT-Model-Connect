/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Included after the vendored XQA types. Tensor-core P*V propagates a masked
// NaN through 0*NaN. Repair a CTA with non-finite outputs using the scalar
// semantics, avoiding a redundant scan of the full cache on the common path.
__device__ inline bool mc_xqa_contract_fallback(const IOHead* query, OutputHead* output,
                                                const bool* mask, const KVCacheList<true>& cache,
                                                unsigned batch, unsigned kv_head,
                                                unsigned first_row, unsigned rows, unsigned queries,
                                                unsigned kv_heads, unsigned group, float* scores,
                                                bool force_scalar = false) {
    // The initial guard uses the whole CTA. After XQA, producer warps have
    // exited and only its consumer half participates in scalar repair.
    const unsigned tid =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (force_scalar ? 0 : threadIdx.z));
    const unsigned threads = blockDim.x * blockDim.y * (force_scalar ? 1 : blockDim.z);
    const unsigned capacity = cache.maxNbPagesPerSeq * TOKENS_PER_PAGE;
    const int raw_length = reinterpret_cast<const int*>(cache.seqLenList)[batch];
    const unsigned length = raw_length < 0 ? 0 : min(unsigned(raw_length), capacity);
    __shared__ int scalar;
    if (tid == 0)
        scalar = force_scalar || length == 0;
    __syncthreads();
    const auto* key_pages = cache.kvCachePageList + batch * cache.maxNbPagesPerSeq;
    const auto* value_pages = cache.valuePageList + batch * cache.maxNbPagesPerSeq;
    if (!force_scalar) {
        for (unsigned page = tid; page < (length + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;
             page += threads) {
            const int kp = key_pages[page], vp = value_pages[page];
            if (kp < 0 || unsigned(kp) >= cache.keyPageCount || vp < 0 ||
                unsigned(vp) >= cache.valuePageCount)
                atomicExch(&scalar, 1);
        }
    }
    __syncthreads();
    if (!scalar)
        return false;

    float* reduction = scores + capacity;
    for (unsigned row = first_row; row < first_row + rows; ++row) {
        const unsigned q = row / group, h = kv_head * group + row % group;
        const auto offset = (size_t(batch) * kv_heads * group + h) * queries + q;
        const auto* visible = mask + ((cache.maskHeads == 1 ? 0 : h) * queries + q) * capacity;
        float local_max = -INFINITY;
        for (unsigned j = tid; j < capacity; j += threads) {
            float score = -INFINITY;
            if (j < length && visible[j]) {
                const int page = key_pages[j / TOKENS_PER_PAGE];
                if (page >= 0 && unsigned(page) < cache.keyPageCount) {
                    const auto& key = cache.pool[(page * kv_heads + kv_head) * TOKENS_PER_PAGE +
                                                 j % TOKENS_PER_PAGE];
                    float dot = 0;
                    for (unsigned d = 0; d < HEAD_ELEMS; ++d)
                        dot = fmaf(__half2float(query[offset][d]), __half2float(key[d]), dot);
                    score = __half2float(__float2half_rn(dot));
                }
            }
            scores[j] = score;
            local_max = fmaxf(local_max, score);
        }
        reduction[tid] = local_max;
        __syncthreads();
        for (unsigned stride = threads / 2; stride; stride /= 2) {
            if (tid < stride)
                reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            __syncthreads();
        }
        const float maximum = reduction[0];
        float sum = 0;
        for (unsigned j = tid; j < capacity; j += threads) {
            const float p = scores[j] == -INFINITY ? 0 : expf(scores[j] - maximum);
            scores[j] = p;
            sum += p;
        }
        __syncthreads();
        reduction[tid] = sum;
        __syncthreads();
        for (unsigned stride = threads / 2; stride; stride /= 2) {
            if (tid < stride)
                reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        const float denominator = reduction[0];
        for (unsigned j = tid; j < capacity; j += threads)
            scores[j] =
                denominator > 0 ? __half2float(__float2half_rn(scores[j] / denominator)) : 0;
        __syncthreads();
        for (unsigned d = tid; d < HEAD_ELEMS; d += threads) {
            float result = 0;
            for (unsigned j = 0; j < length; ++j) {
                if (scores[j] == 0)
                    continue;
                const int page = value_pages[j / TOKENS_PER_PAGE];
                if (page >= 0 && unsigned(page) < cache.valuePageCount) {
                    const auto& value =
                        cache.valuePool[(page * kv_heads + kv_head) * TOKENS_PER_PAGE +
                                        j % TOKENS_PER_PAGE];
                    result = fmaf(scores[j], __half2float(value[d]), result);
                }
            }
            output[offset][d] = __float2half_rn(result);
        }
        __syncthreads();
    }
    return true;
}
