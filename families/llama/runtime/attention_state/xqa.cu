/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernels.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

// Compile only the FP16, D=128, page=64 multi-query kernel required by this
// Llama/EAGLE3 prototype. Other valid plugin geometries retain the scalar path.
#define HEAD_ELEMS 128
#define HEAD_GRP_SIZE 0
#define INPUT_FP16 1
#define BEAM_WIDTH 1
#define TOKENS_PER_PAGE 64
#define CACHE_ELEM_ENUM 0
#define SPEC_DEC 1
#define M_TILESIZE 32
#define SLIDING_WINDOW 0
#define ENABLE_PDL 0
#define ALLOW_MULTI_BLOCK_MODE false
#ifndef NDEBUG
#define NDEBUG
#endif
#include "edge_xqa/mha.cu"

namespace trtmc::llama::attention_state {
cudaError_t xqa_attention(const void* query, const void* keys, const void* values,
                          const int* key_pages, const int* value_pages, const int* lengths,
                          const bool* mask, void* output, int batch, int heads, int kv_heads,
                          int queries, int key_count, int value_count, int logical_pages,
                          int mask_heads, cudaStream_t stream) {
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        return status;
    // Initialization may synchronize; steady-state launches do not. Per-thread,
    // per-current-device storage avoids a host lock in every attention layer.
    static thread_local int cached_device = -1;
    static thread_local unsigned shared_bytes = 0;
    if (cached_device != device) {
        status = cudaMemcpyFromSymbol(&shared_bytes, smemSize, sizeof(shared_bytes));
        if (status != cudaSuccess)
            return status;
        shared_bytes = std::max(shared_bytes, unsigned((4096 + 256) * sizeof(float)));
        status = cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      shared_bytes);
        if (status != cudaSuccess)
            return status;
        cached_device = device;
    }
    const unsigned group = heads / kv_heads;
    const dim3 grid(1, kv_heads * ((queries * group + 31) / 32), batch);
    const dim3 block(128, 1, 2);
    const KVCacheList<true> cache{static_cast<const GMemCacheHead*>(keys),
                                  static_cast<const GMemCacheHead*>(values),
                                  key_pages,
                                  value_pages,
                                  reinterpret_cast<const unsigned*>(lengths),
                                  unsigned(logical_pages),
                                  unsigned(key_count),
                                  unsigned(value_count),
                                  unsigned(mask_heads)};
    kernel_mha<<<grid, block, shared_bytes, stream>>>(
        queries, kv_heads, group, nullptr, 1.0F, static_cast<OutputHead*>(output),
        static_cast<const IOHead*>(query), mask, nullptr, cache, batch, 1.0F, 1.0F, nullptr,
        nullptr);
    return cudaGetLastError();
}
} // namespace trtmc::llama::attention_state
