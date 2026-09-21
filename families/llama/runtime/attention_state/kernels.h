/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuda_runtime_api.h>

namespace trtmc::llama::attention_state {
cudaError_t update(void* cache, const void* rows, const int* slots, int batch, int heads,
                   int queries, int pages, int page_size, int dim, cudaStream_t stream);
cudaError_t attention(const void* query, const void* keys, const void* values, const int* key_pages,
                      const int* value_pages, const int* lengths, const bool* mask, void* output,
                      int batch, int heads, int kv_heads, int queries, int key_count,
                      int value_count, int page_size, int logical_pages, int dim, int mask_heads,
                      cudaStream_t stream);
} // namespace trtmc::llama::attention_state
