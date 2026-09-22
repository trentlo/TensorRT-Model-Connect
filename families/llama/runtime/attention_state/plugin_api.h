/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

// Prototype bridge for TensorRT releases without requireOutputAlias. Registered
// allocations belong to the runtime. A deserialized mutating plugin rejects
// any address outside this set, or a non-identical output address. This checks
// external identity at execution; it does not add engine alias metadata.
extern "C" bool trtmcAttentionStateInit() noexcept;
extern "C" bool trtmcAttentionStateRegister(void* address, std::size_t bytes) noexcept;
extern "C" void trtmcAttentionStateUnregister(void* address) noexcept;
