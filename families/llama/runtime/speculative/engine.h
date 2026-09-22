/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/llama/runtime/speculative/state_layout.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace trtmc::llama::speculative {

enum class Phase { kPrefill, kDecode };

struct ExecutionProfile {
    std::array<int, 3> query, logits;
};

struct Contract {
    int layers, hidden, heads, dim, vocab, capacity, max_query, feature_width;
    bool draft;
    int page_size = 0;
    std::vector<ExecutionProfile> profiles{};
    int query_limit(Phase phase) const {
        return profiles.empty() ? max_query : profiles[phase == Phase::kPrefill ? 0 : 1].query[2];
    }
    static Contract parse(const nlohmann::json& value);
};

struct StepResult {
    std::vector<float> logits;
    // FP16 bits; no host arithmetic is performed on hidden states.
    std::vector<std::uint16_t> features;
};

// Runtime/compiler boundary only. Does not propose or accept tokens.
class Engine {
  public:
    Engine(std::unique_ptr<ITrtModule> module, Contract contract,
           std::unique_ptr<ITrtModule> prefill = nullptr);
    StepResult run(Phase phase, const std::vector<std::int32_t>& tokens, int start,
                   const std::vector<std::int32_t>& parents, bool all_logits,
                   const std::vector<std::uint16_t>& target_features = {},
                   const std::vector<std::uint16_t>& draft_features = {});
    // Copy a verified root-to-leaf path out of speculative slots. Two-phase
    // gather avoids overwriting a source slot that a later destination needs.
    void commit(int start, const std::vector<std::int32_t>& rows);
    void reset();
    const Contract& contract() const { return contract_; }

  private:
    std::unique_ptr<ITrtModule> module_;
    std::unique_ptr<ITrtModule> prefill_;
    Contract contract_;
    StateLayout layout_;
    std::vector<DeviceTensor> keys_, values_;
    std::vector<std::shared_ptr<void>> cache_registrations_;
};

// Host policy helpers; independent of engine lowering and tested without a GPU.
std::vector<std::int32_t> greedy_path(const std::vector<std::int32_t>& tokens,
                                      const std::vector<std::int32_t>& parents,
                                      const std::vector<float>& logits, int vocab);
std::int32_t argmax(const float* values, int count);

} // namespace trtmc::llama::speculative
