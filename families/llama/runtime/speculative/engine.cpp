/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace trtmc::llama::speculative {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("speculative ABI: " + message);
}
} // namespace

Contract Contract::parse(const nlohmann::json& value) {
    require(value.at("version") == 1 && value.at("precision") == "fp16" &&
                value.at("cache_layout") == "batch_heads_capacity_dim" &&
                value.at("cache_update") == "aliased_contiguous_append",
            "unsupported version, precision or state layout");
    const auto role = value.at("role").get<std::string>();
    require(role == "target" || role == "draft", "invalid engine role");
    Contract c{value.at("layers"),    value.at("hidden_size"),   value.at("kv_heads"),
               value.at("head_dim"),  value.at("vocab_size"),    value.at("capacity"),
               value.at("max_query"), value.at("feature_width"), role == "draft"};
    require(c.layers > 0 && c.hidden > 0 && c.heads > 0 && c.dim > 0 && c.vocab > 0 &&
                c.capacity > 0 && c.max_query > 0 && c.max_query <= c.capacity &&
                c.feature_width > 0,
            "invalid dimensions");
    return c;
}

Engine::Engine(std::unique_ptr<ITrtModule> module, Contract contract)
    : module_(std::move(module)), contract_(contract) {
    require(module_ && module_->ok(), "invalid execution module");
    const auto& c = contract_;
    for (const auto* name : {"token_id", "position_id", "logits_indices", "attention_mask",
                             "cache_write_indices", "key_value_lengths"}) {
        require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kInt32,
                std::string("missing or mistyped input ") + name);
    }
    require(module_->has_output("logits") && module_->tensor_dtype("logits") == DType::kFloat32,
            "missing FP32 logits");
    require(module_->has_output("features") && module_->tensor_dtype("features") == DType::kFloat16,
            "missing FP16 features");
    require(module_->optimization_profile_count() == 1, "expected one query profile");
    for (const auto* name : {"token_id", "position_id", "logits_indices"}) {
        require(module_->input_profile_shape(name, 0, ProfileShapeSelector::kMin) ==
                        std::vector<std::int64_t>{1} &&
                    module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax) ==
                        std::vector<std::int64_t>{c.max_query},
                std::string("query profile mismatch: ") + name);
    }
    require(module_->input_profile_shape("attention_mask", 0, ProfileShapeSelector::kMax) ==
                std::vector<std::int64_t>({c.max_query, c.capacity}),
            "visibility profile mismatch");
    for (const auto* name : {"cache_write_indices", "key_value_lengths"})
        require(module_->tensor_shape(name) == std::vector<std::int64_t>{1},
                "invalid state scalar shape");
    for (const auto* name : {"logits", "features"}) {
        const auto output_shape = module_->tensor_shape(name);
        const int width = name[0] == 'l' ? c.vocab : (c.draft ? c.hidden : c.feature_width);
        require(output_shape.size() == 2 && output_shape.back() == width,
                std::string("output row contract mismatch: ") + name);
    }
    if (c.draft) {
        for (const auto* name : {"target_features", "draft_features"}) {
            const int width = name[0] == 't' ? c.feature_width : c.hidden;
            require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kFloat16 &&
                        module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax) ==
                            std::vector<std::int64_t>({c.max_query, width}),
                    std::string("conditioning contract mismatch: ") + name);
        }
    }
    const std::vector<std::int64_t> shape{1, c.heads, c.capacity, c.dim};
    for (int layer = 0; layer < c.layers; ++layer) {
        keys_.push_back(DeviceTensor::zeros(shape, DType::kFloat16, module_->stream()));
        values_.push_back(DeviceTensor::zeros(shape, DType::kFloat16, module_->stream()));
        for (const auto* kind : {"k", "v"}) {
            const auto suffix = std::string(kind) + "_" + std::to_string(layer);
            const auto input = "cache_" + suffix;
            const auto output = "present_" + suffix;
            require(module_->has_input(input) && module_->has_output(output) &&
                        module_->tensor_shape(input) == shape &&
                        module_->tensor_shape(output) == shape &&
                        module_->tensor_dtype(input) == DType::kFloat16 &&
                        module_->tensor_dtype(output) == DType::kFloat16,
                    "cache tensor does not match manifest: " + input);
            auto& tensor = kind[0] == 'k' ? keys_.back() : values_.back();
            require(tensor.ok(), "KV allocation failed");
            module_->bind_external(input, tensor.data());
            // TensorRT must declare the alias so the compiler sees the state
            // effect. Binding two unrelated tensors to one pointer is not an
            // implementation of this contract, including for plugin lowerings.
            require(module_->device_ptr(output) == tensor.data(),
                    "engine does not declare the required cache alias: " + output);
        }
    }
    module_->sync();
}

StepResult Engine::run(const std::vector<std::int32_t>& tokens, int start,
                       const std::vector<std::int32_t>& parents, bool all_logits,
                       const std::vector<std::uint16_t>& target_features,
                       const std::vector<std::uint16_t>& draft_features) {
    const auto& c = contract_;
    const int rows = static_cast<int>(tokens.size());
    require(rows > 0 && rows <= c.max_query && start >= 0 && start <= c.capacity - rows,
            "query exceeds engine profile or cache bounds");
    require(parents.size() == tokens.size(), "parent count mismatch");
    std::vector<std::int32_t> positions(rows), mask(static_cast<std::size_t>(rows) * c.capacity, 0);
    for (int row = 0; row < rows; ++row) {
        require(parents[row] >= -1 && parents[row] < row, "parents must precede children");
        positions[row] = parents[row] < 0 ? start : positions[parents[row]] + 1;
        auto* visible = mask.data() + static_cast<std::size_t>(row) * c.capacity;
        std::fill(visible, visible + start, 1);
        for (int ancestor = row; ancestor >= 0; ancestor = parents[ancestor])
            visible[start + ancestor] = 1;
    }
    std::vector<std::int32_t> selected;
    if (all_logits) {
        for (int row = 0; row < rows; ++row)
            selected.push_back(row);
    } else {
        selected.push_back(rows - 1);
    }
    std::int32_t write_start = start, length = start + rows;
    TensorMap inputs{
        {"token_id", {const_cast<std::int32_t*>(tokens.data()), {rows}, DType::kInt32}},
        {"position_id", {positions.data(), {rows}, DType::kInt32}},
        {"logits_indices",
         {selected.data(), {static_cast<std::int64_t>(selected.size())}, DType::kInt32}},
        {"attention_mask", {mask.data(), {rows, c.capacity}, DType::kInt32}},
        {"cache_write_indices", {&write_start, {1}, DType::kInt32}},
        {"key_value_lengths", {&length, {1}, DType::kInt32}},
    };
    if (c.draft) {
        require(target_features.size() == static_cast<std::size_t>(rows) * c.feature_width &&
                    draft_features.size() == static_cast<std::size_t>(rows) * c.hidden,
                "draft conditioning shape mismatch");
        inputs["target_features"] = {const_cast<std::uint16_t*>(target_features.data()),
                                     {rows, c.feature_width},
                                     DType::kFloat16};
        inputs["draft_features"] = {
            const_cast<std::uint16_t*>(draft_features.data()), {rows, c.hidden}, DType::kFloat16};
    }
    module_->forward_async(inputs);
    StepResult result;
    result.logits.resize(selected.size() * c.vocab);
    result.features.resize(static_cast<std::size_t>(rows) * (c.draft ? c.hidden : c.feature_width));
    checked(cudaMemcpyAsync(result.logits.data(), module_->device_ptr("logits"),
                            result.logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                            module_->stream()));
    checked(cudaMemcpyAsync(result.features.data(), module_->device_ptr("features"),
                            result.features.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                            module_->stream()));
    module_->sync();
    return result;
}

void Engine::commit(int start, const std::vector<std::int32_t>& rows) {
    const auto& c = contract_;
    require(start >= 0 && !rows.empty() && start + static_cast<int>(rows.size()) <= c.capacity,
            "invalid commit destination");
    for (auto row : rows)
        require(row >= 0 && row < c.max_query && start + row < c.capacity, "invalid commit source");
    DeviceTensor scratch({c.heads, static_cast<std::int64_t>(rows.size()), c.dim}, DType::kFloat16,
                         module_->stream());
    require(scratch.ok(), "commit scratch allocation failed");
    const auto width = static_cast<std::size_t>(c.dim) * sizeof(std::uint16_t);
    for (auto* storage : {&keys_, &values_}) {
        for (auto& cache : *storage) {
            for (std::size_t row = 0; row < rows.size(); ++row) {
                checked(cudaMemcpy2DAsync(
                    static_cast<char*>(scratch.data()) + row * width, rows.size() * width,
                    static_cast<char*>(cache.data()) + (start + rows[row]) * width,
                    c.capacity * width, width, c.heads, cudaMemcpyDeviceToDevice,
                    module_->stream()));
            }
            checked(cudaMemcpy2DAsync(static_cast<char*>(cache.data()) + start * width,
                                      c.capacity * width, scratch.data(), rows.size() * width,
                                      rows.size() * width, c.heads, cudaMemcpyDeviceToDevice,
                                      module_->stream()));
        }
    }
    module_->sync();
}

void Engine::reset() {
    // No byte clearing: subsequent inputs expose only the current valid prefix.
    module_->reset_execution_context();
}

std::int32_t argmax(const float* values, int count) {
    require(count > 0, "empty logits");
    for (int index = 0; index < count; ++index)
        require(std::isfinite(values[index]), "non-finite logits");
    return static_cast<std::int32_t>(std::max_element(values, values + count) - values);
}

std::vector<std::int32_t> greedy_path(const std::vector<std::int32_t>& tokens,
                                      const std::vector<std::int32_t>& parents,
                                      const std::vector<float>& logits, int vocab) {
    require(!tokens.empty() && parents.size() == tokens.size() && parents[0] == -1 && vocab > 0 &&
                logits.size() == tokens.size() * static_cast<std::size_t>(vocab),
            "invalid verification rows");
    for (std::size_t row = 1; row < parents.size(); ++row)
        require(parents[row] >= 0 && parents[row] < static_cast<int>(row), "invalid tree");
    std::vector<std::int32_t> path{0};
    for (;;) {
        const auto parent = path.back();
        const auto next = argmax(logits.data() + static_cast<std::size_t>(parent) * vocab, vocab);
        auto found = tokens.size();
        for (std::size_t row = parent + 1; row < tokens.size(); ++row) {
            if (parents[row] == parent && tokens[row] == next) {
                found = row;
                break;
            }
        }
        if (found == tokens.size())
            return path;
        path.push_back(static_cast<std::int32_t>(found));
    }
}
} // namespace trtmc::llama::speculative
