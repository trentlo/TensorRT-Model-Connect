/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"
#ifdef TRTMC_LLAMA_ATTENTION_STATE_PLUGINS
#include "families/llama/runtime/attention_state/plugin_api.h"
#endif

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
    const bool linear =
        value.at("version") == 1 && value.at("cache_layout") == "batch_heads_capacity_dim" &&
        value.at("cache_update") == "aliased_contiguous_append" &&
        value.value("attention_backend", "primitives") == "primitives" &&
        value.value("alias_contract", "engine_required_alias") == "engine_required_alias" &&
        value.value("page_size", 0) == 0;
    const bool paged = value.at("version") == 2 &&
                       value.at("cache_layout") == "pages_heads_slots_dim" &&
                       value.at("cache_update") == "aliased_indexed_write" &&
                       value.at("attention_backend") == "plugin" &&
                       value.at("alias_contract") == "plugin_local_alias_runtime_identity_guard";
    require((linear || paged) && value.at("precision") == "fp16",
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
    if (paged) {
        c.page_size = value.at("page_size");
        require(c.page_size > 0 && c.capacity % c.page_size == 0 && c.capacity <= 4096 &&
                    c.dim <= 256,
                "unsupported plugin page geometry");
    }
    if (value.contains("execution_profiles"))
        require(value.at("execution_profiles").is_array(), "execution profiles must be an array");
    if (value.contains("execution_profiles") && !value.at("execution_profiles").empty()) {
        const auto& profiles = value.at("execution_profiles");
        require(profiles.size() == 2, "expected prefill/decode profiles");
        for (int i = 0; i < 2; ++i) {
            require(profiles[i].at("phase") == (i == 0 ? "prefill" : "decode"),
                    "execution profile phase order mismatch");
            for (const auto* name : {"query", "logits"}) {
                const auto& bounds = profiles[i].at(name);
                require(bounds.is_array() && bounds.size() == 3,
                        "expected MIN/OPT/MAX profile bounds");
                for (const auto& bound : bounds)
                    require(bound.is_number_integer(), "profile bounds must be integers");
            }
            ExecutionProfile p{profiles[i].at("query").get<std::array<int, 3>>(),
                               profiles[i].at("logits").get<std::array<int, 3>>()};
            for (const auto& bounds : {p.query, p.logits})
                require(bounds[0] == 1 && bounds[1] >= 1 && bounds[1] <= bounds[2] &&
                            bounds[2] <= c.max_query,
                        "invalid execution profile bounds");
            for (int j = 0; j < 3; ++j)
                require(p.logits[j] <= p.query[j], "selected rows exceed query profile");
            if (i == 0 || c.draft)
                require(p.logits == std::array<int, 3>{1, 1, 1},
                        "prefill/draft must select one logit row");
            c.profiles.push_back(p);
        }
        require(std::max(c.profiles[0].query[2], c.profiles[1].query[2]) == c.max_query,
                "manifest query bound does not match profiles");
    }
    return c;
}

Engine::Engine(std::unique_ptr<ITrtModule> module, Contract contract,
               std::unique_ptr<ITrtModule> prefill)
    : module_(std::move(module)), prefill_(std::move(prefill)), contract_(contract),
      layout_(contract.capacity, contract.page_size) {
    require(module_ && module_->ok(), "invalid execution module");
    const auto& c = contract_;
    require(c.profiles.empty() == (prefill_ == nullptr), "execution module/profile mismatch");
    if (prefill_)
        require(prefill_->ok() && prefill_->profile_idx() == 0 && module_->profile_idx() == 1 &&
                    prefill_->stream() == module_->stream(),
                "prefill/decode must use matching contexts on one stream");
    // Validate both contexts against the compiler's complete row contract.
    for (const auto* module : {module_.get(), prefill_.get()}) {
        if (!module)
            continue;
        const int profile = module->profile_idx();
        require(module->optimization_profile_count() == (c.profiles.empty() ? 1 : 2),
                "unexpected optimization profile count");
        const ExecutionProfile bounds =
            c.profiles.empty() ? ExecutionProfile{{1, std::min(16, c.max_query), c.max_query},
                                                  {1, std::min(16, c.max_query), c.max_query}}
                               : c.profiles.at(profile);
        auto check_rows = [&](const char* name, std::vector<std::int64_t> shape, int axis,
                              const std::array<int, 3>& rows) {
            constexpr ProfileShapeSelector selectors[]{
                ProfileShapeSelector::kMin, ProfileShapeSelector::kOpt, ProfileShapeSelector::kMax};
            for (int i = 0; i < 3; ++i) {
                shape[axis] = rows[i];
                require(module->input_profile_shape(name, profile, selectors[i]) == shape,
                        std::string("profile mismatch: ") + name);
            }
        };
        for (const auto* name : {"token_id", "position_id"})
            check_rows(name, {0}, 0, bounds.query);
        check_rows("logits_indices", {0}, 0, bounds.logits);
        check_rows("attention_mask", {0, c.capacity}, 0, bounds.query);
        if (c.page_size)
            for (const auto* name : {"key_write_slots", "value_write_slots"})
                check_rows(name, {1, 0}, 1, bounds.query);
        if (c.draft) {
            check_rows("target_features", {0, c.feature_width}, 0, bounds.query);
            check_rows("draft_features", {0, c.hidden}, 0, bounds.query);
        }
    }
    for (const auto* name :
         {"token_id", "position_id", "logits_indices", "attention_mask", "key_value_lengths"}) {
        require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kInt32,
                std::string("missing or mistyped input ") + name);
    }
    require(module_->has_output("logits") && module_->tensor_dtype("logits") == DType::kFloat32,
            "missing FP32 logits");
    require(module_->has_output("features") && module_->tensor_dtype("features") == DType::kFloat16,
            "missing FP16 features");
    for (const auto* name : {"key_value_lengths"})
        require(module_->tensor_shape(name) == std::vector<std::int64_t>{1},
                "invalid state scalar shape");
    if (c.page_size) {
        for (const auto* name : {"key_write_slots", "value_write_slots"})
            require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kInt32,
                    "invalid indexed slot input");
        for (const auto* name : {"key_pages", "value_pages"})
            require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kInt32 &&
                        module_->tensor_shape(name) ==
                            std::vector<std::int64_t>({1, c.capacity / c.page_size}),
                    "invalid page table input");
    } else {
        require(module_->has_input("cache_write_indices") &&
                    module_->tensor_dtype("cache_write_indices") == DType::kInt32 &&
                    module_->tensor_shape("cache_write_indices") == std::vector<std::int64_t>{1},
                "invalid linear write index");
    }
    for (const auto* name : {"logits", "features"}) {
        const auto output_shape = module_->tensor_shape(name);
        const int width = name[0] == 'l' ? c.vocab : (c.draft ? c.hidden : c.feature_width);
        require(output_shape.size() == 2 && output_shape.back() == width,
                std::string("output row contract mismatch: ") + name);
    }
    if (c.draft) {
        for (const auto* name : {"target_features", "draft_features"}) {
            require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kFloat16,
                    std::string("conditioning contract mismatch: ") + name);
        }
    }
    const std::vector<std::int64_t> shape{c.page_size ? c.capacity / c.page_size : 1, c.heads,
                                          c.page_size ? c.page_size : c.capacity, c.dim};
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
            if (prefill_)
                prefill_->bind_external(input, tensor.data());
            if (c.page_size) {
#ifdef TRTMC_LLAMA_ATTENTION_STATE_PLUGINS
                auto registration =
                    std::shared_ptr<void>(tensor.data(), trtmcAttentionStateUnregister);
                require(trtmcAttentionStateRegister(tensor.data(),
                                                    std::size_t(c.heads) * c.capacity * c.dim * 2),
                        "cache registration failed");
                cache_registrations_.push_back(std::move(registration));
                // V3 declares local aliasing. The runtime identity guard rejects
                // compiler-inserted cache copies; this is explicitly a weaker
                // build-time guarantee than future requireOutputAlias support.
                module_->bind_external(output, tensor.data());
                if (prefill_)
                    prefill_->bind_external(output, tensor.data());
#else
                throw std::runtime_error("rebuild with TRTMC_LLAMA_ATTENTION_STATE_PLUGINS=ON");
#endif
            }
            // Native v1 bindings are propagated only through TensorRT's engine
            // alias metadata. Plugin v2 additionally checks the actual addresses
            // at enqueue; equal external bindings alone are insufficient.
            require(module_->device_ptr(output) == tensor.data(),
                    "required cache allocation binding is missing: " + output);
            if (prefill_)
                require(prefill_->device_ptr(output) == tensor.data(),
                        "prefill cache allocation binding is missing: " + output);
        }
    }
    module_->sync();
}

StepResult Engine::run(Phase phase, const std::vector<std::int32_t>& tokens, int start,
                       const std::vector<std::int32_t>& parents, bool all_logits,
                       const std::vector<std::uint16_t>& target_features,
                       const std::vector<std::uint16_t>& draft_features) {
    const auto& c = contract_;
    const int rows = static_cast<int>(tokens.size());
    require(rows > 0 && rows <= c.query_limit(phase) && start >= 0 && start <= c.capacity - rows,
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
        {"key_value_lengths", {&length, {1}, DType::kInt32}},
    };
    std::vector<std::int32_t> key_slots(rows), value_slots(rows);
    if (c.page_size) {
        for (int row = 0; row < rows; ++row) {
            key_slots[row] = layout_.slot(start + row, true);
            value_slots[row] = layout_.slot(start + row, false);
        }
        inputs["key_write_slots"] = {key_slots.data(), {1, rows}, DType::kInt32};
        inputs["value_write_slots"] = {value_slots.data(), {1, rows}, DType::kInt32};
        inputs["key_pages"] = {const_cast<std::int32_t*>(layout_.pages(true).data()),
                               {1, c.capacity / c.page_size},
                               DType::kInt32};
        inputs["value_pages"] = {const_cast<std::int32_t*>(layout_.pages(false).data()),
                                 {1, c.capacity / c.page_size},
                                 DType::kInt32};
    } else {
        inputs["cache_write_indices"] = {&write_start, {1}, DType::kInt32};
    }
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
    auto& module = phase == Phase::kPrefill && prefill_ ? prefill_ : module_;
    if (!c.profiles.empty()) {
        const auto& bounds = c.profiles[phase == Phase::kPrefill ? 0 : 1].logits;
        require(selected.size() >= std::size_t(bounds[0]) &&
                    selected.size() <= std::size_t(bounds[2]),
                "selected logits exceed phase profile");
    }
    module->forward_async(inputs);
    StepResult result;
    result.logits.resize(selected.size() * c.vocab);
    result.features.resize(static_cast<std::size_t>(rows) * (c.draft ? c.hidden : c.feature_width));
    checked(cudaMemcpyAsync(result.logits.data(), module->device_ptr("logits"),
                            result.logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                            module->stream()));
    checked(cudaMemcpyAsync(result.features.data(), module->device_ptr("features"),
                            result.features.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                            module->stream()));
    module->sync();
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
        const bool key = storage == &keys_;
        for (auto& cache : *storage) {
            for (std::size_t row = 0; row < rows.size(); ++row) {
                checked(cudaMemcpy2DAsync(
                    static_cast<char*>(scratch.data()) + row * width, rows.size() * width,
                    static_cast<char*>(cache.data()) +
                        layout_.offset(start + rows[row], key, c.heads, c.dim) * 2,
                    layout_.head_stride() * width, width, c.heads, cudaMemcpyDeviceToDevice,
                    module_->stream()));
            }
            for (std::size_t row = 0; row < rows.size(); ++row)
                checked(cudaMemcpy2DAsync(static_cast<char*>(cache.data()) +
                                              layout_.offset(start + row, key, c.heads, c.dim) * 2,
                                          layout_.head_stride() * width,
                                          static_cast<char*>(scratch.data()) + row * width,
                                          rows.size() * width, width, c.heads,
                                          cudaMemcpyDeviceToDevice, module_->stream()));
        }
    }
    module_->sync();
}

void Engine::reset() {
    // No byte clearing: subsequent inputs expose only the current valid prefix.
    module_->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
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
