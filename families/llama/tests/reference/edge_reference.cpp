/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Independent reference executable, built only in a separate Edge-LLM checkout.
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::runtime_error(
                "usage: edge_reference ENGINE_DIR INPUT_IDS OUTPUT_JSON [OUTPUT_COUNT]");
        const int output_count = argc == 5 ? std::stoi(argv[4]) : 101;
        if (output_count < 1)
            throw std::runtime_error("output count must be positive");
        auto plugins = trt_edgellm::loadEdgellmPluginLib();
        if (!plugins)
            throw std::runtime_error("plugin load failed");
        std::ifstream input(argv[2]);
        nlohmann::json document;
        input >> document;
        const auto ids = document.get<std::vector<int32_t>>();
        cudaStream_t stream{};
        if (cudaStreamCreate(&stream) != cudaSuccess)
            throw std::runtime_error("stream creation failed");
        nlohmann::json report;
        {
            trt_edgellm::rt::SpecDecodeDraftingConfig drafting;
            drafting.draftingTopK = 1;
            drafting.draftingStep = 4;
            drafting.verifySize = 5;
            trt_edgellm::rt::LLMInferenceRuntime runtime(argv[1], "", {}, drafting, stream);
            trt_edgellm::rt::LLMGenerationRequest request{};
            request.requests.resize(1);
            // Edge still validates message structure before honoring explicit
            // token IDs. This text is not tokenized or used as model input.
            request.requests[0].messages.push_back(
                {"user", {{"text", "Pre-tokenized validation fixture."}}});
            request.preTokenizedInputIds = {ids};
            request.temperature = 0.0F;
            request.topK = 1;
            request.topP = 1.0F;
            request.maxGenerateLength = output_count;
            request.applyChatTemplate = false;
            request.contextCacheLookupPolicy = trt_edgellm::rt::ContextCacheLookupPolicy::kBypass;
            for (bool speculative : {false, true}) {
                request.disableSpecDecode = !speculative;
                trt_edgellm::rt::LLMGenerationResponse response;
                const auto start = std::chrono::steady_clock::now();
                if (!runtime.handleRequest(request, response, stream))
                    throw std::runtime_error("Edge-LLM request failed");
                const auto ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
                report[speculative ? "eagle3_ids" : "autoregressive_ids"] =
                    response.outputIds.at(0);
                report[speculative ? "eagle3_ms" : "autoregressive_ms"] = ms;
                report["input_token_counts"] = response.inputTokenCounts;
                if (response.outputIds.at(0).size() != static_cast<std::size_t>(output_count))
                    throw std::runtime_error("Edge-LLM stopped before the requested output count");
            }
            report["strategy"] = runtime.getSpeculativeDecodingStrategyName();
        }
        cudaStreamDestroy(stream);
        report["tokens_equal"] = report["autoregressive_ids"] == report["eagle3_ids"];
        std::ofstream output(argv[3]);
        output << report.dump(2) << '\n';
        return report["tokens_equal"].get<bool>() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
