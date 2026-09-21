/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Independent reference harness; excluded from Model-Connect dependencies.
#include "common/trtUtils.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
void synchronize() {
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 8)
            throw std::invalid_argument("usage: edge_benchmark ENGINE_DIR INPUT_IDS COUNT WARMUP "
                                        "REPEATS CUDA_GRAPH OUTPUT_JSON");
        const int count = std::stoi(argv[3]), warmup = std::stoi(argv[4]),
                  repeats = std::stoi(argv[5]);
        const bool cuda_graph = std::stoi(argv[6]) != 0;
        if (count < 2 || warmup < 1 || repeats < 1)
            throw std::invalid_argument("require count >= 2, warmup >= 1 and repeats >= 1");
        auto plugins = trt_edgellm::loadEdgellmPluginLib();
        if (!plugins)
            throw std::runtime_error("Edge-LLM plugin load failed");
        std::ifstream input(argv[2]);
        if (!input)
            throw std::runtime_error("cannot read input fixture");
        nlohmann::json fixture;
        input >> fixture;
        const auto ids = fixture.get<std::vector<int32_t>>();
        cudaStream_t stream{};
        if (cudaStreamCreate(&stream) != cudaSuccess)
            throw std::runtime_error("stream creation failed");
        nlohmann::json report{{"input_tokens", ids.size()}, {"output_tokens", count},
                              {"warmup_per_mode", warmup},  {"repeats_per_mode", repeats},
                              {"cuda_graph", cuda_graph},   {"samples", nlohmann::json::array()}};
        {
            trt_edgellm::rt::SpecDecodeDraftingConfig drafting;
            drafting.draftingTopK = 1;
            drafting.draftingStep = 4;
            drafting.verifySize = 5;
            trt_edgellm::rt::LLMInferenceRuntime runtime(argv[1], "", {}, drafting, stream);
            if (cuda_graph && !runtime.captureDecodingCUDAGraph(stream))
                throw std::runtime_error("Edge CUDA graph capture failed");
            trt_edgellm::rt::LLMGenerationRequest request{};
            request.requests.resize(1);
            request.requests[0].messages.push_back(
                {"user", {{"text", "Pre-tokenized benchmark fixture."}}});
            request.preTokenizedInputIds = {ids};
            request.temperature = 0;
            request.topK = 1;
            request.topP = 1;
            request.maxGenerateLength = count;
            request.applyChatTemplate = false;
            request.contextCacheLookupPolicy = trt_edgellm::rt::ContextCacheLookupPolicy::kBypass;
            std::vector<int32_t> expected;
            trt_edgellm::setProfilingEnabled(true);
            for (int iteration = -warmup; iteration < repeats; ++iteration) {
                for (int offset = 0; offset < 2; ++offset) {
                    const bool speculative = (iteration + warmup + offset) % 2 != 0;
                    request.disableSpecDecode = !speculative;
                    const auto before = runtime.getSpecDecodeGenerationMetrics();
                    nlohmann::json before_stages = nlohmann::json::object();
                    for (const auto& [name, data] : trt_edgellm::gTimer.getAllTimingData())
                        before_stages[name] = data.getTotalGpuTimeMs();
                    trt_edgellm::rt::LLMGenerationResponse response;
                    synchronize();
                    const auto start = std::chrono::steady_clock::now();
                    if (!runtime.handleRequest(request, response, stream))
                        throw std::runtime_error("Edge-LLM request failed");
                    synchronize();
                    const double wall_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - start)
                                               .count();
                    if (expected.empty())
                        expected = response.outputIds.at(0);
                    if (response.outputIds.at(0).size() != static_cast<std::size_t>(count) ||
                        response.outputIds.at(0) != expected ||
                        response.inputTokenCounts !=
                            std::vector<int32_t>{static_cast<int32_t>(ids.size())})
                        throw std::runtime_error("Edge benchmark output mismatch");
                    const auto& after = runtime.getSpecDecodeGenerationMetrics();
                    const char* mode = speculative ? "eagle3_chain" : "autoregressive";
                    if (iteration >= 0) {
                        nlohmann::json stages;
                        for (const auto& [name, data] : trt_edgellm::gTimer.getAllTimingData())
                            stages[name] =
                                data.getTotalGpuTimeMs() - before_stages.value(name, 0.0);
                        report["samples"].push_back({
                            {"iteration", iteration},
                            {"mode", mode},
                            {"wall_ms", wall_ms},
                            {"verification_rounds", after.totalIterations - before.totalIterations},
                            {"speculative_generated_tokens",
                             after.totalGeneratedTokens - before.totalGeneratedTokens},
                            {"stage_cuda_event_ms", stages},
                        });
                    }
                    std::cout << (iteration < 0 ? "warmup " : "sample ") << iteration << ' ' << mode
                              << " wall_ms=" << wall_ms << std::endl;
                }
            }
            report["output_ids"] = expected;
            report["all_outputs_equal"] = true;
        }
        cudaStreamDestroy(stream);
        std::ofstream output(argv[7]);
        if (!output)
            throw std::runtime_error("cannot open benchmark output");
        output << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
