/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernels.h"
#include "plugin_api.h"

#include <NvInfer.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <new>
#include <unordered_map>

namespace trtmc::llama::attention_state {
namespace {
using namespace nvinfer1;
constexpr char kNamespace[] = "trtmc.llama.attention_state";
std::mutex allocation_mutex;
std::unordered_map<const void*, std::size_t> allocations;

bool registered(const void* address, std::size_t bytes) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    const auto it = allocations.find(address);
    return it != allocations.end() && it->second == bytes;
}

class Plugin : public IPluginV3,
               public IPluginV3OneCore,
               public IPluginV3OneBuildV2,
               public IPluginV3OneRuntime {
  public:
    Plugin(bool write, bool runtime) : write_(write), runtime_(runtime) {}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE:
            return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:
            return static_cast<IPluginV3OneBuildV2*>(this);
        case PluginCapabilityType::kRUNTIME:
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new (std::nothrow) Plugin(write_, runtime_); }
    const char* getPluginName() const noexcept override {
        return write_ ? "IndexedKVCacheUpdate" : "PagedAttention";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getAliasedInput(int32_t output) noexcept override {
        return write_ && output == 0 ? 0 : -1;
    }
    int32_t getOutputDataTypes(DataType* outputs, int32_t nout, const DataType* inputs,
                               int32_t nin) const noexcept override {
        if (nout != 1 || nin != input_count())
            return -1;
        outputs[0] = inputs[0];
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* inputs, int32_t nin, const DimsExprs*, int32_t nshape,
                            DimsExprs* outputs, int32_t nout, IExprBuilder&) noexcept override {
        if (nin != input_count() || nout != 1 || nshape != 0)
            return -1;
        outputs[0] = inputs[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* tensors, int32_t nin,
                                   int32_t nout) noexcept override {
        if (nin != input_count() || nout != 1 || pos < 0 || pos > nin)
            return false;
        const auto type = pos == nin || pos < (write_ ? 2 : 3) ? DataType::kHALF
                          : (!write_ && pos == 6)              ? DataType::kBOOL
                                                               : DataType::kINT32;
        return tensors[pos].desc.format == TensorFormat::kLINEAR && tensors[pos].desc.type == type;
    }
    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t nin,
                            const DynamicPluginTensorDesc*, int32_t nout) noexcept override {
        return nin == input_count() && nout == 1 ? 0 : -1;
    }
    int32_t onShapeChange(const PluginTensorDesc* in, int32_t nin, const PluginTensorDesc*,
                          int32_t nout) noexcept override {
        if (nin != input_count() || nout != 1 || in[0].dims.nbDims != 4 || in[1].dims.nbDims != 4)
            return -1;
        const auto& a = in[0].dims;
        const auto& b = in[1].dims;
        for (int i = 0; i < 4; ++i)
            if (a.d[i] <= 0 || b.d[i] <= 0)
                return -1;
        if (write_) {
            const auto& slots = in[2].dims;
            return slots.nbDims == 2 && slots.d[0] == b.d[0] && slots.d[1] == b.d[2] &&
                           a.d[1] == b.d[1] && a.d[3] == b.d[3]
                       ? 0
                       : -1;
        }
        const auto& v = in[2].dims;
        const auto& kt = in[3].dims;
        const auto& vt = in[4].dims;
        const auto& len = in[5].dims;
        const auto& mask = in[6].dims;
        if (v.nbDims != 4 || kt.nbDims != 2 || vt.nbDims != 2 || len.nbDims != 1 ||
            mask.nbDims != 4)
            return -1;
        const int64_t capacity = int64_t(kt.d[1]) * b.d[2];
        return v.d[0] > 0 && v.d[1] == b.d[1] && v.d[2] == b.d[2] && v.d[3] == b.d[3] &&
                       a.d[1] % b.d[1] == 0 && a.d[3] == b.d[3] && a.d[3] <= 256 &&
                       kt.d[0] == a.d[0] && vt.d[0] == a.d[0] && vt.d[1] == kt.d[1] &&
                       capacity > 0 && capacity <= 4096 && len.d[0] == a.d[0] &&
                       mask.d[0] == a.d[0] && (mask.d[1] == 1 || mask.d[1] == a.d[1]) &&
                       mask.d[2] == a.d[2] && mask.d[3] == capacity
                   ? 0
                   : -1;
    }
    int32_t enqueue(const PluginTensorDesc* in, const PluginTensorDesc*, const void* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        const auto& a = in[0].dims;
        const auto& b = in[1].dims;
        if (write_) {
            const std::size_t bytes = std::size_t(a.d[0]) * a.d[1] * a.d[2] * a.d[3] * 2;
            const std::size_t update_bytes = std::size_t(b.d[0]) * b.d[1] * b.d[2] * b.d[3] * 2;
            const auto cache_begin = reinterpret_cast<std::uintptr_t>(inputs[0]);
            const auto update_begin = reinterpret_cast<std::uintptr_t>(inputs[1]);
            if (update_begin < cache_begin + bytes && cache_begin < update_begin + update_bytes) {
                std::fputs("MC indexed KV update: update/cache overlap is unsupported\n", stderr);
                return -1;
            }
            if (inputs[0] != outputs[0] || (runtime_ && !registered(inputs[0], bytes))) {
                std::fputs("MC indexed KV update: external cache identity was not preserved\n",
                           stderr);
                return -1;
            }
            return update(outputs[0], inputs[1], static_cast<const int*>(inputs[2]), b.d[0], b.d[1],
                          b.d[2], a.d[0], a.d[2], a.d[3], stream) == cudaSuccess
                       ? 0
                       : -1;
        }
        return attention(inputs[0], inputs[1], inputs[2], static_cast<const int*>(inputs[3]),
                         static_cast<const int*>(inputs[4]), static_cast<const int*>(inputs[5]),
                         static_cast<const bool*>(inputs[6]), outputs[0], a.d[0], a.d[1], b.d[1],
                         a.d[2], b.d[0], in[2].dims.d[0], b.d[2], in[3].dims.d[1], a.d[3],
                         in[6].dims.d[1], stream) == cudaSuccess
                   ? 0
                   : -1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    const PluginFieldCollection* getFieldsToSerialize() noexcept override { return &fields_; }

  private:
    int input_count() const { return write_ ? 3 : 7; }
    bool write_, runtime_;
    PluginFieldCollection fields_{0, nullptr};
};

class Creator : public IPluginCreatorV3One {
  public:
    explicit Creator(bool write) : write_(write) {}
    const char* getPluginName() const noexcept override {
        return write_ ? "IndexedKVCacheUpdate" : "PagedAttention";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fields,
                            TensorRTPhase phase) noexcept override {
        if (!fields || fields->nbFields != 0)
            return nullptr;
        return new (std::nothrow) Plugin(write_, phase == TensorRTPhase::kRUNTIME);
    }

  private:
    bool write_;
    PluginFieldCollection fields_{0, nullptr};
};
} // namespace
} // namespace trtmc::llama::attention_state

extern "C" bool trtmcAttentionStateInit() noexcept {
    using namespace trtmc::llama::attention_state;
    static Creator writer(true), reader(false);
    static const bool initialized = getPluginRegistry()->registerCreator(writer, kNamespace) &&
                                    getPluginRegistry()->registerCreator(reader, kNamespace);
    return initialized;
}
extern "C" bool trtmcAttentionStateRegister(void* address, std::size_t bytes) noexcept {
    using namespace trtmc::llama::attention_state;
    if (!address || !bytes)
        return false;
    try {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        return allocations.emplace(address, bytes).second;
    } catch (...) {
        return false;
    }
}
extern "C" void trtmcAttentionStateUnregister(void* address) noexcept {
    using namespace trtmc::llama::attention_state;
    std::lock_guard<std::mutex> lock(allocation_mutex);
    allocations.erase(address);
}
