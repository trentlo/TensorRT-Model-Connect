# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family adapter for additions A/B of the attention-state minimal API.

These are prototype methods, not additions to the installed TensorRT bindings.
TensorRT still sees ordinary tensors, V3 plugins and declared mutation aliases.
"""

import ctypes
from enum import IntEnum
from pathlib import Path

import numpy as np
import tensorrt as trt

_libraries = {}
NAMESPACE = "trtmc.llama.attention_state"


class KVCacheMode(IntEnum):
    INDEXED = 1


def load_plugins(path):
    path = str(Path(path).resolve(strict=True))
    if path not in _libraries:
        library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        library.trtmcAttentionStateInit.restype = ctypes.c_bool
        library.trtmcAttentionStateRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.trtmcAttentionStateRegister.restype = ctypes.c_bool
        library.trtmcAttentionStateUnregister.argtypes = [ctypes.c_void_p]
        if not library.trtmcAttentionStateInit():
            raise RuntimeError("could not register MC attention-state plugins")
        _libraries[path] = library
    return _libraries[path]


def _plugin(network, name, inputs):
    creator = trt.get_plugin_registry().get_creator(name, "1", NAMESPACE)
    if creator is None:
        raise RuntimeError("load the standalone MC attention-state library before building")
    instance = creator.create_plugin(name, trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD)
    return network.add_plugin_v3(inputs, [], instance)


class AttentionStateGraph:
    def __init__(self, network, build_config):
        self.network = network
        self._updated = {}
        build_config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)

    def add_kv_cache_update(self, cache, update, write_indices, mode):
        if mode != KVCacheMode.INDEXED:
            raise ValueError("plugin prototype implements INDEXED updates only")
        if cache.name in self._updated or not cache.is_network_input:
            raise ValueError("require one update per external cache input")
        if (len(cache.shape) != 4 or len(update.shape) != 4 or len(write_indices.shape) != 2
                or cache.dtype != trt.float16 or update.dtype != trt.float16
                or write_indices.dtype != trt.int32
                or cache.shape[1] != update.shape[1] or cache.shape[3] != update.shape[3]):
            raise ValueError("invalid indexed KV update operands")
        layer = _plugin(self.network, "IndexedKVCacheUpdate", [cache, update, write_indices])
        self._updated[cache.name] = layer.get_output(0)
        return layer

    def validate_effects(self):
        """Reject old-state readers and unobservable mutations before compilation."""
        readers = dict.fromkeys(self._updated, 0)
        for index in range(self.network.num_layers):
            layer = self.network.get_layer(index)
            for operand in range(layer.num_inputs):
                tensor = layer.get_input(operand)
                if tensor is not None and tensor.name in readers:
                    readers[tensor.name] += 1
        for name, present in self._updated.items():
            if readers[name] != 1 or not present.is_network_output:
                raise ValueError("state requires a single writer, no old-state readers, and a marked output")

    def add_attention_v2(self, query, keys, values, normalization, causal_kind):
        if normalization != trt.AttentionNormalizationOp.SOFTMAX or causal_kind != trt.CausalMaskKind.NONE:
            raise ValueError("prototype requires softmax and an explicit logical visibility mask")
        return Attention(self.network, query, keys, values)


class Attention:
    """Configure tensor operands before requesting the output, as with IAttention."""

    def __init__(self, network, query, keys, values):
        self.network, self.query, self.keys, self.values = network, query, keys, values
        self.key_value_lengths = None
        self.mask = None
        self._key_pages = self._value_pages = self._layer = None

    def set_key_value_page_tables(self, key_pages, value_pages):
        if self._layer is not None:
            raise ValueError("attention has already been materialized")
        if (key_pages is None) != (value_pages is None):
            raise ValueError("K/V page tables must both be present or both be absent")
        self._key_pages, self._value_pages = key_pages, value_pages
        return True

    def get_key_page_table(self):
        return self._key_pages

    def get_value_page_table(self):
        return self._value_pages

    def get_output(self, index):
        if index != 0:
            raise IndexError("attention has one output")
        if self._layer is None:
            if self.mask is None or self.key_value_lengths is None:
                raise ValueError("explicit logical mask and KV lengths are required")
            if self.mask.dtype != trt.bool:
                raise ValueError("prototype attention mask must be boolean")
            key_pages, value_pages = self._key_pages, self._value_pages
            if key_pages is None:
                batch = self.query.shape[0]
                if batch <= 0 or self.keys.shape[0] != batch or self.values.shape[0] != batch:
                    raise ValueError("dense attention requires a static matching batch")
                table = np.arange(batch, dtype=np.int32).reshape(batch, 1)
                key_pages = value_pages = self.network.add_constant(table.shape, table).get_output(0)
            self._layer = _plugin(self.network, "PagedAttention", [
                self.query, self.keys, self.values, key_pages, value_pages,
                self.key_value_lengths, self.mask,
            ])
        return self._layer.get_output(0)
