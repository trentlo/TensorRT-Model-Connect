# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone GPU contract probe; no checkpoint or Edge-LLM dependency.

Run: python -m families.llama.tests.attention_state_gpu PATH_TO_PLUGIN_SO
"""

import json
import sys

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda

from ..speculative.attention_state import AttentionStateGraph, KVCacheMode, load_plugins


def checked(result):
    if result[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(str(result))
    return result[1] if len(result) == 2 else result[1:]


def reference(query, keys, values, kp, vp, lengths, mask, slots_k, slots_v, new_k, new_v):
    keys, values = keys.copy(), values.copy()
    page_size = keys.shape[2]
    for pool, updates, slots in ((keys, new_k, slots_k), (values, new_v, slots_v)):
        for b, q in np.ndindex(slots.shape):
            s = slots[b, q]
            if s >= 0:
                pool[s // page_size, :, s % page_size, :] = updates[b, :, q, :]
    output = np.zeros_like(query)
    for b, h, q in np.ndindex(query.shape[:3]):
        kh = h // (query.shape[1] // keys.shape[1])
        mh = 0 if mask.shape[1] == 1 else h
        visible = np.flatnonzero(mask[b, mh, q, :lengths[b]])
        if not len(visible):
            continue
        k = np.stack([keys[kp[b, j // page_size], kh, j % page_size] for j in visible])
        v = np.stack([values[vp[b, j // page_size], kh, j % page_size] for j in visible])
        scores = (k.astype(np.float32) @ query[b, h, q].astype(np.float32)).astype(np.float16).astype(np.float32)
        probabilities = np.exp(scores - scores.max())
        probabilities = (probabilities / probabilities.sum()).astype(np.float16)
        output[b, h, q] = probabilities.astype(np.float32) @ v.astype(np.float32)
    return keys, values, output


def xqa_arrays(case, random, rng):
    """Exercise the demanded XQA geometry with independent pool maps and full masks."""
    if case == "prefill1024":
        queries, capacity = 1024, 1088
        arrays = {
            "q": random((1, 4, queries, 128)),
            "k": random((17, 1, 64, 128)), "v": random((17, 1, 64, 128)),
            "new_k": random((1, 1, queries, 128)), "new_v": random((1, 1, queries, 128)),
            "ks": np.full((1, queries), -1, np.int32),
            "vs": np.full((1, queries), -1, np.int32),
            "kp": np.array([list(reversed(range(16))) + [-1]], np.int32),
            "vp": np.array([list(range(16)) + [-1]], np.int32),
            "length": np.array([queries], np.int32),
            "mask": (np.arange(capacity)[None, None, None, :] <=
                     np.arange(queries)[None, None, :, None]),
        }
        arrays["mask"][:, :, -1] = False
        arrays["k"][16] = np.nan
        arrays["v"][16] = np.nan
        return arrays
    queries = {"decode": 1, "chain": 5, "tree": 9, "prefill": 64,
               "masked_nan": 9, "tail_nan": 9, "short_history": 64,
               "empty_history": 5}[case]
    batch, heads, kv_heads, pages, page_size, dim = 2, 8, 2, 5, 64, 128
    capacity = pages * page_size
    lengths = np.array([141, 129], np.int32)
    kp = np.array([[2, 0, 3, -1, -1], [0, 3, 2, -1, -1]], np.int32)
    vp = np.array([[1, 3, 0, -1, -1], [3, 0, 1, -1, -1]], np.int32)
    if case == "tail_nan":
        batch = 1
        lengths, kp, vp = lengths[:1], kp[:1], vp[:1]
    arrays = {
        "q": random((batch, heads, queries, dim)),
        "k": random((pages, kv_heads, page_size, dim)),
        "v": random((pages, kv_heads, page_size, dim)),
        "new_k": random((batch, kv_heads, queries, dim)),
        "new_v": random((batch, kv_heads, queries, dim)),
        "ks": np.full((batch, queries), -1, np.int32),
        "vs": np.full((batch, queries), -1, np.int32),
        "kp": kp, "vp": vp, "length": lengths,
        "mask": np.zeros((batch, heads, queries, capacity), np.bool_),
    }
    # Shared physical pools are read-only here; the original small fixture
    # separately verifies writes, aliases and overlapping page boundaries.
    for b in range(batch):
        prefix = int(lengths[b]) - queries
        for q in range(queries):
            arrays["mask"][b, :, q, :prefix] = True
            if case in ("tree", "masked_nan"):
                row = q
                while row >= 0:
                    arrays["mask"][b, :, q, prefix + row] = True
                    row = [-1, 0, 0, 1, 1, 3, 3, 5, 5][row]
            else:
                arrays["mask"][b, :, q, prefix:prefix + q + 1] = True
        # Prefix holes and head-specific visibility must not be mistaken for
        # Edge's implicit all-visible prefix plus packed candidate-only mask.
        arrays["mask"][b, 1::2, :, 3:17] = False
    arrays["mask"][:, -1, -1] = False
    arrays["k"][4] = np.nan
    arrays["v"][4] = np.nan
    if case == "masked_nan":
        # Poison a mapped row, hidden from every query that could read it.
        for b in range(batch):
            arrays["k"][kp[b, 0], :, 7] = np.nan
            arrays["v"][vp[b, 0], :, 7] = np.nan
        arrays["mask"][:, :, :, 7::page_size] = False
    if case == "empty_history":
        arrays["length"][:] = 0
        arrays["k"][:] = np.nan
        arrays["v"][:] = np.nan
    if case == "tail_nan":
        for j in range(int(lengths[0]), 3 * page_size):
            arrays["k"][kp[0, j // page_size], :, j % page_size] = np.nan
            arrays["v"][vp[0, j // page_size], :, j % page_size] = np.nan
    if case == "short_history":
        arrays["length"][:] = 1
    return arrays


def run(path, case="small"):
    library = load_plugins(path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    state = AttentionStateGraph(network, config)
    rng = np.random.default_rng(24)
    def random(shape):
        return rng.normal(0, 0.2, shape).astype(np.float16)
    batch, heads, kv_heads, queries, pages, page_size, dim = 2, 4, 2, 7, 5, 4, 8
    capacity = pages * page_size
    arrays = {
        "q": random((batch, heads, queries, dim)),
        "k": random((pages, kv_heads, page_size, dim)),
        "v": random((pages, kv_heads, page_size, dim)),
        "new_k": random((batch, kv_heads, queries, dim)),
        "new_v": random((batch, kv_heads, queries, dim)),
        "ks": np.array([[0, 4, 8, 12, 1, -1, 17], [2, 6, 10, 14, 3, 7, 11]], np.int32),
        "vs": np.array([[12, 8, 4, 0, 13, -1, 17], [14, 10, 6, 2, 15, 11, 7]], np.int32),
        "kp": np.array([[0, 1, 2, -1, -1], [2, 0, 1, -1, -1]], np.int32),
        "vp": np.array([[3, 2, 1, -1, -1], [1, 3, 2, -1, -1]], np.int32),
        "length": np.array([12, 9], np.int32),
        "mask": rng.random((batch, 1, queries, capacity)) > 0.4,
    }
    arrays["k"][4] = np.nan
    arrays["v"][4] = np.nan
    arrays["mask"][:, :, -1, :] = False  # Defined empty-row behavior.
    if case != "small":
        arrays = xqa_arrays(case, random, rng)
    types = {np.dtype("float16"): trt.float16, np.dtype("int32"): trt.int32, np.dtype("bool"): trt.bool}
    tensors = {name: network.add_input(name, types[value.dtype], value.shape)
               for name, value in arrays.items()}
    uk = state.add_kv_cache_update(tensors["k"], tensors["new_k"], tensors["ks"], KVCacheMode.INDEXED).get_output(0)
    uv = state.add_kv_cache_update(tensors["v"], tensors["new_v"], tensors["vs"], KVCacheMode.INDEXED).get_output(0)
    attention = state.add_attention_v2(tensors["q"], uk, uv, trt.AttentionNormalizationOp.SOFTMAX, trt.CausalMaskKind.NONE)
    attention.set_key_value_page_tables(tensors["kp"], tensors["vp"])
    attention.key_value_lengths = tensors["length"]
    attention.mask = tensors["mask"]
    for name, tensor in (("present_k", uk), ("present_v", uv), ("output", attention.get_output(0))):
        tensor.name = name
        network.mark_output(tensor)
    state.validate_effects()
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, "plugin engine did not build"
    network.add_identity(tensors["k"])
    try:
        state.validate_effects()
    except ValueError:
        pass
    else:
        raise AssertionError("old-state reader was accepted")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    stream = checked(cuda.cudaStreamCreate())
    pointers = {}
    registered = []
    try:
        for name, value in arrays.items():
            pointer = checked(cuda.cudaMalloc(value.nbytes))
            pointers[name] = pointer
            checked(cuda.cudaMemcpy(pointer, value.ctypes.data, value.nbytes, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
            assert context.set_tensor_address(name, pointer)
        for name in ("k", "v"):
            assert library.trtmcAttentionStateRegister(pointers[name], arrays[name].nbytes)
            registered.append(pointers[name])
            assert context.set_tensor_address("present_" + name, pointers[name])
        pointers["output"] = checked(cuda.cudaMalloc(arrays["q"].nbytes))
        context.set_tensor_address("output", pointers["output"])
        assert context.execute_async_v3(stream), "plugin enqueue failed"
        checked(cuda.cudaStreamSynchronize(stream))
        expected = reference(arrays["q"], arrays["k"], arrays["v"], arrays["kp"], arrays["vp"],
                             arrays["length"], arrays["mask"], arrays["ks"], arrays["vs"],
                             arrays["new_k"], arrays["new_v"])
        actual = []
        for name, wanted in zip(("k", "v", "output"), expected):
            found = np.empty_like(wanted)
            checked(cuda.cudaMemcpy(found.ctypes.data, pointers[name], found.nbytes, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            actual.append(found)
        # Bitwise equality covers untouched bytes (including NaNs), skip rows,
        # cross-page writes, and independent K/V slot maps.
        np.testing.assert_array_equal(actual[0].view(np.uint16), expected[0].view(np.uint16))
        np.testing.assert_array_equal(actual[1].view(np.uint16), expected[1].view(np.uint16))
        np.testing.assert_allclose(actual[2], expected[2], rtol=0.005, atol=0.0005)
        empty = ~arrays["mask"].any(axis=-1)
        if empty.shape[1] == 1:
            empty = np.repeat(empty, arrays["q"].shape[1], axis=1)
        assert np.all(actual[2][empty] == 0)
        aliases = {name: engine.get_aliased_input_tensor(name) for name in ("present_k", "present_v")}
        # A private compiler allocation must not be accepted as persistent state.
        # Removing ownership is a deterministic way to exercise that guard.
        library.trtmcAttentionStateUnregister(pointers["k"])
        registered.remove(pointers["k"])
        assert not context.execute_async_v3(stream), "unregistered state was accepted"
        checked(cuda.cudaStreamSynchronize(stream))
        checks = ["untouched_bytes", "independent_kv_maps", "gqa", "mask", "length", "empty_row",
                  "stale_nan", "unregistered_state_rejected", "old_state_reader_rejected"]
        if case == "small":
            checks += ["indexed_write", "skip", "cross_page"]
        elif case == "prefill1024":
            checks += ["xqa_geometry", "broadcast_causal_mask", "1024_query_rows"]
        else:
            checks += ["xqa_geometry", "per_head_mask", "prefix_holes"]
        print(json.dumps({"passed": True, "case": case, "engine_aliases": aliases,
                          "max_attention_error": float(np.max(np.abs(actual[2] - expected[2]))),
                          "cases": checks}))
    finally:
        checked(cuda.cudaStreamSynchronize(stream))
        for pointer in registered:
            library.trtmcAttentionStateUnregister(pointer)
        for pointer in pointers.values():
            checked(cuda.cudaFree(pointer))
        checked(cuda.cudaStreamDestroy(stream))


if __name__ == "__main__":
    for case in ("small", "decode", "chain", "tree", "prefill", "prefill1024", "masked_nan", "tail_nan",
                 "short_history", "empty_history"):
        run(sys.argv[1], case)
