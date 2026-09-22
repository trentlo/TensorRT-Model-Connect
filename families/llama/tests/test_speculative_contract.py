# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import replace

from ..speculative.contract import EngineContract, ExecutionProfile, tree_visibility


def test_siblings_are_hidden_and_ancestors_visible():
    mask = tree_visibility([-1, 0, 0, 1, 2])
    assert mask[3] == [1, 1, 0, 1, 0]
    assert mask[4] == [1, 0, 1, 0, 1]


@pytest.mark.parametrize("parents", [[], [0], [-1, 1], [-1, -1], [-1, 2, 0]])
def test_invalid_tree_rejected(parents):
    with pytest.raises(ValueError):
        tree_visibility(parents)


def test_contract_rejects_incompatible_precision_and_capacity():
    fields = dict(role="target", layers=32, hidden_size=4096, kv_heads=8,
                  head_dim=128, vocab_size=128256, capacity=2048, feature_width=12288)
    with pytest.raises(ValueError):
        EngineContract(**fields, precision="bf16")
    with pytest.raises(ValueError):
        EngineContract(**fields, max_query=2049)


def test_paged_contract_is_explicit_and_rejects_incompatible_geometry():
    contract = EngineContract(
        "target", 32, 4096, 8, 128, 128256, 2048, feature_width=12288,
        version=2, attention_backend="plugin", page_size=64,
        cache_layout="pages_heads_slots_dim", cache_update="aliased_indexed_write",
        alias_contract="plugin_local_alias_runtime_identity_guard")
    assert contract.to_dict()["page_size"] == 64
    for changes in ({"page_size": 0}, {"page_size": 63}, {"capacity": 8192},
                    {"head_dim": 512}, {"alias_contract": "engine_required_alias"},
                    {"version": 1}, {"attention_backend": "primitives"}):
        with pytest.raises(ValueError):
            replace(contract, **changes)


def test_phase_profiles_separate_query_and_selected_rows():
    profiles = (ExecutionProfile("prefill", (1, 1024, 1024), (1, 1, 1)),
                ExecutionProfile("decode", (1, 5, 9), (1, 5, 9)))
    contract = EngineContract("target", 32, 4096, 8, 128, 128256, 2048,
                              max_query=1024, feature_width=12288, execution_profiles=profiles)
    assert contract.profile_shapes("attention_mask", (-1, 2048)) == [
        ((1, 2048), (1024, 2048), (1024, 2048)), ((1, 2048), (5, 2048), (9, 2048))]
    assert contract.profile_shapes("logits_indices", (-1,)) == [
        ((1,), (1,), (1,)), ((1,), (5,), (9,))]
    assert contract.profile_shapes("key_write_slots", (1, -1))[1] == ((1, 1), (1, 5), (1, 9))
    for changes in ({"execution_profiles": profiles[::-1]}, {"max_query": 2048},
                    {"role": "draft"}, {"execution_profiles": profiles[:1]}):
        with pytest.raises(ValueError):
            replace(contract, **changes)
    legacy = replace(contract, execution_profiles=(), max_query=64)
    assert legacy.profile_shapes("logits_indices", (-1,)) == [((1,), (16,), (64,))]


@pytest.mark.parametrize("query,logits", [((0, 5, 9), (1, 1, 1)), ((1, 9, 5), (1, 1, 1)),
                                         ((1, 5, 9), (1, 6, 9)), ((1, 5, 9), (1, 5, 10))])
def test_invalid_profile_bounds(query, logits):
    with pytest.raises(ValueError):
        ExecutionProfile("decode", query, logits)
