# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ..speculative.contract import EngineContract, tree_visibility


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
