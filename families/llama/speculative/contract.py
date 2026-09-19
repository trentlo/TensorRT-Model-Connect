# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned engine ABI. No dependency on TensorRT or a decoding algorithm."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EngineContract:
    """Fixed-capacity linear state with explicit query visibility and positions.

    Tensor semantics, including aliasing and row alignment, are documented in
    ../SPECULATIVE_DECODING.md. A lowering may change without changing this ABI.
    """

    role: str
    layers: int
    hidden_size: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    capacity: int
    max_query: int = 64
    feature_width: int = 0
    version: int = 1
    precision: str = "fp16"
    cache_layout: str = "batch_heads_capacity_dim"
    cache_update: str = "aliased_contiguous_append"

    def __post_init__(self):
        if self.version != 1 or self.role not in {"target", "draft"}:
            raise ValueError("unsupported speculative engine contract")
        if self.precision != "fp16":
            raise ValueError("the first speculative ABI supports FP16 only")
        if (self.cache_layout != "batch_heads_capacity_dim"
                or self.cache_update != "aliased_contiguous_append"):
            raise ValueError("unsupported speculative state layout or effects")
        for name in ("layers", "hidden_size", "kv_heads", "head_dim", "vocab_size",
                     "capacity", "max_query", "feature_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_query > self.capacity:
            raise ValueError("max_query exceeds cache capacity")

    def to_dict(self):
        return asdict(self)


def tree_visibility(parents: list[int]) -> list[list[int]]:
    """Inclusive ancestors in topological row order; root has parent -1."""
    if not parents or parents[0] != -1:
        raise ValueError("a candidate tree must start with one root")
    result = [[0] * len(parents) for _ in parents]
    for row, parent in enumerate(parents):
        if row and not 0 <= parent < row:
            raise ValueError("parents must precede their children")
        if row:
            result[row] = result[parent].copy()
        result[row][row] = 1
    return result
