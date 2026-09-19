# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require exact greedy output agreement across independently built runtimes."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("modelconnect", type=Path)
    parser.add_argument("edge", type=Path)
    parser.add_argument("--output-tokens", type=int, default=101)
    args = parser.parse_args()
    fixture = args.fixture.read_bytes()
    input_ids = json.loads(fixture)
    mc = json.loads(args.modelconnect.read_text())
    edge = json.loads(args.edge.read_text())
    if mc["input_tokens"] != len(input_ids) or edge["input_token_counts"] != [len(input_ids)]:
        raise ValueError("input token count differs between runtimes")
    streams = {
        "modelconnect_autoregressive": mc["autoregressive_ids"],
        "modelconnect_chain": mc["chain_ids"],
        "modelconnect_tree": mc["eagle3_ids"],
        "edge_autoregressive": edge["autoregressive_ids"],
        "edge_eagle3": edge["eagle3_ids"],
    }
    expected = streams["edge_autoregressive"]
    for name, ids in streams.items():
        if len(ids) != args.output_tokens:
            raise ValueError(f"{name}: expected {args.output_tokens} tokens, got {len(ids)}")
        for index, (actual, reference) in enumerate(zip(ids, expected)):
            if actual != reference:
                raise ValueError(f"{name}: token {index}: {actual} != {reference}")
    if not mc["reset_equal"]:
        raise ValueError("Model-Connect request reset changed the output")
    print(json.dumps({
        "input_tokens": len(input_ids),
        "input_ids_sha256": hashlib.sha256(fixture).hexdigest(),
        "output_tokens": len(expected),
        "all_five_streams_equal": True,
        "modelconnect_reset_equal": True,
        "verification_rounds": mc["verification_rounds"],
        "output_ids": expected,
    }, indent=2))


if __name__ == "__main__":
    main()
