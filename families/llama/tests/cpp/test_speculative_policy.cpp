/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"

#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using trtmc::llama::speculative::greedy_path;
    try {
        trtmc::llama::speculative::StateLayout layout(16, 4);
        if (layout.slot(3, true) != 15 || layout.slot(4, true) != 8 || layout.slot(4, false) != 4 ||
            layout.offset(4, true, 2, 8) != 128 || layout.head_stride() != 4)
            throw std::runtime_error("logical/physical page mapping mismatch");
        // Greedy target picks sibling row 2, then its child row 4. Physical
        // row order differs from the logical accepted path.
        const std::vector<std::int32_t> tokens{0, 1, 2, 3, 1};
        const std::vector<std::int32_t> parents{-1, 0, 0, 1, 2};
        std::vector<float> logits(5 * 4, 0);
        logits[2] = 4;
        logits[2 * 4 + 1] = 4;
        if (greedy_path(tokens, parents, logits, 4) != std::vector<std::int32_t>({0, 2, 4}))
            throw std::runtime_error("branch acceptance chose the wrong path");
        logits[2] = 0;
        logits[3] = 5;
        if (greedy_path(tokens, parents, logits, 4) != std::vector<std::int32_t>({0}))
            throw std::runtime_error("rejected children must leave only the pending root");
        bool rejected = false;
        try {
            greedy_path({0, 1}, {-1, 1}, std::vector<float>(8), 4);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("accepted a cyclic parent relation");
        std::cout << "speculative policy tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
