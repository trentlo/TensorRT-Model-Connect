/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc::llama::speculative {

// Physical state addressing, independent of proposal/acceptance policy. This
// prototype reserves the complete pool per engine; it is not a serving allocator.
// Different K/V page assignments exercise the logical/physical separation.
class StateLayout {
  public:
    StateLayout(int capacity, int page_size) : capacity_(capacity), page_size_(page_size) {
        if (capacity <= 0 || page_size < 0 || (page_size && capacity % page_size))
            throw std::invalid_argument("invalid state page geometry");
        if (page_size) {
            const int pages = capacity / page_size;
            for (int page = 0; page < pages; ++page) {
                key_pages_.push_back(pages - 1 - page);
                value_pages_.push_back(page);
            }
        }
    }
    const std::vector<std::int32_t>& pages(bool key) const {
        return key ? key_pages_ : value_pages_;
    }
    int slot(int logical, bool key) const {
        if (logical < 0 || logical >= capacity_)
            throw std::out_of_range("logical cache row outside capacity");
        return page_size_ ? pages(key)[logical / page_size_] * page_size_ + logical % page_size_
                          : logical;
    }
    std::size_t offset(int logical, bool key, int heads, int dim) const {
        const int physical = slot(logical, key);
        return page_size_ ? (std::size_t(physical / page_size_) * heads * page_size_ +
                             physical % page_size_) *
                                dim
                          : std::size_t(physical) * dim;
    }
    int head_stride() const { return page_size_ ? page_size_ : capacity_; }

  private:
    int capacity_, page_size_;
    std::vector<std::int32_t> key_pages_, value_pages_;
};
} // namespace trtmc::llama::speculative
