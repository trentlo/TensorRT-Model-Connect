# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Include at the end of the reference Edge-LLM checkout's root CMakeLists.txt.
# Deliberately excluded from the Model-Connect build and dependency graph.
add_executable(specdecode_reference "${CMAKE_CURRENT_LIST_DIR}/edge_reference.cpp")
target_link_libraries(specdecode_reference PRIVATE edgellmCore commonLibraryExt)
target_include_directories(specdecode_reference PRIVATE ${COMMON_INCLUDE_DIRS})
