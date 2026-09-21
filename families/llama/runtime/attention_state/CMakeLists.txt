# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_llama_attention_state SHARED plugins.cpp kernels.cu)
option(TRTMC_LLAMA_EDGE_XQA "Use standalone Edge-LLM XQA for qualified Llama plugin shapes" ON)
if(TRTMC_LLAMA_EDGE_XQA)
  target_sources(trtmc_llama_attention_state PRIVATE xqa.cu)
  target_compile_definitions(trtmc_llama_attention_state PRIVATE TRTMC_LLAMA_EDGE_XQA=1)
  set_source_files_properties(xqa.cu PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr;--extended-lambda")
endif()
target_include_directories(trtmc_llama_attention_state SYSTEM PRIVATE ${TRTMC_TRT_INCLUDE_DIR})
target_link_libraries(trtmc_llama_attention_state PRIVATE ${TRTMC_TRT_LIBRARY} CUDA::cudart)
set_target_properties(trtmc_llama_attention_state PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
install(TARGETS trtmc_llama_attention_state LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
target_link_libraries(trtmc_model_llama PRIVATE trtmc_llama_attention_state)
target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_LLAMA_ATTENTION_STATE_PLUGINS=1)
