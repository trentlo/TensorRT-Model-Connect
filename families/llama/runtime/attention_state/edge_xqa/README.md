# Standalone XQA dependency closure

These sources are extracted from TensorRT-Edge-LLM revision
`e8b29522938901f6df19ebeedd4b69bc8edbcd97`:

- `kernelSrcs/xqa/`: the multi-query XQA kernel and the headers it includes.
- `cpp/common/cudaMacros.h` and
  `cpp/kernels/decodeAttentionKernels/xqaKernelTypes.h`: CUDA feature/type helpers.

Edge's XQA README identifies the earlier TensorRT-LLM source revision as
`a4b4ed45359167eb6cf3c2100d5d0dcd326bc588`.

The individual files retain their original copyright and license notices,
including the NVIDIA TensorRT Source Code License Agreement notices on XQA.
They are not relicensed under Model-Connect's default Apache license.

MC adaptations are confined to the launch boundary and state/visibility access:
separate K/V pools and page maps, `[page,head,slot,dim]` storage, BHQD queries
and outputs, full BOOL masks (including prefix visibility), bounds checks,
FP16 score rounding, empty-row output, and masked-nonfinite fallback. The
tensor-core matrix products, online softmax and shared-memory pipeline come
from XQA. `../xqa.cu` compiles one demanded specialization, FP16/D128/page64,
ahead of time with NVCC; `../xqa_contract.cuh` preserves MC's edge-case semantics.
The donor JIT compiler, runners, plugins, Python packages and runtime are not
dependencies. Other plugin geometries use MC's original scalar kernel.

The fast path checks page metadata, then flags non-finite output registers.
Only affected tiles use scalar repair; it does not scan the entire KV cache
on every invocation. Keep producer warps' exit before the consumer epilogue:
joining barrier 0 from the producer side would interfere with the donor's
consumer-only merge barriers. Empty histories and invalid pages take the
scalar path before any tiled loads. The same plugin still requires zero
TensorRT workspace, so existing serialized engines remain usable.

Keep the imported dependency closure separate from family-owned integration.
When updating the donor, review the adapters and rerun the state-contract and
full-model comparisons; copying a newer kernel without those checks is unsafe.
