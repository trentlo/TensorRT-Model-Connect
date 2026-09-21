# Standalone XQA attention port

The original MC attention-state plugin was slow because its attention kernel
used scalar dot products and serial value accumulation. On GB100, it accounted
for **78.5% of aggregate GPU kernel time** in a six-request profile. Indexed KV
writes accounted for only **0.7%**. We therefore extracted Edge-LLM's multi-query
XQA attention kernel and retained MC's indexed-write implementation.

The port improves EAGLE3 request latency by **2.66×**, from 1732.47 to 652.48 ms.
This fixes the largest measured kernel bottleneck, while leaving MC's engine
composition, prefill strategy and runtime unchanged.

## Repeated end-to-end measurements

Measured September 21, 2026 on one GB100/SM100. FP16 Llama-3.1-8B-Instruct and
EAGLE3, B1/TP1, 1024 input tokens, 101 output tokens, four-candidate top-one
chain, CUDA graphs off. Each backend/mode has three warm-ups and ten measured
requests. All four backends ran sequentially on the same GPU. Throughput
includes prefill; engine loading is excluded.

| Implementation | AR median ms | EAGLE3 chain median ms | Chain P10–P90 ms | Chain output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| MC native graph primitives | 973.02 | 608.23 | 606.75–610.96 | 166.06 |
| MC original scalar plugin | 3773.82 | 1732.47 | 1732.04–1733.23 | 58.30 |
| MC extracted XQA plugin | 1071.94 | 652.48 | 651.65–653.64 | 154.79 |
| Edge-LLM | 624.13 | 248.37 | 244.78–253.70 | 406.64 |

MC's two-child tree medians were 634.79 ms (native), 1788.72 ms (scalar plugin)
and 673.73 ms (XQA plugin). The port improves autoregressive latency by 3.52×
and tree latency by 2.65× against the original plugin.

XQA MC still takes 7.3% longer than native MC and 2.63× as long as Edge for
the chain request. MC uses one query profile and 64-token prefill chunks;
Edge uses separate context/generation profiles and its own context attention
path. This port uses one multi-query XQA specialization, including for MC's
prefill chunks. It does not import Edge's FMHA context path, complete engine
composition or runtime. These results compare complete implementations and
do not isolate how much of the remaining gap each difference causes.

All measured and warm-up requests matched the independent Edge reference for
all 101 IDs. MC chain uses 26 target verification rounds; Edge uses 25. MC's
tree uses 25. MC proposal/acceptance behavior remains unchanged by the port.

The [original measurement report](SPECULATIVE_PERFORMANCE.md) explains timing
boundaries, the pinned weights/fixture and runtime differences. These are
fresh measurements, so baseline numbers differ slightly from that earlier
run. The repetitive fixture gives high acceptance; this is not a general
prompt benchmark. The reference's default RoPE versus MC's llama3 scaling
remains a limitation of broader numerical/long-context comparisons.

## Kernel profile

Each capture includes six requests: one warm-up and one measured request for
each of AR, chain and tree. Both execute 13,192 attention calls. These are
aggregate traced kernel durations, distinct from the unprofiled wall times
above and not attributable to a single execution phase.

| Attention implementation | Total GPU time | Mean per call | Median per call | Share of all kernel time |
| --- | ---: | ---: | ---: | ---: |
| Original scalar | 10.437 s | 791.16 µs | 750.27 µs | 78.5% |
| Extracted XQA | 0.607 s | 46.01 µs | 44.67 µs | 16.7% |

Attention itself improves **17.2×** on this mix. The indexed-write kernel is
unchanged: its aggregate time is 96 ms before and 106 ms after, with its share
increasing as attention becomes faster. After the port, the two largest GEMM
kernel categories account for 13.0% and 11.9%; attention is no longer the
overwhelming GPU bottleneck.

`nsys stats` and the memcpy/synchronization/gap/utilization analysis rules
were run on both captures. The optimized trace contains 1.539 s H2D, 30 ms D2H
and 27 ms D2D transfer activity. The large H2D intervals are engine weight
loading, outside request timing. `cudaMemcpyAsync` dominates aggregate CPU API
duration (4.537 s), which includes waiting and must not be equated with
physical copy time or added to GPU durations. Analysis flags pageable-memory
copies and synchronization; separate phase-specific tracing is needed before
assigning the remaining end-to-end gap to them. CPU sampling/context-switch
collection was disabled, and the model has no phase NVTX ranges here.

Clocks were not locked or modified. Telemetry including initialization and
warm-up had median active SM clock 1965 MHz and memory clock 4000 MHz, with
active temperatures 53–66 °C. Some active samples recorded lower SM clocks;
the saved per-request distributions are the performance evidence. Board
power draw was unavailable.

## Integration and contract

Only the demanded source dependency closure was copied from Edge-LLM revision
`e8b29522938901f6df19ebeedd4b69bc8edbcd97`. It is compiled ahead of time with
NVCC; MC does not import or link Edge's package, runtime, plugin library or
JIT compiler. Original donor license notices are retained. See the
[source provenance](runtime/attention_state/edge_xqa/README.md).

`TRTMC_LLAMA_ATTENTION_STATE_PLUGINS=ON` enables the existing plugin backend.
Within that backend, `TRTMC_LLAMA_EDGE_XQA=ON` selects the extracted kernel
for aligned FP16/D128/page64 tensors with up to 64 query rows. Set it to `OFF`
to build the original scalar plugin. Other geometries retain scalar fallback;
the native `primitives` backend is unchanged.

MC keeps independent K/V pools and page tables, BHQD queries/outputs, full
logical visibility masks, positions, externally owned state, exact indexed
writes and alias checks. The port adapts XQA's accesses to this contract;
it does not replace MC's contract with Edge's packed-mask/cache ABI. Plugin
names/version, serialized fields, tensor bindings and zero TensorRT workspace
are unchanged. Existing v2 plugin engines were reused without rebuilding.

XQA's tensor-core products and online softmax replace the scalar arithmetic.
Masked non-finite storage and empty histories retain MC's defined behavior;
the uncommon affected tile uses scalar repair. No full-cache scan or reset
clearing was added. The [contract documentation](ATTENTION_STATE_PLUGINS.md)
distinguishes exact state semantics from numerical attention tolerances and
explains the route to future native lowering.

## Validation and evidence

- Full-model validation: AR/chain/tree IDs and reset/reuse checks passed.
- Nine GPU attention cases cover the original write/alias checks, decode,
  chain, tree, 64-query prefill, per-head/prefix-hole masks, masked NaNs,
  partially initialized pages, short histories and empty histories. The
  original tolerance remains `rtol=0.005, atol=0.0005`; maximum observed
  attention error was `6.1035e-5`, with exact cache-byte preservation.
- CUDA memcheck and synccheck: all nine cases passed, zero reported errors.
- Separate XQA-on and XQA-off standalone builds passed all nine GPU cases
  with only MC mounted and no Edge package installed. Dynamic dependencies
  are TensorRT, CUDA runtime and system libraries.
- Eight Python compiler/state-contract tests and the C++ speculative policy
  test passed. Modified Python lint and MC adapter formatting passed.

MC baseline source is `54468933`; the new port is this change. Software:
TensorRT 11.1.0.106, CUDA 13.3.73, driver 595.58.03, Nsight Systems
2026.3.1.117, Release build for SM100a. Weights, engines, input-ID fixture and
reference engines are unchanged from the original measurement.

Raw JSON, logs, telemetry, scripts and Nsight reports are retained at
`/home/trentl/Working/specdecode-xqa/`; its README identifies passing final
evidence separately from intermediate debugging runs. The benchmark harness
commands are unchanged from the original report. Reproduce the profile with
one warm-up and one measured request per mode (`101 1 1`), using
`nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`.
