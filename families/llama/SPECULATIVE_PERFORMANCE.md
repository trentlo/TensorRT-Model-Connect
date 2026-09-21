# Llama 3.1 / EAGLE3 performance measurement

This records the original scalar-plugin baseline. See the subsequent
[standalone XQA port and fresh comparison](SPECULATIVE_XQA_PERFORMANCE.md)
for the optimized plugin and profiling evidence.

Measured September 21, 2026 on one Blackwell GB100 (SM100), using the existing
FP16 Llama-3.1-8B-Instruct / EAGLE3 bundles. These are complete synchronous
request measurements of the current prototypes, including runtime scheduling
and state management.

## Main comparison

Batch one, one GPU, 1024 input tokens, 101 generated tokens, greedy decoding,
four draft candidates in a top-one chain. CUDA graph capture is off. Each mode
has three warm-up requests and ten measured requests. Throughput is generated
tokens divided by complete request time, **including prefill**.

| Implementation | AR median ms | EAGLE3 median ms | EAGLE3 P10–P90 ms | EAGLE3 output tokens/s | Speedup over its AR |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC native TensorRT primitives | 950.23 | 599.15 | 598.01–600.17 | 168.57 | 1.59× |
| MC indexed KV / paged attention plugins | 3741.15 | 1726.09 | 1722.84–1733.96 | 58.51 | 2.17× |
| Edge-LLM | 617.56 | 245.16 | 243.20–246.01 | 411.97 | 2.52× |

Edge-LLM is 2.44× faster than native MC and 7.04× faster than plugin MC for this
request. Plugin MC takes 2.88× as long as native MC. Its larger speculative
speedup is relative to its own slower autoregressive baseline.

All measured and warm-up requests produced the same 101 token IDs. The saved
outputs also match the independent Edge-LLM correctness reference. In the AR
baseline, those outputs comprise the prefill prediction and 100 decode calls;
speculative execution requires fewer target calls.

MC chain uses 26 target verification rounds, giving 3.85 emitted post-prefill
tokens per verification. Edge chain uses 25 rounds, giving 4.00. Nominal draft
depth and width match, but proposal/acceptance behavior is not identical. This
is a comparison of the complete implementations, not an isolated kernel test.

## Supplementary measurements

| Mode | Median request ms | Output tokens/s | Verification rounds |
| --- | ---: | ---: | ---: |
| MC native, two-child tree | 632.52 | 159.68 | 25 |
| MC plugin, two-child tree | 1792.43 | 56.35 | 25 |
| Edge chain, graph capture requested | 246.40 | 409.90 | 25 |

MC's tree verifies up to nine rows, versus five for the chain. It saves one
verification round on this fixture but increases total latency by 5.6% for
native MC and 3.8% for plugin MC. This tree policy expands the best branch with
a sibling at each depth; it is not Edge's full beam policy.

Edge's public graph-capture call succeeded before warm-up. The capture-enabled
run's P10–P90 was 244.46–247.98 ms, overlapping the capture-disabled range.
No improvement was observed. Graph replay coverage was not independently
traced; this result does not establish that every decode invocation replayed
a graph or that graph capture is generally ineffective.

## Timing boundaries and phase measurements

The primary timer is a CPU steady clock around `Pipeline::generate_ids` for
MC or `LLMInferenceRuntime::handleRequest` for Edge, with device synchronization
before and after. It includes request reset/preparation, target and draft
execution, host/device transfers, sampling, feature feedback, cache compaction
and output text conversion. It excludes engine loading, construction, graph
capture, input tokenization, output validation and benchmark JSON writing.
Requests reuse loaded engines but do not reuse prompt KV state. Mode order
rotates between repetitions within each backend, and backends run sequentially.
Existing runtime timing instrumentation is enabled: MC records engine events;
Edge records stage events. This is not a measurement with all instrumentation
removed. Request logging inside the timed call remains included.

MC's own CPU phase timers give:

| MC mode | Prefill including draft preparation, ms | Decode loop, ms | Decode-only output tokens/s |
| --- | ---: | ---: | ---: |
| Native AR | 128.96 | 821.01 | 121.80 |
| Native chain | 155.31 | 443.65 | 225.41 |
| Native tree | 154.86 | 477.60 | 209.38 |
| Plugin AR | 589.66 | 3151.32 | 31.73 |
| Plugin chain | 628.95 | 1096.93 | 91.16 |
| Plugin tree | 629.09 | 1163.72 | 85.93 |

Decode-only throughput uses the 100 tokens emitted after the prefill prediction.
For speculative modes, MC's prefill timer also includes draft prefill. Edge
performs draft prefill during its first decode iteration. These phase boundaries
are different, so the main comparison uses complete request wall time. Neither
the MC speculative prefill timer nor Edge's CUDA-event prefill interval is a
measured, comparable user-visible time to first token.

Edge chain's median **CUDA-event stage intervals**, separately from CPU wall
measurements, are 34.95 ms target prefill, 1.87 ms draft prefill, 41.57 ms draft
proposal, 152.58 ms target verification and 13.94 ms draft acceptance/feedback.
Stage medians need not sum to the median request time, and CUDA-event intervals
are not sums of isolated kernel durations.

## Interpretation and scope

The native MC path uses native KV updates and explicit QK/mask/softmax/PV graph
operations; it does not promise a fused native attention tactic. The plugin
path uses the original correctness-oriented kernels documented in
[the plugin contract](ATTENTION_STATE_PLUGINS.md). Its attention kernel assigns
one block per query/head and uses scalar dot products and reductions; it is
not a port of Edge's optimized attention kernels. The results establish the
current implementation gap, not an inherent disadvantage of plugins or paging.
Attributing portions of that gap requires a kernel/runtime profile.

MC uses one query profile and 64-token prefill chunks. Edge uses separate
context and generation profiles. Native MC has linear KV storage; plugin MC
and Edge use different paged ABIs. Host scheduling, feature movement and
acceptance also differ. All are included in the end-to-end numbers.

The fixture repeats a deterministic paragraph and produces high speculative
acceptance. These measurements do not predict throughput on diverse user
prompts, sampling, other batch sizes or longer sequences. The pinned Edge
revision maps llama3 RoPE scaling to default RoPE; MC retains the checkpoint's
trained scaling. Exact greedy IDs on this fixture do not establish broader
numerical or long-context equivalence.

## Reproduction and provenance

Runtime/model code: MC `0f5a29fd1247c7aaa1e313a33e99a1067e48a022`, with only the
benchmark harness/CMake additions for this measurement. Edge reference source:
`e8b29522938901f6df19ebeedd4b69bc8edbcd97`.

- Target: `meta-llama/Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- Draft: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`, revision
  `ada412b672e293d682423de84a095447bf38a637`.
- TensorRT 11.1.0.106, CUDA toolkit 13.3.73, driver 595.58.03, Release builds.
- FP16 weights and KV, capacity 2048, target vocabulary 128256, draft vocabulary
  32000. MC `max_query=64`, `draft_depth=4`; plugin page size 64.
- Input-ID JSON SHA256:
  `1e265729ff51f6d4874c70a32f1abde42b2feddc0b24623400c1c998937d1c4c`.

See [bundle build and fixture instructions](SPECULATIVE_DECODING.md) and
[plugin build flags](ATTENTION_STATE_PLUGINS.md). Build the MC harness with
`TRTMC_BUILD_TESTS=ON` and each desired plugin setting:

```bash
cmake --build build-primitives --target llama_speculative_benchmark
cmake --build build-plugin --target llama_speculative_benchmark
build-primitives/families/llama/llama_speculative_benchmark \
  /models/llama-eagle3.bundle /fixture/input_ids.json 101 3 10 /results/mc-native.json
build-plugin/families/llama/llama_speculative_benchmark \
  /models/llama-eagle3-paged.bundle /fixture/input_ids.json 101 3 10 /results/mc-plugin.json
```

For the independent Edge build, include `tests/reference/edge_reference.cmake`
in a separate reference checkout and build `specdecode_benchmark`. It is not
a Model-Connect dependency. Use the same reference engines as correctness
validation; the documented reference-only compiler workaround still applies.

```bash
EDGELLM_IGNORE_EOS=1 EDGELLM_PLUGIN_PATH=/reference/build/libNvInfer_edgellm_plugin.so \
  /reference/build/specdecode_benchmark /reference/engines \
  /fixture/input_ids.json 101 3 10 0 /results/edge.json
```

The penultimate argument is the graph-capture request: use `1` for the
supplementary capture-enabled measurement. EOS is ignored to keep the output
budget fixed. Prefix reuse and chat templates are disabled.

Raw per-request JSON, output IDs, logs, telemetry, fixture, software/artifact
identity and the summary are retained in the local measurement workspace
`/home/trentl/Working/specdecode-performance/`. The `README.md` there links the
evidence and scripts. No model, runtime policy or attention kernel was changed
for these measurements.
