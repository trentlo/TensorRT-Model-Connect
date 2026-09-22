# Separate prefill and decode profiles

Two specialized profiles with 1024-token prefill reduce EAGLE3 chain request
latency by **16.1% for native attention** and **15.3% for the XQA plugin**, versus
fresh single-profile controls. Keeping 64-token prefill chunks gives only a
0.8% native and 2.4% plugin reduction. Most of the gain comes from larger prefill
chunks; decode latency is essentially unchanged.

The Llama speculative builder now supports `--execution-profiles split` for
both native TensorRT graph primitives and the standalone XQA plugin. Each
target/draft engine has a prefill profile and a decode profile. Two persistent
contexts share that engine's weights, stream and KV allocations. See the
[compiler-runtime contract](SPECULATIVE_DECODING.md#separate-prefill-and-decode-profiles)
for profile bounds and state ownership. The default remains `single`.

## Measured request latency

The follow-up [graph and runtime profile](SPECULATIVE_GRAPH_PROFILE.md) separates
GPU execution from host overhead and confirms the largest remaining CPU cost
with a controlled diagnostic.

Medians of ten timed requests, including target and draft prefill. Throughput
is all 101 generated tokens divided by complete request time.

| Attention backend | Profiles | Prefill chunk | EAGLE3 chain ms | P10–P90 ms | Output tokens/s |
|---|---|---:|---:|---:|---:|
| MC native primitives | Single | 64 | 622.16 | 609.72–633.50 | 162.34 |
| MC native primitives | Split | 64 | 616.95 | 595.10–618.36 | 163.71 |
| MC native primitives | Split | 1024 | **521.98** | 511.87–530.80 | **193.49** |
| MC XQA plugin | Single | 64 | 653.65 | 651.94–655.49 | 154.52 |
| MC XQA plugin | Split | 64 | 637.68 | 636.24–641.02 | 158.39 |
| MC XQA plugin | Split | 1024 | **553.92** | 552.69–558.15 | **182.34** |
| Edge-LLM | Existing context/generation | 1024 | 267.96 | 248.03–269.81 | 376.92 |

Full-prefill MC is 1.19× faster (native) and 1.18× faster (plugin) than its
single-profile control. It still takes 1.95× and 2.07× Edge's request time,
respectively. Separating profiles does not close the remaining runtime and
engine-composition gap. The 0.8% native profile-only difference is smaller than
the observed run variation and does not establish a meaningful speedup.

All 260 warm-up and timed requests matched the independent Edge reference for
every output ID. MC chain still uses 26 verification rounds, its tree uses 25,
and Edge chain uses 25; acceptance counts did not change across MC variants.

| MC chain variant | Prefill including draft, ms | Decode, ms |
|---|---:|---:|
| Native single / 64 | 175.89 | 446.25 |
| Native split / 64 | 166.12 | 450.81 |
| Native split / 1024 | **73.43** | 445.46 |
| Plugin single / 64 | 173.68 | 479.72 |
| Plugin split / 64 | 163.43 | 474.52 |
| Plugin split / 1024 | **75.85** | 475.65 |

Prefill time falls by 58.3% for native and 56.3% for the plugin. Native decode
changes by less than 1 ms and plugin decode by about 4 ms. This supports
prioritizing runtime overhead and decode engine execution for the remaining
gap; these measurements do not attribute that gap to individual kernels or
CPU operations.

Autoregressive request medians change from 992.20 to 920.73 ms (native) and
1070.64 to 977.74 ms (plugin) between single/64 and split/1024. Tree medians
change from 647.97 to 555.78 ms and 673.32 to 581.17 ms, respectively. Edge AR
measured 669.67 ms. The raw summary includes all modes for all six MC variants.

## Experiment

September 21, 2026, one GB100/SM100, FP16 Llama-3.1-8B-Instruct and EAGLE3,
B1/TP1, IST=1024, 101 generated IDs, greedy decoding, draft depth four,
KV capacity 2048, CUDA graphs off. MC uses builder optimization level 1 and a
16 GiB workspace limit. Each variant/mode has three warm-ups and ten timed requests.
The chain verifies up to five rows; MC's two-child tree verifies up to nine.
Autoregressive execution produces the prefill prediction and 100 decode calls.

All six MC variants use the same final runtime binary. The single-profile
controls reuse the existing native and plugin plans; split plans are rebuilt
from the same pinned weights. Every backend runs sequentially on the same
allocated GPU. Engine loading is excluded; request reset, target/draft prefill,
decode, transfers, acceptance, cache commit and output conversion are included.
Requests reuse loaded engines without reusing prompt KV. Mode order rotates
within each backend. The Edge-LLM comparison is a fresh run of the existing
reference engines and benchmark harness on this GPU.

`split64` retains sixteen target and sixteen draft prefill chunks. It changes
the TensorRT tuning bounds and specializes selected logits to one row during
prefill and draft execution. `split1024` additionally reduces each model's
prefill to one invocation. The plugin's existing XQA dispatch is extended from
64 to 1024 query rows after qualification; its kernel body is unchanged.
The experiment therefore separates profile specialization from larger prefill
chunks. It does not isolate every individual tactic or fusion change caused
by rebuilding an engine.

CPU token selection, pageable host staging, synchronization, feature feedback
and KV commit code are unchanged. MC still uses the extracted multi-query XQA
path for plugin prefill; this change does not add Edge's FMHA context path.
There are no new Edge-LLM package, build or runtime dependencies in MC.

MC's prefill and decode columns use CPU phase timers. Its speculative prefill
includes draft preparation, while Edge performs draft prefill within its first
decode iteration. Edge's CUDA-event stage intervals are retained separately
in the raw summary and are not directly comparable to those MC phase timers.
Medians of phases need not sum exactly to median request latency.

Clocks were not locked or modified. Active telemetry, including initialization
and warm-up, had median SM clock 1965 MHz (range 765–1965), memory clock
4000 MHz, and temperatures 51–66 °C. Board power readings were unavailable.
Some request series showed distinct faster/slower intervals; the P10–P90
ranges above preserve that variability. Use the fresh controls rather than
comparing these medians to a different allocation's earlier results.

## Correctness and build checks

- All four new bundles passed IST=1024 / 101-ID comparisons across MC AR,
  chain and tree against Edge AR and EAGLE3, including reset/reuse checks.
- One-token and 65-token prompts passed MC AR-versus-chain comparisons. These
  exercise a one-row prefill and the short final chunk of 64-token prefill;
  they are additional internal consistency tests, not Edge reference cases.
- Thirteen Python profile/state-contract tests and the C++ policy/manifest
  tests passed. The runtime rejects swapped profiles and malformed bounds.
- Ten attention cases passed CUDA memcheck and synccheck with zero reported
  errors. The new 1024-query case covers a broadcast causal mask, independent
  K/V page maps, an empty row and unused NaN storage. Maximum attention error
  was `0.000244140625`; the existing `rtol=0.005, atol=0.0005` is unchanged,
  and cache bytes are checked exactly.
- Plugin-enabled and plugin-disabled runtime builds passed. The native-only
  runtime also passed the 1024-prefill bundle's full-model and Edge comparison.
  The build environment has no importable `tensorrt_edgellm` package.
- Modified Python lint and C++/CUDA formatting checks passed.

## Reproduction and evidence

Example native build:

```bash
python3 -m tensorrt_model_connect llama build-speculative \
  --model-dir /path/to/target --draft-dir /path/to/draft \
  --max-sequence-length 2048 --max-query 64 --draft-depth 4 \
  --attention-backend primitives \
  --execution-profiles split --prefill-query 1024 \
  --output llama-eagle3-split1024.bundle
```

For the plugin, build MC with `TRTMC_LLAMA_ATTENTION_STATE_PLUGINS=ON` and
`TRTMC_LLAMA_EDGE_XQA=ON`; select `--attention-backend plugin --kv-page-size 64`
and supply `--attention-plugin-library /path/to/libtrtmc_llama_attention_state.so`.
Set `--prefill-query 64` for the profile-only comparison. Existing single-profile
bundles remain readable; split bundles require the updated runtime.

Run the unchanged MC harness as:

```bash
llama_speculative_benchmark BUNDLE input_ids.json 101 3 10 output.json
```

Source baseline: `587ed1d0665940ea632d08e7a32ca0610a97eb53`. Software:
TensorRT 11.1.0.106, CUDA 13.3.73, driver 595.58.03, Release/SM100a.
Target revision: `0e9e39f249a16976918f6564b8830bc894c89659` of
`meta-llama/Llama-3.1-8B-Instruct`; draft revision:
`ada412b672e293d682423de84a095447bf38a637` of
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`.
Fixture SHA256: `1e265729ff51f6d4874c70a32f1abde42b2feddc0b24623400c1c998937d1c4c`.

Raw JSON, logs, telemetry, scripts, serialized contract metadata and source
patch are retained in `/home/trentl/Working/specdecode-profiles/`. Its README
identifies the remote bundles and exact reproduction commands. The previous
[XQA report](SPECULATIVE_XQA_PERFORMANCE.md) provides the kernel-port baseline;
use this experiment's freshly measured controls for profile speedups.

This is one repetitive, high-acceptance fixture, not a representative prompt
suite. The existing reference's default RoPE versus MC's llama3 scaling remains
a limitation of broader numerical and long-context comparisons.
