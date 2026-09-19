# Llama speculative decoding prototype

This family owns an FP16, batch-one, single-GPU Llama 3.1 8B Instruct target
and EAGLE3 draft. The target retains its full 128256-token vocabulary. The
draft uses its trained 32000-token vocabulary and an explicit draft-to-target
map. No Edge-LLM package, library, source tree, or exporter is needed to build
or run the Model-Connect bundle.

The first implementation uses TensorRT-native graph primitives, including
`IKVCacheUpdateLayer` and explicitly masked grouped-query attention. No donor
kernel has been ported yet: execution and correctness do not require one.
This is a correctness and architecture prototype, not a performance claim.

## Ownership and entry points

- `speculative/contract.py`: versioned compiler/runtime tensor and state ABI.
- `speculative/graph.py`: target and draft graphs and the attention lowering seam.
- `speculative/build.py`: checkpoint mapping and bundle composition.
- `runtime/speculative/engine.*`: ABI validation, bindings, execution and KV copies.
- `runtime/speculative/eagle3.*`: method-specific conditioning, proposal and feedback.
- `runtime/speculative/pipeline.*`: request lifecycle and target verification.

The existing Llama family remains the owner. Speculative decoding is not a
new model family. Other methods can reuse the engine adapter when their
tensor/state semantics match, and supply their own graphs and scheduling.
Recurrent-state methods need a new state contract; this KV ABI does not claim
to cover every speculative technique.

The family-owned build command is discoverable through Model-Connect's CLI:

```bash
PYTHONPATH=core/builder:. python -m tensorrt_model_connect llama build-speculative \
  --model-dir /models/Llama-3.1-8B-Instruct \
  --draft-dir /models/EAGLE3-LLaMA3.1-Instruct-8B \
  --spec-dec eagle3 --max-sequence-length 2048 --max-query 64 \
  --draft-depth 4 --output /models/llama-eagle3.bundle

cmake --build build-sm100 --target trtmc trtmc_backend_trt trtmc_model_llama
```

The standard family factory loads the resulting bundle as `ITextGeneration`.
`text_generation_mode=auto` uses EAGLE3; `autoregressive` provides a baseline
with the same compiled target. Sampling beyond greedy selection and repetition
penalties are rejected. Existing non-speculative bundles use the original
pipeline.

The prototype expands a top-one chain and, by default, includes its top-two
sibling at each depth in the verification tree. It does not reproduce Edge's
full beam scoring and dynamic tree selection. It does exercise real branching
visibility, depth-based positions and noncontiguous accepted-path compaction.

## Compiler/runtime ABI v1

`speculative.json` names two engine contracts and the EAGLE3 configuration.
For either engine, let `Q` be query rows, `M` selected logits rows, `C` fixed
cache capacity, `Hkv` KV heads, and `D` head dimension. All tensors are dense,
contiguous, batch one. Runtime pointers are device pointers after the backend
copies the supplied host inputs. Tensor names are part of the ABI.

| Binding | Direction/type/shape | Meaning |
|---|---|---|
| `token_id` | input INT32 `[Q]` | Full target-vocabulary IDs, including for the draft embedding lookup. |
| `position_id` | input INT32 `[Q]` | Logical RoPE positions. Siblings share a position; positions do not identify physical cache slots. |
| `cache_write_indices` | input INT32 `[1]` | Physical start `s` of the tentative contiguous write. Row `r` writes slot `s+r`. |
| `key_value_lengths` | input INT32 `[1]` | Initialized span after this invocation, `s+Q`; not the accepted length. |
| `attention_mask` | input INT32 `[Q,C]` | `1` means visible, `0` means hidden. Each row sees the committed prefix and its inclusive ancestors only. |
| `logits_indices` | input INT32 `[M]` | Query-row indices to run through the final norm and LM head, in requested order. |
| `cache_k_i`, `cache_v_i` | input FP16 `[1,Hkv,C,D]` | Runtime-owned persistent state for layer `i`; K is already rotated. |
| `present_k_i`, `present_v_i` | output FP16 `[1,Hkv,C,D]` | Alias the corresponding cache input. Only `[s,s+Q)` changes; all other slots are preserved. |
| `logits` | output FP32 `[M,V]` | Row `j` predicts the token after query row `logits_indices[j]`, under that row's visible history. Raw logits, not probabilities. |
| `features` (target) | output FP16 `[Q,12288]` | Concatenated inputs to target decoder layers 2, 16, 28, before their norms; identical query-row order. |
| `target_features` (draft) | input FP16 `[Q,12288]` | Verified target features from the preceding logical token, or zero during recurrent drafting. |
| `draft_features` (draft) | input FP16 `[Q,4096]` | Previous draft residual features, or zero when using target features. |
| `features` (draft) | output FP16 `[Q,4096]` | Unnormalized draft residual stream, in query-row order. |

Both engines have one optimization profile, `1 <= M <= Q <= max_query`.
Prefill is chunked at `max_query` (64 by default). A prefill call selects only
its last logits row while returning all feature rows. Verification selects
every candidate row. The runtime enforces `s >= 0` and `s+Q <= C`.
The two conditioning inputs must have the same `Q` as the draft's token input.

The compiler encodes KV reads, writes and aliases in the graph. The sidecar
describes that graph contract; it is not a replacement for compiler-visible
state effects. The runtime rejects engines whose cache outputs are not declared
as aliases of their inputs; manually binding unrelated tensors to the same
address does not satisfy this contract. TensorRT owns engine workspace.
The runtime owns KV buffers,
host input/output storage and temporary compaction storage. Every execution
and copy uses its module's CUDA stream. This prototype synchronizes before
returning outputs and before crossing between the target and draft streams.
Thus host staging is safe but adds overhead. An asynchronous implementation
must preserve these dependencies with events and buffer lifetimes.

Only initialized visible slots may affect attention. The current lowering
sanitizes inactive cache rows before matmuls, so masked stale NaNs cannot
contaminate the result. Reset changes logical state; clearing all KV bytes is
not necessary. Unaccepted rows remain inaccessible until overwritten.

## EAGLE3 state alignment

After target prefill of `N` tokens, the target emits a root token whose KV has
not yet been computed. Draft prefill consumes the shifted tokens
`prompt[1:] + root` paired with the unshifted target features. Its positions
are `0..N-1`. Its last output predicts the first draft candidate.

A target verification invocation consumes `root + candidates`. Each logits
row predicts a child, not the token occupying that row. Greedy acceptance
walks the matching root-to-leaf path. The runtime commits KV for the root and
accepted candidates, emits accepted candidates and a target bonus token, and
leaves that bonus pending. For trees it first gathers accepted physical rows
to temporary storage, then compacts them into the committed prefix. This
two-phase copy is safe even when source and destination slots overlap.

Draft feedback uses verified target features for the committed path, paired
with the next accepted token or the bonus. It overwrites tentative recurrent
draft rows. Target and draft committed lengths therefore advance together,
despite the one-token offset in their conditioning semantics.

## Swapping attention implementations

`Graph.attention()` currently forms Q/K/V, applies RoPE, performs native linear
KV updates, and emits an explicit masked attention graph. A future plugin
lowering can replace this region without changing the external ABI if it:

1. Preserves the same FP16 Q/K/V, scale `1/sqrt(D)`, rotate-half RoPE and
   grouped-query semantics.
2. Accepts arbitrary declared query visibility and logical positions,
   including siblings with equal positions and different physical slots.
3. Reads/writes exactly the specified cache layout and write interval, exposes
   the state effects and returns the same aliases.
4. Preserves selected logits and feature row order, dtype and shape profiles.
5. Respects stream ordering, workspace ownership and output lifetime.

Conversely, replacing such a plugin with TensorRT `IAttention` or Myelin
lowering requires encoding these same semantics in native graph operations.
A plugin name or sidecar flag does not make an opaque plugin compiler-visible.
The conversion must replace its graph node/region and explicitly model its
state effects. Compiler fusion/tactic selection can then occur internally.

Rebuilding an engine may change its internal attention implementation while
the runtime keeps the same bindings and algorithm. Changing public KV layout,
alias rules or feature semantics requires an ABI version or an explicit
adapter. In particular, current Edge-LLM's paged, packed-mask ABI is **not**
binary compatible with this linear layout. A donor kernel must be adapted at
this seam; its runtime memory assumptions cannot be copied silently.

The initial extension point is source-level graph selection. No unused plugin
loader, kernel copy, or speculative-technique registry is introduced. A future
plugin port should include only the required kernel and its dependency closure,
retain its license/provenance, and run the same contract tests against both
lowerings. Plugin performance and native fused-attention coverage remain
separate qualification work.

## Validation

Build `llama_speculative_validation` and `test_llama_speculative_policy` with
`TRTMC_BUILD_TESTS=ON`. The validation executable loads the public bundle and
family runtime, compares autoregressive, chain and branching outputs, then
checks a second request after speculative state has been used:

```bash
python families/llama/tests/prepare_speculative_fixture.py /models/target /tmp/spec-fixture
build-sm100/families/llama/llama_speculative_validation \
  /models/llama-eagle3.bundle /tmp/spec-fixture/input_ids.json 100 /tmp/result.json
```

The fixture is 1024 raw input IDs with no chat templating. A 100-token output
contains the first prefill prediction and 99 ordinary decode calls in the
autoregressive baseline. Use output count 101 for exactly 100 such calls.
The report includes complete output IDs, acceptance lengths, verification
rounds and wall times. Timings include host scheduling and copies; they are
not an optimized serving benchmark.

Independent Edge-LLM validation must use these exact input IDs, checkpoints,
precision and greedy settings. Its build and export dependencies belong in a
separate environment. Source topology was inspected at
`e8b29522938901f6df19ebeedd4b69bc8edbcd97`; the public draft revision used in
bring-up is `ada412b672e293d682423de84a095447bf38a637`.

The independent harness is in `tests/reference/edge_reference.cpp`. It calls
Edge-LLM's public runtime with `preTokenizedInputIds`, greedy selection and
context-cache lookup bypassed, first with speculative decoding disabled and
then with EAGLE3. It uses a depth-four, top-one draft policy. To build it,
include `tests/reference/edge_reference.cmake` by absolute path at the end of
a **separate reference checkout's** root `CMakeLists.txt`, then build its
`specdecode_reference` target. That CMake fragment is never included by the
Model-Connect build.

Export the target with Edge's `--eagle-base --eagle-draft-dir` options and the
draft through its ordinary exporter. Build the resulting engines into the
same directory using `llm_build --specBase` and `llm_build --specDraft`, with
`--maxInputLen 1024 --maxKVCacheCapacity 2048 --maxBatchSize 1` and tree-size
profiles of at least 9. Then run:

```bash
EDGELLM_PLUGIN_PATH=/reference/build/libNvInfer_edgellm_plugin.so \
  /reference/build/specdecode_reference \
  /reference/engines /tmp/spec-fixture/input_ids.json /tmp/edge.json 101

python families/llama/tests/reference/compare_outputs.py \
  /tmp/spec-fixture/input_ids.json /tmp/modelconnect-101.json /tmp/edge.json
```

The comparison requires exactly 101 matching IDs in five streams:
Model-Connect autoregressive, EAGLE3 chain and EAGLE3 tree; and Edge-LLM
autoregressive and EAGLE3. It also checks prompt counts and Model-Connect
request-reset equivalence. The GPU test and reference test report IDs rather
than relying on detokenized text equality.

### Recorded prototype result

Validated on a Blackwell GB100 (SM100), TensorRT 11.1.0.106, CUDA 13.3 and
driver 595.58.03, using FP16 engines and KV state. The target checkpoint was
`meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659`; the draft and reference revisions
are listed above.

- Exactly 1024 input IDs and 101 output IDs: all five streams matched exactly.
- The autoregressive baseline made 100 decode calls after its prefill prediction.
- Model-Connect's branching path used 25 target verification rounds; its chain
  path used 26. Both acceptance and rejection occurred.
- Reusing the pipeline for an autoregressive request after speculative writes
  produced the same output. Zero-output and invalid-count checks passed.
- The public `trtmc run` bundle-loading path passed an additional smoke run.
- The C++ policy test, 25 focused Python tests, family ownership validation,
  Ruff and C++ formatting checks passed.
- Model-Connect was built and run in a separate container with no importable
  Edge-LLM package; its runtime library has no Edge-LLM library dependency.

For this reference build, Edge-LLM's CuTe FMHA artifacts were generated for
SM100. Its default TensorRT compiler route failed with a generated-GEMM
workspace-query error. Setting `__LUNOWUD=-mlir:collective=off` for the two
**reference engine builds** succeeded. No Edge-LLM model or kernel source was
changed, and this workaround is not part of Model-Connect's build or runtime.

This is greedy-token parity for one deterministic fixture. It is not a
tensorwise-logit or general model-quality qualification. In particular, this
Edge revision's `collectRopeConfig` maps `llama3` scaling to default RoPE;
Model-Connect retains the checkpoint's scaled RoPE. Broader prompt and
long-context numerical qualification must account for that reference
difference. Full beam-policy parity, sampled decoding, batching, paged KV,
quantization, plugin lowering and fused native attention remain follow-up work.
