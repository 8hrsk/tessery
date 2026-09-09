# Independent engine status

On 2026-09-08 the user clarified the scope: a reusable inference system talking
**directly to Metal**, not an MLX wrapper or a Yuri-specific runtime. This
supersedes the original MLX dependency and Go-fixture sequencing. The original
specification is preserved as historical input.

## Implemented

* Public `tessery` Python package and CLI, distribution name `tessery`; old
  `metal_inference` imports retain class identity. Version 0.5.1a1 is published
  on PyPI; see [release verification](PYPI_RELEASE.md).

* Objective-C++ runtime: owned shared Metal buffers, pipeline cache, serialized
  command encoding, explicit synchronization and resource release.
* Original Metal kernels: uint4 embedding/linear, float32 matrix multiplication,
  RMSNorm, split-half RoPE, causal grouped-query attention with online softmax,
  SiLU/gating, residual add, last-token pooling and MRL/L2 projection.
* Qwen3 0.6B forward: 28 layers, 16 query/8 KV heads, 128 head dimensions.
  Quantized weights remain on GPU. Attention does not allocate S-by-S scores.
* Original NFC/Unicode-split/byte-level BPE tokenizer. Tokens and masks match
  all 18 saved baseline batches exactly.
* Data-only immutable model profiles with checked artifacts, explicit architecture,
  tokenizer, pooling, dimensions and model identity. Compatible packs can be added
  through manifests without editing core code; architecture features remain bounded.
* BERT float32 encoder, WordPiece, absolute position embeddings, LayerNorm,
  bidirectional attention, erf-form GELU and normalized CLS pooling. The existing
  BGE-small-en-v1.5 pack is validated against offline CPU reference outputs.
* Bounded sync/async embedding API, cancellation, lifecycle/memory reporting,
  CLI, in-memory cosine lookup and general compute primitives.
* Eager float32 Metal tensors: addition, matrix multiplication, transpose, SiLU,
  explicit host readback and deterministic cleanup at tensor/runtime scope.
  Intermediate results stay on the device; operations preserve their inputs.
* Tiled uint4 matmul with chunked F32 accumulation and bounded 64 MiB scratch
  reuse; explicit cache accounting, trimming and disable option.
* SQLite exact retrieval snapshots with embedding-contract checks, conservative
  chunking, source offsets, a working RAG example and index/search CLI commands.
* Loopback HTTP embeddings with bounded admission/connections, timeout cancellation,
  optional bearer authentication, metadata routes and graceful shutdown.
* Short-K tiled F32 matmul, bounded length buckets with output-order restoration,
  shared sync/async submission queue and recovery from partial command creation.
* Cumulative CPU/GPU command diagnostics and reproducible latency/memory/soak tools;
  see [performance diagnostics](PERFORMANCE.md).
* Opt-in per-kernel GPU stage profiling, small-tail uint4 dispatch, seeded
  cancellation/overload/reload stress and an isolated-process MLX comparison;
  see [profiling and stress](PROFILING_AND_STRESS.md).
* Bounded F32 tiled attention for verified long aligned shapes, with scalar
  fallback and stable cosine handling for extreme finite vectors;
  see [attention results](TILED_ATTENTION.md).
* Bounded execution padding to measured matrix/attention tile shapes while
  preserving logical token lengths; see [alignment results](ALIGNED_BATCHING.md).
* Larger uint4 projection tiles with shared weight reuse and unchanged F32
  reduction order; see [quantized tile results](QUANTIZED_TILES.md).
* Chunked F32 BGE projection tiles with bounded scalar tails and unchanged
  accuracy gates; see [F32 projection results](F32_PROJECTIONS.md).
* Execution-aware GPU profiling and isolated Qwen/BGE MLX baselines checked
  in both engine orders; see [updated comparison and priorities](PERFORMANCE_REFRESH.md).
* Bounded attention query/key tails for unaligned widths in the verified
  Qwen/BGE head layouts; see [tail measurements](ATTENTION_TAILS.md).
* Mixed 16-row/eight-row Qwen projection dispatch with bounded tails and exact
  before/after vectors; see [mixed tile measurements](MIXED_QUANTIZED_TILES.md).
* Direct isolated MLP comparisons with real weights, identical-operation timing
  controls and an explicit noise screen; see [MLP comparison](MLP_COMPARISON.md).
* Reproducible in-memory threadgroup traversal experiments with exact-output
  and Shader Validation checks; see [traversal results](QUANTIZED_TRAVERSAL.md).
* Native macOS arm64 wheel and source builds. Runtime dependencies are only
  NumPy and regex, with no MLX, torch, transformers or Hugging Face runtime.

## Architecture

`api` coordinates admission/tokenization/batches. `profiles` binds artifact hashes
to supported adapter contracts. `qwen3` and `bert` call shared kernels through `metal`.
Its C ABI lives in `native/runtime.mm`,
which uses Apple Metal directly. `native/kernels.metal` contains the numerical
operations. `weights` validates bytes/offsets/shapes; `tokenizer` implements BPE.
`tensor` provides the reusable device-resident compute interface through `metal`.
These modules do not import the `yuri_mlx_embeddings` compatibility package.

The existing 335 MB Qwen3 and 133 MB BGE packs are reused in place; no new weights
were downloaded or copied. The BGE cache manifest addresses regular blob files
directly rather than following snapshot symlinks.
All three consumed artifacts are hashed. Verified snapshots, not subsequently
reopened paths, supply GPU weights and tokenizer/configuration data.

## Limits and next development

This alpha supports two architecture/tokenizer/pooling combinations and two
verified real packs. Other packs within those bounded contracts can be described
with manifests; this does not qualify their embedding quality automatically.
New architectures, pooling modes, quantization formats or sharded weights still
need implementation and validation. There is no autograd/training, general tensor graph,
generation/KV cache, ANN database, UDS daemon or production supervisor.
The HTTP server is for trusted local applications; see [its limits](HTTP_API.md).
This is not a complete MLX replacement or a claim of MLX performance parity.

The wheel needs an installed Python and Apple frameworks. It is not a bundled
CPython archive, signed/notarized application or a two-host production release.
Multi-hour soak, per-kernel GPU counter instrumentation, kernel fault recovery and more
advanced tiling/fusion remain future work. Python coverage does not measure
Metal/C++ branch coverage. Hosted CI cannot assume local model availability.

Legacy Yuri compatibility is separate. The native engine has a new float32 space
ID; old/new vectors must not be presumed interchangeable. Missing Go fixtures
block legacy activation, not standalone inference. Yuri Agent and its databases
were not modified.
