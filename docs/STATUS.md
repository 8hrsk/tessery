# Independent engine status

On 2026-09-08 the user clarified the scope: a reusable inference system talking
**directly to Metal**, not an MLX wrapper or a Yuri-specific runtime. This
supersedes the original MLX dependency and Go-fixture sequencing. The original
specification is preserved as historical input.

## Implemented

* Objective-C++ runtime: owned shared Metal buffers, pipeline cache, serialized
  command encoding, explicit synchronization and resource release.
* Original Metal kernels: uint4 embedding/linear, float32 matrix multiplication,
  RMSNorm, split-half RoPE, causal grouped-query attention with online softmax,
  SiLU/gating, residual add, last-token pooling and MRL/L2 projection.
* Qwen3 0.6B forward: 28 layers, 16 query/8 KV heads, 128 head dimensions.
  Quantized weights remain on GPU. Attention does not allocate S-by-S scores.
* Original NFC/Unicode-split/byte-level BPE tokenizer. Tokens and masks match
  all 18 saved baseline batches exactly.
* Bounded sync/async embedding API, cancellation, lifecycle/memory reporting,
  CLI, in-memory cosine lookup and general compute primitives.
* Native macOS arm64 wheel and source builds. Runtime dependencies are only
  NumPy and regex, with no MLX, torch, transformers or Hugging Face runtime.

## Architecture

`api` coordinates admission/tokenization/batches. `qwen3` is a registered model
adapter calling kernels through `metal`. Its C ABI lives in `native/runtime.mm`,
which uses Apple Metal directly. `native/kernels.metal` contains the numerical
operations. `weights` validates bytes/offsets/shapes; `tokenizer` implements BPE.
These modules do not import the `yuri_mlx_embeddings` compatibility package.

The existing 335 MB pack is reused in place; no new weights were downloaded.
All three consumed artifacts are hashed. Verified snapshots, not subsequently
reopened paths, supply GPU weights and tokenizer/configuration data.

## Limits and next development

This alpha supports one exact embedding pack. More models need explicit adapters
and verified weight formats. There is no autograd/training, general tensor graph,
generation/KV cache, ANN database, HTTP/UDS daemon or production supervisor.
This is not a complete MLX replacement or a claim of MLX performance parity.

The wheel needs an installed Python and Apple frameworks. It is not a bundled
CPython archive, signed/notarized application or a two-host production release.
Long soak, comprehensive GPU instrumentation, kernel fault recovery and more
advanced tiling/fusion remain future work. Python coverage does not measure
Metal/C++ branch coverage. Hosted CI cannot assume local model availability.

Legacy Yuri compatibility is separate. The native engine has a new float32 space
ID; old/new vectors must not be presumed interchangeable. Missing Go fixtures
block legacy activation, not standalone inference. Yuri Agent and its databases
were not modified.
