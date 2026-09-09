# BGE F32 projection tiles

On 2026-09-09, Tessery's BGE-small-en-v1.5 full encode API measured **2.04x
at 128 tokens** and **1.79x at 512 tokens** against the previous Tessery
projection dispatch, on the local Apple M1. Numerical gates are unchanged.
These are same-run before/after measurements, not a new comparison with MLX.

## Kernel and dispatch

`matmul_f32_chunk32` computes an 8-row by 32-channel output tile with four
SIMD groups (128 threads). It reads F32 inputs and transposed `[N,K]` weights
directly from device buffers. Each SIMD group owns one 8x8 fragment, computes
four eight-product matrix operations into a fresh F32 partial accumulator,
then adds that partial to its running F32 sum. The inner sum resets every
32 products. There is no additional shared-memory array, device scratch
allocation, weight conversion or model copy. Metal fast math remains disabled.

The original sequential tiled kernel can exceed the existing error gate for
long K; the exploratory K=1536 cases reproduced that failure. Partial chunks
of 32, 64 and 128 were evaluated. The 32-product variant passed the same
F64 checks and was the fastest of those candidates across the explored
projection shapes. Exploration is retained locally in ignored
`artifacts/f32-11/explore.json`; only the selected kernel ships.

Host selection requires at least eight rows and one of these `(N,K)` pairs:

| Projection | N | K | Calls per BGE layer |
| --- | ---: | ---: | ---: |
| Query, key, value, attention output | 384 | 384 | 4 |
| MLP expansion | 1536 | 384 | 1 |
| MLP output | 384 | 1536 | 1 |

Complete eight-row tiles run first. If one to seven rows remain, one additional
scalar dispatch computes only those rows, preserving its original reduction
order. The scalar kernel's `Params.n` is the starting row, zero by default.
Collective tiled loads/stores never cross the logical row extent. All other
shapes retain their previous routing, including the K<=512 restriction on
the older generic tiled kernel. Public API, padding policy, profiles and
compatibility IDs are unchanged. This does not extend the supported model
architectures.

## Measurements

The committed reports contain raw timing samples, source and harness hashes,
model identity (for encode), and the baseline dispatch commit
`b2fcc2fa8664518a56fee60341cd5690a22a9d8d`.
Both harnesses use three warmups followed by 15 measured pairs in seeded
randomized order. GPU workloads were run sequentially. Thermals and other
system activity were not controlled; the figures are observations on this
M1, not guarantees for every Apple GPU or input distribution.

[Direct projection measurements](../benchmarks/native-metal/f32-20260909/kernels.json)
cover seven row counts and all three projection shapes (21 cases). They
compare the complete previous and selected dispatch routes, including the
extra scalar dispatch where needed. GPU command time and wall time are stored
separately. At 512 rows, the GPU speedups were **2.41x** for 384x384,
**2.02x** for 1536x384, and **4.15x** for 384x1536. The nine/fifteen-row
cases improved by 1.27–2.28x. Seven-row controls use identical kernels and
measured 0.96–1.00x.

[Full API measurements](../benchmarks/native-metal/f32-20260909/model.json)
include tokenization, admission, the existing batching policy, inference and
readback. Both routes receive identical text and use the same attention and
padding implementation. Lengths below are logical token counts, including
special tokens.

| Token lengths | Previous p50, ms | Selected p50, ms | Speedup |
| --- | ---: | ---: | ---: |
| 3 | 7.59 | 5.59 | 1.36x |
| 7 | 8.19 | 5.43 | 1.51x |
| 9 | 9.31 | 6.97 | 1.34x |
| 16 | 14.29 | 8.28 | 1.73x |
| 32 | 17.84 | 10.57 | 1.69x |
| 64 | 29.83 | 16.20 | 1.84x |
| 128 | 58.68 | 28.76 | 2.04x |
| 512 | 226.23 | 126.20 | 1.79x |
| 128, 127, 33 | 126.13 | 61.39 | 2.05x |
| 8 × 33 | 115.39 | 62.95 | 1.83x |

The three-token control uses identical scalar work but measured 1.36x. This
shows substantial timing noise for very short requests; no short-request
speedup claim is justified by this run. Longer requests also vary, so do not
combine absolute timings from separate runs or old MLX reports to derive a
cross-engine speed ratio. All paired vectors met `atol=5e-6, rtol=1e-4`;
maximum absolute difference was **1.20e-7** (rounded up), not bitwise equality.

## Validation and package

* **614 tests passed, 94.55% coverage**, including both complete native models,
  the independent frozen BGE CPU reference, public API and generic tensor tests.
* **61 native cases passed with Metal Shader Validation and the debug layer**:
  all three projection shapes, each one-to-seven-row tail, 128/4096 rows,
  random inputs and opposing products with a small residual. A sentinel test
  proves the scalar tail leaves completed rows untouched. Independent F64
  references use the existing `atol=5e-5, rtol=5e-5` gate. This is evidence for
  tested distributions, not a bound for all possible F32 inputs.
* Portable tests cover dispatch boundaries, unqualified dimensions and the
  preserved generic long-K fallback. A real-model test confirms 72 selected
  calls per 128-token encode: six projections in each of 12 layers.
* Ruff, mypy, dependency policy and frozen input checks passed.
* The [four-round BGE stress report](../benchmarks/native-metal/f32-20260909/bge-stress.json)
  covers batch sizes 1/3/8/32, four cancellations during forward, queue overload
  and recovery, one reopen and two closed lifecycles. Batched versus independent
  vectors matched exactly in this seeded corpus. Each trim retained only
  132,848,640 weight bytes with zero cache; closing released all active buffers.
* An offline macOS arm64 wheel passed isolated installed-package Qwen/BGE smoke
  checks. A separate installed-BGE check confirmed 72 calls to the new kernel,
  finite unit-normalized output and zero active bytes after close.

Local cached models were reused. No models or dependencies were downloaded.
The wheel remains a local `0.6.0a1` candidate; this change does not publish a
PyPI release. Short regression/stress runs do not replace multi-hour
qualification of these new kernels or validation on other Apple GPUs.

## Reproduction

```sh
METAL_INFERENCE_TEST=1 .venv/bin/python -m pytest --cov
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 METAL_INFERENCE_TEST=1 \
  .venv/bin/python -m pytest tests/real_metal/test_kernels.py \
  -k 'f32_chunked or f32_scalar_tail'
.venv/bin/python tools/benchmark_f32_tiles.py --output artifacts/f32-kernels-new.json
.venv/bin/python tools/benchmark_f32_model.py \
  --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --output artifacts/f32-model-new.json
.venv/bin/python tools/stress_embeddings.py \
  --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --rounds 4 --output artifacts/f32-stress-new.json
```

Set `BGE_MODEL_DIR` to the existing verified local pack. Reports require new
output paths. Next useful work is a fresh full-model profile and paired MLX
comparison on the updated runtime, followed by measurement of fusion for
BGE's separate bias, activation and residual passes.
