# Bounded attention tails

This follows the [profile refresh](PERFORMANCE_REFRESH.md), which identified
scalar-attention fallbacks around 129 tokens. It extends the independent Metal
runtime with bounded query/key tails and leaves the public API, execution
padding policy, model profiles and projection kernels unchanged.

## Implementation

Two entry points, `attention_tail_32` and `attention_tail_128`, instantiate the
same original F32 implementation for the verified head dimensions. Each group
of 128 threads computes eight queries against 32 keys at a time, with stable
online softmax. Complete query/key tiles load directly from device memory.
For the incomplete physical block, guarded reads fill missing values with zero;
stores exclude query rows outside the sequence. Batch indexing uses the ceiling
of sequence width divided by eight, preventing a final partial tile from
crossing into the next batch row.

The boundary buffer stages 32 keys/values by 32 features (4096 bytes), and is
reused as feature strips advance. The existing partial-output buffer doubles
as bounded query storage before the value product. Uniform threadgroup barriers
separate scratch writes, matrix reads and reuse. Shared arrays total **7264
bytes for dimension 32** and **13408 bytes for dimension 128**, independent of
sequence length. No sequence-squared buffer, additional global workspace or
model-weight copy is introduced. Fast math stays disabled.

An initial full-head staging buffer used 25696 shared bytes. Numerical tests
passed, but Metal Shader Validation reported 51392 bytes after instrumentation,
exceeding the M1's 32768-byte threadgroup limit. That candidate was discarded.
The selected strip buffer and dimension-specific storage passed validation
without disabling the diagnostic layer. No Metal toolchain download was needed;
the existing runtime compiler builds these kernels.

Dispatch selects the new kernels only for sequence widths **64 through 512**,
not divisible by 32, and head dimensions 32 or 128. Existing aligned tiled
attention and other scalar routes are preserved. Causal, bidirectional,
grouped-query and ragged-length semantics are checked against independent F64
references. As with the earlier tiled kernel, the scope is finite internal
model tensors; masked NaN/Inf behavior is not a new contract.

## Measurements

The baseline is the complete attention dispatcher at
`9b7bb1e5015c3d6731a4803bf050dde5ef6f4869`, including its already-accelerated
aligned path. This is a comparison with previous Tessery, not a fresh MLX
comparison. Old MLX timings must not be combined with these observations to
claim a newly measured cross-engine ratio.

Both harnesses use three warmups and 15 measured pairs in seeded randomized
order on the local Apple M1, macOS 26.3. Inputs and all other execution policies
are identical across each pair. Reports contain source/harness hashes and raw
samples. GPU jobs were run sequentially; thermals and background activity were
not controlled, so small differences should be treated cautiously.

[Direct kernel results](../benchmarks/native-metal/attention-tails-20260909/kernels.json)
cover widths 64/65/71/127/128/129/136/255/511/512 for both BGE and Qwen head
layouts. GPU command time is distinct from wall time. At width 129 the GPU
speedups were **4.04x for BGE** and **1.75x for Qwen**; at 511 they were
**7.88x** and **2.97x**. Aligned controls execute identical kernels. Their GPU
ratios ranged 0.84–1.05, illustrating measurement noise even with unchanged
work. All direct outputs passed `atol=2e-6, rtol=2e-5` against F64.

The full API comparison includes tokenization, admission, current batching,
inference and readback. Logical lengths are verified using each model's own
tokenizer. In particular, 129 executes at width 129 for Qwen and 136 for BGE;
257 executes at 257 and 264 respectively. Both sides keep those same widths.
Synthetic repeated-token texts are a performance fixture, not a RAG-quality
benchmark.

### BGE full encode

| Logical lengths | Previous p50, ms | Selected p50, ms | Speedup |
| --- | ---: | ---: | ---: |
| 64 | 16.41 | 15.91 | 1.03x |
| 65 | 23.07 | 16.99 | 1.36x |
| 128 | 27.91 | 29.30 | 0.95x |
| 129 | 53.00 | 32.55 | 1.63x |
| 136 | 55.30 | 30.57 | 1.81x |
| 257 | 152.00 | 67.23 | 2.26x |
| 129, 127, 65 | 131.50 | 83.28 | 1.58x |

[BGE raw report](../benchmarks/native-metal/attention-tails-20260909/bge-model.json).

### Qwen full encode

| Logical lengths | Previous p50, ms | Selected p50, ms | Speedup |
| --- | ---: | ---: | ---: |
| 64 | 205.56 | 205.99 | 1.00x |
| 65 | 314.09 | 299.89 | 1.05x |
| 128 | 414.42 | 413.89 | 1.00x |
| 129 | 643.69 | 601.72 | 1.07x |
| 136 | 634.21 | 565.44 | 1.12x |
| 257 | 1426.94 | 1192.36 | 1.20x |
| 129, 127, 65 | 1814.45 | 1686.37 | 1.08x |

[Qwen raw report](../benchmarks/native-metal/attention-tails-20260909/qwen-model.json).

Qwen benefits less because projections still dominate its runtime and its
unaligned projection route is unchanged. This change reduces the attention
component of the 129-token cliff; it does not eliminate the remaining projection
cost. At 257 tokens Qwen improved by 1.20x, while BGE improved by 2.26x.

The aligned 64/128-token controls use identical work. Full-model control ratios
ranged 0.95–1.03 for BGE and about 1.00 for Qwen. They are not evidence of a new
algorithmic speedup or slowdown on those unchanged routes. Every paired output
passed `atol=5e-6, rtol=1e-4`. Maximum absolute vector differences were
**2.09e-7 for Qwen** and **1.50e-7 for BGE** (rounded up).

## Validation

The direct attention tests cover causal and bidirectional attention, GQA,
head dimensions 32/128, two batch rows with different valid lengths, widths
64/65/71/95/127/128/129/136/255/511/512, very short valid prefixes, logits near
+/-1131, late maxima, uniform logits and large finite masked values. All **68
attention cases passed Metal API and Shader Validation**, using the unchanged
independent F64 thresholds. No diagnostic mode was disabled to qualify the
selected kernel.

Portable route tests cover lower/upper bounds, unsupported head dimensions,
existing aligned routes, and the ceiling-based dispatch size. Real-model tests
verify one new attention dispatch per layer at 129 logical tokens: 28 for Qwen
and 12 for BGE. The production bridge ABI is unchanged.

The full local suite passed **671 tests, 94.59% coverage**, including both
real-model CPU/reference checks and public API/tensor regressions. Ruff,
formatting, mypy, installed dependency policy and frozen input checks passed.
No numerical tolerance was relaxed. The existing local model packs were reused
without downloading or copying model weights.

Both [Qwen](../benchmarks/native-metal/attention-tails-20260909/qwen-stress.json)
and [BGE](../benchmarks/native-metal/attention-tails-20260909/bge-stress.json)
passed four seeded stress rounds (batches 1/3/8/32), four cancellations during
forward, queue overload/recovery, one reopen and two closed lifecycles each.
Batched versus independent vectors matched exactly on this stress corpus.
After trim, Qwen retained 335,218,496 weight bytes and BGE 132,848,640, with zero
cache; closing released all active model buffers. Threadgroup storage is not
part of those global buffer counters.

The offline macOS arm64 wheel passed isolated installed-package smoke checks
for both models. A separate installed check at 129 tokens confirmed 28 calls to
`attention_tail_128` for Qwen and 12 to `attention_tail_32` for BGE, finite
unit-normalized outputs, and zero active buffers after close. Wheel/source
layout validation passed. The version remains a local `0.6.0a1` candidate;
this does not publish a PyPI release. These short runs do not qualify the new
kernel for a multi-hour release soak or establish performance on other GPUs.

## Reproduction

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 METAL_INFERENCE_TEST=1 \
  .venv/bin/python -m pytest tests/real_metal/test_kernels.py -k tiled_attention -s
METAL_INFERENCE_TEST=1 .venv/bin/python -m pytest --cov
.venv/bin/python tools/benchmark_attention_tails.py \
  --output artifacts/attention-tails-new.json
.venv/bin/python tools/benchmark_attention_tail_model.py \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/qwen-attention-tails-new.json
.venv/bin/python tools/benchmark_attention_tail_model.py \
  --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --output artifacts/bge-attention-tails-new.json
```

Use existing verified local model paths and fresh output paths. The following
phase implements [mixed 16-row/eight-row Qwen projections](MIXED_QUANTIZED_TILES.md),
with unchanged numerical gates and same-input API timing controls.
