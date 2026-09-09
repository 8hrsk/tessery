# Larger uint4 projection tiles

This records the initial aligned-row optimization. The later
[mixed tile phase](MIXED_QUANTIZED_TILES.md) extends dispatch to incomplete
16-row inputs; the measurements below retain their original scope.

Tessery's selected Qwen projection kernel uses a **16-row by 32-channel**
output tile, eight SIMD groups (256 threads), and **8 KiB of threadgroup
memory** for 32-by-64 decoded weights. It reuses each decoded weight across
twice as many input rows as the previous 8-by-32 kernel. Loading 64 K elements
at a time also halves the number of shared-memory barriers per dot product.
The complete model's float32 weights are never materialized.

Accumulation remains float32, with a separate partial sum for every 32 K
elements, added in the original order. A larger output/decode tile does not
mean a longer floating-point reduction. The new kernel and old tiled kernel
produced bit-identical outputs in the checked direct-kernel cases, including
the cancellation fixture and 4096-row resource boundary.

## Dispatch boundary

`linear4_16x32_k64` is selected only for complete 16-row tiles and the measured
`(output_channels, input_channels)` pairs:

* `(1024, 1024)`
* `(2048, 1024)`
* `(3072, 1024)`
* `(1024, 2048)`
* `(1024, 3072)`

These cover all seven projections in each current Qwen3 0.6B layer, including
the attention output projection with 2048 input channels. Other shapes retain the
existing eight-row tiles and small-tail handling. In particular, eight-row
inputs and batches with 264 projection rows kept the previous path in this phase. No model
profile, tokenizer, padding rule, embedding contract or Python API changed.
BGE's float32 projection path does not use the new kernel.

There are no additional device scratch buffers or runtime dependencies. The
new shared-memory allocation is local to a threadgroup and disappears when
that group completes; existing cache accounting and limits remain unchanged.

## Candidate selection

The initial larger output tiles assigned multiple accumulators to each of
four SIMD groups. They preserved numerical results but were slower on most
tested shapes, so they were removed. Assigning the two eight-row blocks to
separate SIMD groups worked better. A 128-element decode step used 16 KiB of
shared memory but was generally slower than the selected 64-element step.
An eight-row tile with only the decode step enlarged also did not justify
replacement. These exploratory observations do not isolate the hardware
cause of the performance difference.

Only the selected implementation is shipped. Exploratory source snapshots
and short diagnostic runs remain in the ignored local artifacts directory.

## Reproduction and measurement scope

```sh
python tools/benchmark_quantized_tiles.py --samples 15 \
  --output artifacts/quantized-kernels.json
python tools/benchmark_quantized_model.py --model-dir /path/to/qwen \
  --samples 15 --output artifacts/quantized-model.json
```

Both harnesses use three warmups, randomized paired order, raw samples and
source/harness hashes. Direct-kernel comparisons use an independent float64
matrix product at the existing `atol=5e-5, rtol=5e-5` thresholds. GPU command
time and wall time are recorded separately; shader validation is disabled
for timing runs. Full-model pairs use the entire synchronous `encode()` API
and unchanged `atol=5e-6, rtol=1e-4` embedding thresholds.

The full-model baseline dispatch is copied from commit `6592ded`, including
its small-tail behavior. Both sides use the same current padding and attention
code, so these measurements isolate the projection dispatch change. The
reports are a single M1 host observation with uncontrolled power, thermal and
background conditions. They are not a comparison with MLX or a performance
guarantee for other devices.

## M1 results, 2026-09-09

The direct-kernel run measured 25 shapes with 15 samples per variant. The
16–128-row cases improved GPU command time by 1.21–1.28x. The 512-row cases
varied more widely (1.20–2.07x), so the full-model result is the more useful
measure of application benefit. All direct outputs matched the previous
kernel bit for bit and passed the independent F64 tolerance check.
See [raw kernel report](../benchmarks/native-metal/quantized-20260909/kernels.json).

The full-model run also used 15 pairs per case:

| Real token lengths | Previous p50, ms | Selected p50, ms | Speedup |
|---|---:|---:|---:|
| 7, unchanged path | 37.91 | 42.20 | 0.898x |
| 9, execution width 16 | 65.67 | 52.53 | 1.250x |
| 16 | 61.81 | 48.71 | 1.269x |
| 32 | 135.63 | 110.63 | 1.226x |
| 64 | 254.82 | 203.57 | 1.252x |
| 128 | 510.93 | 401.91 | 1.271x |
| 512 | 1610.23 | 1286.08 | 1.252x |
| [128, 127, 33] | 1167.71 | 959.87 | 1.217x |
| eight rows of 33, unchanged path | 1056.07 | 1055.16 | 1.001x |

The seven-token control has identical kernel work but measured 0.898x, while
the eight-row control measured 1.001x. This illustrates the timing variability
of these measurements; a stable latency guarantee cannot be inferred from
the short control. Absolute times also varied between exploratory runs, so
only same-run pairs should be compared. Token lengths are logical lengths;
the existing alignment
policy is applied equally on both sides. These are before/after Tessery
measurements. The older MLX results should not be combined with these timings
to claim a newly measured cross-engine ratio.

All full-model paired vectors matched bit for bit. Full raw samples and model
identity are in [the model report](../benchmarks/native-metal/quantized-20260909/model.json).

## Validation

The complete local native suite passed: **489 tests, 94.53% coverage**. The
12 new native cases check the selected kernel against an independent F64
calculation and exact equality with the old tiled accumulation, covering all
five projection shapes, paired positive/negative cancellation, and 4096 rows.
They also passed with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`.
An additional full-model route test verifies all seven projections per layer,
including the distinct attention output shape, use the selected kernel.

Portable route tests cover complete/incomplete tiles, each selected projection,
small K, small/unaligned channel counts and unsupported projection dimensions.
Ruff, mypy, dependency policy and frozen input checks passed. No numerical
tolerance was relaxed. Local Qwen/BGE packs were reused without downloading or
copying weights.

The final [Qwen stress run](../benchmarks/native-metal/quantized-20260909/qwen-stress.json)
passed four seeded rounds with batch sizes 1/3/8/32, four in-forward
cancellations, queue overload/recovery, one reopen and two closed lifecycles.
Batch versus independent-row vectors matched exactly. After each trim,
335,218,496 active bytes remained for weights and cached bytes were zero;
close released all active allocations.

The rebuilt macOS arm64 wheel passed isolated installed-package smoke checks
with Qwen and BGE. A separate installed-Qwen check confirmed **196 calls** to
the new kernel at 128 tokens (seven projections in each of 28 layers), finite
unit-normalized output and zero active bytes after close. This also checks
that the delivered Python dispatch and Metal source agree.

This is a short regression and stress qualification. Previous multi-hour
qualification runs do not cover this new kernel. Remaining performance work
includes evaluating mixed 16-row/eight-row dispatch for Qwen shapes that
currently use the older path. BGE float32 full tiles with a bounded tail are
now covered by the [subsequent F32 projection report](F32_PROJECTIONS.md).
