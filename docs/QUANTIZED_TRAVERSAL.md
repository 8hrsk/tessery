# Quantized threadgroup traversal experiment

Measured on 2026-09-09 on the local Apple M1, against the production runtime
at `92e2284384aa04d213e229af50dfcb17d41dc32d`. This follows the
[direct MLP comparison](MLP_COMPARISON.md). No production shader or selector
is changed by this experiment. Existing local Qwen weights are used offline.

## Candidate and protocol

The existing `linear4_16x32_k64` visits output-channel tiles within each
16-row tile. Two candidates instead visit groups of **two or four adjacent
row tiles** for each output-channel tile. A final incomplete row group uses
its actual height; the existing eight-row and bounded scalar/tiled tails
remain in place. This is a possible weight-cache locality improvement, not
a measured diagnosis of cache misses or bandwidth saturation.

`tools/benchmark_quantized_traversal.py` copies the current kernel into two
experimental functions **in memory**, replacing only the function name and
the row/channel mapping. Tile geometry, eight SIMD groups, 8 KiB of decoded
shared weights, BF16 metadata, K64 decoding, independent K32 partial sums,
barriers and F32 reduction order remain the same. The normal runtime source
is never overwritten; only construction of an experimental runtime reads
the generated source. Dispatch replacement is local to that runtime.

The direct experiment uses verified **layer-zero gate/up weights** and
seeded synthetic F32 activations at 128, 129, 264 and 512 rows. Both candidates
and two labels invoking the identical baseline run in randomized paired
order, with at least 20 calls and 100 ms warmup per route. There are 20 samples
per route, ten dispatches per completed command; the table uses command GPU
timestamps divided by ten. Wall times are also retained. Loading, compilation,
readback and independent F64 validation are outside the timing loop. CPU
reference BLAS is limited to one thread. The second fresh process reverses
case order and keeps exactly the same input/weight hashes.

These timings differ from the preceding synchronized-per-operation MLX
comparison: **do not combine their ratios**. MLX is not executed in this
experiment. GPU command timing is normal, without per-kernel profiling;
thermals, clocks and external system activity remain uncontrolled.

## Direct projection results

Speedup is baseline latency divided by candidate latency; above 1 is faster.
Ranges retain both process results. A timing screen requires both baseline
A/B ratios within 0.90–1.10 and each route's two process medians within 15%.
This is a noise screen, not a statistical confidence interval or a minimum
useful gain. Raw samples, including failed controls, are retained.

| Projection | Rows | Group 2 speedup | Group 4 speedup | Timing screen |
| --- | ---: | ---: | ---: | --- |
| gate | 128 | 0.934–0.972x | 0.970–1.060x | fail: baseline A/B |
| up | 128 | 0.977–1.053x | 1.098–1.191x | pass |
| gate | 129 | 1.041–1.048x | 1.016–1.019x | pass |
| up | 129 | 0.994–1.133x | 0.982–1.003x | fail: baseline A/B and process drift |
| gate | 264 | 0.968–0.999x | 0.980–1.020x | pass |
| up | 264 | 0.980–0.984x | 0.995–1.009x | pass |
| gate | 512 | 1.000–1.019x | 0.996–1.010x | pass |
| up | 512 | 0.960–1.000x | 1.000–1.001x | pass |

Neither grouping demonstrates a material, repeated improvement for the
264/512-row target. Group 4's up-projection gain at 128 rows is not reproduced
by the gate projection with the same dimensions. Gate at 129 shows a small
repeatable group-2 gain, while the corresponding up case fails its controls.
These isolated results do not justify a production shape guard.

## Full-model check and decision

The isolated up-projection result at 128 rows motivates a full-model check
of group 4. `tools/benchmark_traversal_model.py` runs the normal public
`EmbeddingModel.encode()` API at 128 and 512 logical tokens, changing only
the gate/up projection traversal in a private experimental runtime. Gate/up
in every layer uses the candidate; the other projection shapes retain their
normal routes. A/B both call the identical baseline. Each case has three
warmup rounds and 15 randomized paired samples per route. Token IDs are
hashed and all outputs must be bit-identical on every round. A second fresh
process reverses case order.

| Tokens | First baseline → group 4 | First speedup | Repeat baseline → group 4 | Repeat speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 401.59 → 412.46 ms | 0.974x | 410.30 → 401.38 ms | 1.022x |
| 512 | 1808.95 → 1819.23 ms | 0.994x | 1774.07 → 1771.34 ms | 1.002x |

Baseline is the median of pooled A/B samples. All four full-model A/B
controls pass (ratios 0.983–1.019); each route's process medians drift by
less than 3%. The 128-token direction changes between runs; at 512 tokens
the difference stays below 1%. There is **no useful repeated full-model
acceleration**. All 216 encode calls, including warmups, produce bit-identical
baseline/candidate vectors. After trimming, only 335,218,496 weight bytes
remain and cache bytes are zero; closing releases all GPU buffers in both
processes.

**Decision: keep the production traversal unchanged.** The long-projection
hypothesis did not yield a useful repeated gain. The isolated 128-row up
result is insufficient to qualify a general shape-based selector. This
rejects these two tested groupings for promotion; it does not prove that
all possible traversal orders are equivalent or identify the hardware
bottleneck.

The next bounded candidate is **fusing gate/up projections with SiLU gating**.
The current Qwen path runs two separate projections followed by `silu_gate`,
materializing both intermediate arrays. A fused implementation could avoid
those intermediate writes and dispatches. It would also need more live
accumulators and potentially more shared storage, so the speed benefit is
unproven. Preserve the current arithmetic/accuracy gates and compare a
prototype with the complete existing three-operation path before changing
production. The present experiment supplies no evidence for a new MLX
performance claim.

## Numerical and boundary validation

All timed cases pass unchanged `atol=5e-5, rtol=5e-5` against an independent
F64 matmul and **exact array equality** against the original runtime. Maximum
absolute error across both timed processes is `2.847e-6` (rounded upward).
Before every validation dispatch the output plus a two-row guard is reset
to NaN: unwritten outputs cannot inherit a previous correct value, and the
guard must remain untouched.

A separate run with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` passes **32
projection/row cases** (four routes each), covering both actual weight tensors
at rows 1, 8, 16, 17, 24, 31, 48, 49, 63, 80, 81, 95, 129, 264, 512 and 4095.
This includes fallback controls, final groups shorter than two/four tiles,
eight-row suffixes and both kinds of partial tail. It checks real weights
with synthetic activations, not arbitrary quantization distributions.
Maximum absolute F64 error is `2.331e-6` (rounded upward). Each case closes
all GPU buffers and asserts zero active bytes. Validation timings are not
performance evidence.

The unchanged portable suite passes **471 tests**. Formatting, lint, type,
dependency and frozen-input checks pass. No package release, model download
or repeat of the older multi-hour qualification is part of this experiment.

## Reproduction and evidence

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_quantized_traversal.py \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/traversal-new.json
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_quantized_traversal.py \
  --model-dir "$QWEN_MODEL_DIR" --reverse-cases \
  --output artifacts/traversal-repeat-new.json
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_quantized_traversal.py \
  --model-dir "$QWEN_MODEL_DIR" --validate-only \
  --rows 1 8 16 17 24 31 48 49 63 80 81 95 129 264 512 4095 \
  --output artifacts/traversal-validation-new.json
```

For the full API check, run the following twice, adding `--reverse-cases`
and choosing another fresh output path on the second run:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_traversal_model.py \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/traversal-model-new.json
```

Reports contain source, generated shader, harness, profile, activation and
weight hashes, plus all samples and buffer accounting:

- [First kernel process](../benchmarks/native-metal/traversal-20260909/kernels.json)
- [Reverse-order kernel process](../benchmarks/native-metal/traversal-20260909/kernels-repeat.json)
- [Shader Validation](../benchmarks/native-metal/traversal-20260909/shader-validation.json)

- [First full API process](../benchmarks/native-metal/traversal-20260909/model.json)
- [Reverse-order full API process](../benchmarks/native-metal/traversal-20260909/model-repeat.json)
