# Fused Qwen gate/up and SiLU

Experiment on 2026-09-09 on the local Apple M1, following the
[traversal experiment](QUANTIZED_TRAVERSAL.md). The initial baseline is
`819006e6e6497302dffe8c824642e093e54814b6`, with the existing Qwen3 0.6B
uint4/BF16 pack. No model or runtime dependency download is needed.

## Selected production path

`MetalRuntime._gated4` now selects `gated4_16x32_k64` only for
`(rows, cols, K) = (128,3072,1024)` or `(512,3072,1024)`. Rows are the actual
execution matrix height, including batches; this is not a requested-token
limit. Every other shape uses the original two `_linear4` calls and
`silu_gate`, including unaligned tails and the isolated 256-row case for
which this phase has no full-model timing qualification.

Qwen calls this shared private method once per layer. The up scratch
allocation is intentionally retained; there is **no claimed reduction in
allocated memory**. The public API and package version remain unchanged.
The selected shader is byte-identical to the qualified parallel prototype
apart from its entry-point name. The serial kernel stays developer-only.

The original experiment and its exact helper hashes are preserved in
commit `a976f8d`. The updated full-model tool can compare the actual production
route via `--kernel selected` against an explicit unfused baseline, regardless
of the runtime's default selector.

## Experiment

Two private Metal kernels combine gate projection, up projection and
`SiLU(gate) * up` into one dispatch. Both preserve the original 16-by-32
tile, eight SIMD groups, K64 quantization groups and independent K32 F32
partial sums. The epilogue uses the same expression as `silu_gate`.

- **Serial, 8 KiB shared:** computes the complete gate matmul, then up,
  retaining gate accumulators in registers until the final epilogue.
- **Parallel, 16 KiB shared:** decodes both weight tiles, loads each input
  matrix fragment once and updates separate gate/up partial accumulators.
  Both sums retain their own original arithmetic order.

The parallel variant reduces input-fragment loads and barriers relative to
the separate operations; both avoid materializing the separate gate/up
outputs before the final gated output. Those are source-level changes, not
hardware-counter measurements of cache, bandwidth or occupancy. Extra live
accumulators and shared storage can offset the benefit.

The experiment appends `tools/shaders/fused_mlp.metal` to the normal shader
source **in memory**. Each experimental runtime is private. The full-model
hook identifies paired gate/up weight buffers, verifies their shared input
and dimensions, and skips only the immediately following SiLU dispatch.
The prototype retains the original up buffer allocation. Unaligned and
short fallback cases retain the old path.

## Resident operation measurements

`tools/benchmark_fused_mlp.py` uses verified layer-zero gate/up weights and
seeded synthetic F32 activations. The baseline includes **both matmuls and
SiLU**, compared with the complete fused operation. There are two identical
baseline labels, 20 samples per route and ten sequences per completed
command, with randomized paired ordering after at least 20 warmup calls
and 100 ms per route. CPU validation/readback are outside timing; CPU BLAS
uses one thread. A second fresh process reverses cases with identical data.

The table reports baseline/candidate GPU time ratios: above 1 means faster.
Raw command GPU times and wall times are retained. A/B controls must stay
within 0.90–1.10 and each route's medians across processes within 15% to pass
the noise screen. This is not a confidence interval. System thermals and
clocks remain uncontrolled. This experiment does not execute MLX, and its
ratios must not be combined with earlier synchronized MLX timings.

| Rows | Serial speedup range | Parallel speedup range | Timing screen |
| ---: | ---: | ---: | --- |
| 16 | 0.972–0.979x | 1.159–1.164x | fail: first baseline A/B 0.867 |
| 48 | 0.992–1.010x | 1.060–1.072x | pass |
| 128 | 1.021–1.031x | 1.076–1.140x | pass |
| 256 | 1.065–1.078x | 1.222–1.228x | pass |
| 512 | 1.015–1.019x | 1.161–1.176x | pass |

The parallel variant is the candidate for full-model evaluation. Small-row
results are not sufficient to select a production route. These measurements
use layer-zero weights and synthetic activations, not captured hidden states.

## Full API experiment

The normal public `encode()` is tested in two fresh processes with reversed
case order, three warmup rounds and 15 paired randomized samples per route.
The two baseline labels execute identical operations. The parallel candidate
uses exactly 28 fused calls and zero separate SiLU calls per aligned forward;
the seven-token fallback retains 28 separate SiLU calls. Token IDs are hashed.
All **324 encode calls**, including warmups, produce bit-identical vectors.

The repeated aligned results qualify promotion: **1.045–1.059x at 128 tokens**
and **1.047–1.056x at 512 tokens**. Baseline A/B controls pass in both processes.
The seven-token fallback is an identical-path control, not an optimization
claim. The prototype retains the original up allocation. After trim, active
GPU memory consists of 335,218,496 weight bytes and zero cached bytes; closing
releases everything in both processes.

| Tokens | First baseline → fused | First speedup | Repeat baseline → fused | Repeat speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 406.72 → 384.23 ms | 1.059x | 403.51 → 386.26 ms | 1.045x |
| 512 | 1778.75 → 1698.86 ms | 1.047x | 1776.21 → 1682.04 ms | 1.056x |

Baseline is the median of pooled A/B samples. Each aligned route's process
medians drift by at most 3.1%. Seven-token fallback speedups range from
0.981 to 1.039x with the identical path. These results support a small full
model gain, not the 16–23% isolated-operation gain applied to the whole model.

After integration, a third A/B process selects the actual production method
(`--kernel selected`) against an explicit unfused baseline. It records
**1.037x at 128 tokens and 1.031x at 512**, with exact vectors and passing
A/B controls. Seven-token and 129-token fallback controls are 1.022x and
0.997x. Thus the full API evidence spans approximately **3–6% throughput
improvement on the two selected shapes**, not a fixed 5% guarantee. This
post-integration run adds 216 encode calls with exact output equality.
Like the prototype runs, this developer harness appends the unused candidate
functions to its private Metal library; installed-wheel smoke separately
checks the ordinary production library and its dispatch choice.

## Numerical checks

Both original projections are checked separately against independent F64
matmuls; their F64 gated output is also checked against each complete route.
The gates remain `atol=5e-5, rtol=5e-5`. Every fused output must also equal
the original three-operation output **bit for bit**. Before validation the
output and a two-row guard are filled with NaN; every output must be written
and the guard must remain untouched. The maximum absolute F64 error in the
two timed processes is below `5.092e-6`.

A separate `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` run passes all **15
native cases**, each checking both candidates and baseline controls at rows
16, 48, 128, 512 and 4096. Cases cover exact binary-fraction inputs, cancellation
and SiLU saturation at positive/negative 96. All buffers are released after
each case. Shader Validation timings are not performance evidence.

## Production validation

The integrated runtime passes **877 tests**, including both local model
profiles, with **94.58% coverage**. Another **16 cases pass Metal API and
Shader Validation** around the selector boundaries (16, 127, 128, 129, 256,
511, 512 and 513 rows, random binary-fraction and cancellation inputs).
They compare the selected route with independent F64 and the explicit
unfused output, preserve output guards and assert the actual kernel choice.
Full-model tests compare exact vectors for individual 128/512-token inputs,
short/ragged fallback, and batches whose total matrix height is 128.

The four-round Qwen stress run passes batch sizes 1/3/8/32, queue overload,
four cancellations during forward, one reload and two closed lifecycles.
Batch results remain exact; after trim only weights remain and after close
active bytes are zero. Formatting, lint, mypy, dependency and frozen-input
checks pass.

An offline native wheel and source archive are built and verified. The wheel
contains the exact runtime sources, and the source archive includes the
experimental shader needed by the developer tests. An isolated installed
`0.6.0a1` environment passes Qwen/BGE smoke checks and explicitly records 28
fused dispatches at 128/512 tokens, zero at 129, repeated equal vectors and
zero active bytes after close. The up scratch allocation remains unchanged.
These short checks do not replace multi-hour qualification of this revision
or establish performance on other Apple GPUs. No PyPI upload occurs.

## Reproduction

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_fused_mlp.py \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/fusion-kernels-new.json
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_fused_mlp.py \
  --model-dir "$QWEN_MODEL_DIR" --reverse-cases \
  --output artifacts/fusion-kernels-repeat-new.json
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 METAL_INFERENCE_TEST=1 \
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/real_metal/test_fused_mlp.py
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_fused_mlp_model.py \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/fusion-model-new.json
```

Add `--kernel selected` to check the integrated production selector.
Repeat the full-model command with `--reverse-cases` and a fresh output path.
Reports retain hashes of inputs, weights, profile, native sources, candidate
shader and harness dependencies, alongside raw timing samples and diagnostics.

- [First resident process](../benchmarks/native-metal/fusion-20260909/kernels.json)
- [Reverse-order resident process](../benchmarks/native-metal/fusion-20260909/kernels-repeat.json)

- [First full API process](../benchmarks/native-metal/fusion-20260909/model.json)
- [Reverse-order full API process](../benchmarks/native-metal/fusion-20260909/model-repeat.json)

- [Integrated Qwen stress](../benchmarks/native-metal/fusion-20260909/qwen-stress.json)

- [Post-integration selected route](../benchmarks/native-metal/fusion-20260909/model-selected.json)
- [Validation and local build hashes](../benchmarks/native-metal/fusion-20260909/validation.json)
