# Qualified Tessery versus compiled MLX — 2026-09-13

On this Apple M1 and local Qwen3-Embedding-0.6B DWQ model, Tessery retains an
advantage at 7 and 24 tokens. The compiled MLX reference leads at 3 tokens and
at 128–512 tokens. Mixed short batches are close to parity. The long-input gap
persists in prepared backend timing, so it remains primarily inside the backend
path after removing tokenization, admission and batch packing.

Runtime commit: `d7edc5bcda8926b411f5dff36b90ec8446b9187d`.
Its preceding two-hour Qwen qualification passed 48,772 calls across 16 scenarios.
This comparison changes no runtime code and uses the exact qualified native
library and shader. Wheel SHA256:
`260ce768960940f76bb49f4c38c22c3798976b4ab8973ba2c7222142c7c6320d`.

## Measurement boundary

The baseline is the existing independent F32 Qwen graph using public MLX 0.32.2
operations and `mx.compile`; it is not the mlx-embeddings package or every possible
MLX configuration. Both engines use the same pinned weights, uint4 quantization,
F32 arithmetic boundary, tokenizer logic, input IDs, execution buckets, pooling
and normalization. BF16 weight metadata is promoted to F32 in this reference.
Compilation/first evaluation is excluded from steady-state samples and recorded
separately. Dynamic IDs, lengths and masks remain graph inputs.

Each scope used Tessery→MLX followed by MLX→Tessery, in fresh engine processes.
The second run also reversed the seeded case order. Each of nine cases had
30 samples for each of two randomized identical A/B labels per engine process,
after at least three warmups and 100 ms. In total, four paired runs produced
4,320 timed batch calls, plus warmups and validation. GPU jobs ran sequentially.
A/B labels control within-process noise; they are not interleaved engine calls.

API scope measures full `encode()`. Backend scope prepares tokenization and batch
packing before timing; it includes forward, input conversion, readback and output
assembly. It is not a GPU-only timer. Prepared output is also checked against the
public API. Python 3.12.13 and NumPy 2.5.2 match. Regex differs: Tessery 2025.9.18,
MLX environment 2026.9.3; API results therefore include this environment difference.

## Results

Ranges are the two process medians, **not confidence intervals**. Ratios are
**Tessery latency / MLX latency**: below 1 favors Tessery. The millisecond columns
are full API timings; backend ratios are independently measured in their own runs.

| Token lengths | Tessery API p50, ms | MLX API p50, ms | API ratio | Backend ratio |
|---|---:|---:|---:|---:|
| 3 | 16.47–16.51 | 11.96–12.15 | 1.358–1.377 | 1.308–1.352 |
| 7 | 18.53–18.59 | 25.18–25.47 | 0.730–0.736 | 0.679–0.701 |
| 24 | 36.32–36.86 | 39.31–39.77 | 0.913–0.938 | 0.900–0.906 |
| 128 | 128.99–129.34 | 103.14–103.28 | 1.251–1.252 | 1.244–1.256 |
| 160 | 161.14–161.30 | 120.77–121.52 | 1.327–1.334 | 1.328–1.338 |
| 256 | 260.70–261.35 | 192.00–192.11 | 1.358–1.360 | 1.366–1.369 |
| 512 | 555.57–556.43 | 367.97–368.63 | 1.509–1.510 | 1.513–1.515 |
| 4 × 33 | 163.46–163.81 | 122.37–122.76 | 1.334–1.336 | 1.327–1.328 |
| 3, 7, 10 | 49.80–49.89 | 50.58–51.08 | 0.977–0.985 | 0.964–0.972 |

All 9/9 cases passed the numerical and timing screens in both scopes. The largest
observed process median drift was 1.62% for API and 3.66% for backend.
The historical screen allows 10% identical-control imbalance and 15% process
drift; passing it is not a significance test. The 1.5–2.3% API edge on the mixed
short case is best treated as near parity rather than a general speed advantage.

At 512 tokens, backend medians are approximately 547 ms for Tessery and 361 ms
for MLX, close to the API ratio. At 128 tokens the API gap is about 25%; at 512
it is about 51%. The previous long-M tile optimization improved Tessery relative
to its own baseline but does not establish an overall lead over compiled MLX.
Raw milliseconds from different dates are not used to infer an additional gain.

## Numerical and provenance checks

- Maximum cross-engine absolute vector difference: **7.674098e-07**, within
  unchanged `atol=5e-6`, `rtol=1e-4`.
- Repeated timed outputs are exact within each worker. Tessery outputs also match
  exactly across both orders and both scopes.
- Compiled versus uncompiled MLX dynamic-input validation passed, with maximum
  difference **2.682209e-07** and trace increments `[1,0,1,0]`.
  Warm measurements rejected retracing; API used nine graph specializations.
- Backend validation also creates executor-thread specializations, resulting in
  18 traces; its allocator values include those validation-only graphs and are
  not a minimal standalone backend memory estimate.
- All 97 frozen source/helper/dependency files were hash-checked before and after
  each run. Actual loaded Tessery native/shader hashes match the qualified snapshot.
- Exact selected-kernel totals were checked: 44,100 dispatches per API worker and
  45,500 per backend worker. Native-plan hits were present.

Full p50/p95 values, versions, fingerprints, memory diagnostics and raw-report
hashes are in [summary.json](../benchmarks/native-metal/compiled-comparison-20260913/summary.json).
Raw reports, orchestration and audit scripts are gitignored under
`artifacts/mlx-comparison-20260913/`. No models or dependencies were downloaded.
AC power was observed at launch; GPU clocks, thermals and background activity were
not controlled. No new BGE benchmark, runtime test suite or PyPI upload was run.
The existing 1351-test and 40-Shader-Validation evidence belongs to qualification.

## Next bounded experiments

The profiling and fused32 work below is now complete; see the
[follow-up report](FUSED32_EXPERIMENT_20260913.md). Both prototypes remain on an
isolated experiment branch, and main retains the qualified runtime.

1. **Profile the remaining quantized projections and fused MLP against MLX.**
   Current backend measurements locate the gap inside that path but do not
   identify its per-kernel shares. Refresh shape-level evidence before selecting
   the next kernel; older profile percentages must not be presented as current.
2. **Test 32-row weight reuse in fused gate/up + SiLU in its own worktree.**
   The ordinary linear tile already benefits from reuse. The fused kernel has
   additional shared state and accumulators, so register pressure may offset the
   benefit. Preserve K32 accumulation order, exact shape guards and old tolerances.
   Accept only after repeatable full-API improvement and numerical checks.
3. **Treat the 3-token path as a separate optimization target.** Its result cannot
   be inferred from the 7-token advantage. For attention, a future hypothesis
   should change actual data reuse or work, rather than repeat the rejected
   traversal-only experiment.

Reproduce one pair with the existing pinned interpreters and local model:

```sh
PYTHONPATH=src /path/to/tessery-python tools/benchmark_mlx_isolated.py \
  --model-dir /path/to/local-qwen --mlx-python /path/to/mlx-python \
  --mlx-mask causal --mlx-compile --timing-scope api \
  --lengths 3 7 24 128 160 256 512 --include-batches \
  --paired-controls --samples 30 --output artifacts/api-first.json
```

Repeat with `--engine-order mlx-first --reverse-cases` and a fresh output path;
compare with `tools/compare_embedding_runs.py`. Run `--timing-scope backend` as
its own pair. Never overlap the GPU workers or overwrite retained evidence.
