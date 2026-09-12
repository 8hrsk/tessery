# Long-Qwen profiling and fused32 experiment — 2026-09-13

The 256-thread fused32 prototype is slower and is rejected. The 512-thread
prototype produces a small repeatable full-API improvement on selected long
inputs: old/new latency ratios of **1.0048–1.0090** in symmetric forward/reverse
runs. It remains an experiment rather than a production change. The gain is
below 1% on one Apple M1, unchanged-route controls also move by sub-percent
amounts, and safe pipeline-capability selection/fallback is not implemented.
This is a decision to defer a small positive result, not a numerical failure.

Main retains qualified runtime `d7edc5bcda8926b411f5dff36b90ec8446b9187d`.
Experiment baseline is `1f324e1b61aa650ef3594d9b0ef87c7f1a097364`, whose runtime
is identical. Prototype source, tests and harnesses are preserved separately in
[`perf/fused32-20260913`](https://github.com/8hrsk/tessery/tree/perf/fused32-20260913),
commit `1c1ab933dd1414fdfd19156d5e0c0970af06ba7c`. Neither prototype is selected
by normal Python routing, even on that branch.

## Refreshed profile

Local Qwen3-Embedding-0.6B-4bit-DWQ, Apple M1, 8 GiB unified memory. Five
samples per length, current selected execution policy. The profiler inserts
per-dispatch timestamps and changes encoder boundaries. These are shares of the
sum of intrusive kernel-group medians, **not shares of normal `encode()` time**.

| Tokens | Ordinary linear, ms / share | Fused gate/up, ms / share | Attention, ms / share |
|---|---:|---:|---:|
| 128 | 66.26 / 52.4% | 44.77 / 35.4% | 7.63 / 6.0% |
| 160 | 78.99 / 50.3% | 54.11 / 34.4% | 11.17 / 7.1% |
| 256 | 130.30 / 51.0% | 87.00 / 34.1% | 25.45 / 10.0% |
| 512 | 253.97 / 46.3% | 172.96 / 31.6% | 95.69 / 17.5% |

Ordinary quantized projections remain the largest aggregate target. The fused
gate/up projection is the largest individual projection shape. These timings
do not distinguish bandwidth, register pressure or occupancy; those remain
hypotheses requiring targeted experiments or hardware counters.

The independent projection comparison used four Tessery/MLX/MLX/Tessery worker
processes, real first-layer weights and synthetic F32 activations. It measured
one public MLX quantized matmul per synchronized invocation, **not a compiled
whole graph**. All numerical comparisons passed; only 6/12 shape/row cases
passed the existing timing screen. Retained Tessery/MLX latency ranges:

| Projection | Rows | Tessery / MLX |
|---|---:|---:|
| down | 128 | 1.022–1.136 |
| down | 160 | 1.080–1.122 |
| down | 512 | 1.345–1.402 |
| gate | 128 | 1.348–1.390 |
| gate | 512 | 1.307–1.315 |
| up | 512 | 1.336–1.339 |

Excluded timings remain in the evidence with their screen outcomes. They must
not be used as accepted speed claims. Gate/up here are separate projections;
this does not compare Tessery's fused gate/up/SiLU with an equivalent MLX fusion.
The preceding [compiled whole-model comparison](COMPILED_MLX_COMPARISON_20260913.md)
remains the appropriate source for the full Tessery-versus-MLX gap.

## Two isolated prototypes

Both kernels compute a 32×32 output tile with K64 weight decode and preserve
the old K32 partial accumulation order and SiLU expression. Shared storage stays
at 16 KiB. Both use the existing eight-buffer ABI.

- **256 threads:** each SIMD group computes two row fragments, increasing its
  accumulator state. On stable microbenchmark cases M160/256/512, old/new GPU
  ratios were about 0.913/0.908/0.905. Full API selected-case ratios in the initial
  run were 0.940–0.963. This version does not qualify for promotion.
- **512 threads:** sixteen SIMD groups each compute one fragment; only the
  first 256 threads decode weights and all threads reach both barriers. Stable
  M160/256/512 microbenchmark ratios were 1.037/1.021/1.022. These modest kernel
  gains motivated the symmetric full-API follow-up.

Microbenchmarks used 24 samples per label, five dispatches per command and at
least 20 warmup calls / 100 ms. M128 in the two-candidate micro run had a 1.248
identical-control ratio and is excluded. The initial single-candidate M160
micro run is also excluded. Their raw values are retained without treating
them as evidence of either a win or a loss. Thermals were not controlled.

## Full API evidence for the 512-thread variant

Two resident models, each with two labels, use separate fixed native plans.
The baseline loads the actual historical shader; the candidate loads the new
shader. Both retain current ordinary linear32 routing. Each of nine cases uses
all 24 label permutations, with 24 timed calls per label after at least one
second and three warmups. The second fresh process reverses model allocation
and case order. The two runs contain **1,728 timed `encode()` calls**.

Ratios below are pooled baseline latency / pooled candidate latency; above 1
favors the candidate. These two medians are **not confidence intervals**.

| Token lengths | First API ratio | Reverse API ratio | Route |
|---|---:|---:|---|
| 128 | 1.00476 | 1.00842 | candidate |
| 160 | 1.00569 | 1.00886 | candidate |
| 256 | 1.00734 | 1.00774 | candidate |
| 512 | 1.00810 | 1.00882 | candidate |
| 4 × 33 | 1.00716 | 1.00897 | candidate, execution rows 160 |
| 16 | 0.99285 | 0.99620 | unchanged |
| 24 | 0.99517 | 0.99961 | unchanged |
| 3, 7, 10 | 0.99342 | 0.99925 | unchanged |
| 159, 160 | 0.99970 | 1.00020 | unchanged, execution rows 320 |

Normal command GPU timings also favor the candidate at all selected cases
(ratios 1.0058–1.0101). Six-round time-block diagnostics are preserved: some
first-run 128 and 4×33 blocks cross 1.0, despite positive pooled medians.
They are descriptive checks, not independent replications or a significance
test. Identical-label controls alone cannot explain away or prove a sub-percent
effect; unchanged routes and reversed allocation provide additional context.

An earlier three-model exploratory run contained another 864 timed calls.
It gave the wide variant ratios of 1.006–1.012 at selected cases except 512,
which was 0.984. The reviewer identified asymmetric model reuse: baseline had
two labels while each candidate had one. The symmetric follow-up fixes this.
The first run remains evidence of sensitivity to benchmark state; it is not
pooled with the symmetric runs or used to cancel their positive result.

## Validation and provenance

- 24 GPU test cases passed under Metal Shader Validation, each checking both
  kernels. Rows 32/64/128/160/256/512 cover random integer inputs, cancellation,
  saturation and zero. Separate real-weight synthetic-F32 micro cases passed.
- Old/new kernel arrays match exactly; independent dequantized F64 references
  pass unchanged `atol=5e-5`, `rtol=5e-5`. Fresh NaN outputs and trailing guard
  rows detect incomplete stores and overruns. This is bounded tested coverage,
  not a proof for every possible activation or device.
- Every timed model output matches the historical baseline exactly, including
  cross-process output checks. Native-plan hits are present with no timed builds.
- A final audit checked all **2,592 timed API calls**. After replacing candidate
  gated counts with old gated16 counts, every dispatch dictionary matches its
  baseline, including ordinary linear, attention and fallback paths. Selected
  buckets have exactly 28 candidate dispatches.
- All 92 frozen source/helper files for the symmetric pair match their final
  hashes. Loaded shader and profile identities are recorded. No source file
  changed while GPU workers ran.
- Sampled memory pressure stayed normal; system swap did not grow. Peak process
  RSS was about 1,196 MiB for the symmetric runs, including model-loading peaks.
  The harness asserts zero owned active/cache/plan bytes after closing models.
- Ruff checks and formatting passed for the four new Python files. A read-only
  subagent reviewed kernels and both full-API harness designs; its only write
  was the separate gitignored `REVIEW.md`.

The [machine-readable summary](../benchmarks/native-metal/fused32-20260913/summary.json)
retains timings, controls, input/vector fingerprints, normalized dispatches,
source hashes and raw artifact SHA256 values. Raw reports, validation logs and
audit/orchestration scripts remain gitignored in `artifacts/fused32-20260913/`.
Python 3.12.13 and NumPy 2.5.2 were used; MLX was 0.32.2. No models or dependencies
were downloaded. GPU work was sequential, on AC power; background activity and
clocks were uncontrolled. No candidate wheel, long soak, BGE or non-M1 run was
performed, and no new PyPI package was published.

## Decision and next targets

Keep the qualified main runtime. Retain wide fused32 as a small positive
experiment: promotion requires pipeline-specific 512-thread eligibility with
fallback, production route/plan tests and broader qualification. The native
runtime currently rejects dispatches above a pipeline's maximum threadgroup
size, so a successful M1 run alone is not a portable selection rule.

The next high-value experiment is shape-specific ordinary uint4 projection
tuning, starting with down projection at M512/N1024/K3072. Test decode layout
and output-tile reuse separately, preserving arithmetic and the current route
for other shapes. Measure full `encode()` alongside kernel time and unchanged
controls. At 512 tokens, attention data reuse is the second independent target;
a traversal-only rearrangement is not sufficient. Short 3-token inference
remains a separate workload, as established by the compiled comparison.
