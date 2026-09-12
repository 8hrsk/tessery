# Down projection layout and row reuse — 2026-09-13

Two isolated changes to the Qwen uint4 down projection failed to improve the
target M512/N1024/K3072 kernel. Transposing decoded weights in shared memory
increased kernel latency by about **5.9–6.0%**. Doubling the output tile to
64 rows with 512 threads increased latency by **29.8–30.5%**. Both results
repeated on the target shape with stable identical-operation controls.

The transposed variant also slowed full `encode()` by **0.98–1.02%** at 512
tokens in two symmetric runs. Both candidates are rejected for production.

Neither prototype is enabled in the production runtime. The experiment is based
on `f6e9169`; the qualified runtime remains `d7edc5b`. Source and reproduction
commands are retained on branch
[`perf/down-projection-20260913`](https://github.com/8hrsk/tessery/tree/perf/down-projection-20260913),
commit `6e403ba87878d293f48171f368bbc7873c1b82b1`. No model or
dependency downloads were needed.

## Measurement boundary

Apple M1, 8 GiB unified memory, AC power, existing local
Qwen3-Embedding-0.6B-4bit-DWQ. Jobs ran sequentially. GPU clocks, background load
and thermals were not controlled. The reference is the current Tessery
`linear4_32x32_k64`, not MLX. The preceding compiled-MLX and intrusive profiling
reports remain separate evidence; this experiment does not revise their ratios.

Both candidates preserve F32 arithmetic, K32 partial accumulation order,
eight-value uint4 decode, BF16 scales/biases and the existing buffer ABI.

- **Transposed shared layout:** the same 32-row tile, 256 threads and 8 KiB
  shared storage. Decode writes a K×N tile; the right matrix load no longer
  requests a transpose. This changes layout without increasing row reuse.
- **64-row tile:** 512 threads and sixteen SIMD groups, each with two row
  fragments; decoded weights still occupy 8 KiB. The first 256 threads decode,
  and all threads reach both barriers. This changes reuse and scheduling while
  preserving the old shared layout. It requires a 512-thread-capable pipeline;
  no production capability selection/fallback was added.

The results do not prove a bandwidth or occupancy cause. Larger reuse and
unchanged shared-memory capacity alone were insufficient to improve this kernel.

## Kernel measurements

Two fresh processes reverse case order. Real first-layer down-projection weights
use deterministic synthetic F32 inputs. Each case has 24 samples per label,
five dispatches per command and at least 20 calls / 100 ms warmup per label.
Baseline A/B labels invoke the same current kernel on the same resident buffers.
Validation and readback are outside timing.

Ratios are **old latency / candidate latency**, so below 1 is slower. The
identical-control screen was set to ±2% before these measurements. It is a
noise filter, not a significance test.

| M | Transposed, first / reverse | 64 rows, first / reverse | Baseline A/B, first / reverse | Use |
|---|---:|---:|---:|---|
| 128 | 0.9412 / 0.9401 | 0.7450 / 0.7615 | 1.0546 / 1.0057 | first excluded |
| 256 | 0.8518 / 0.7703 | 0.7039 / 0.6409 | 1.1673 / 0.9266 | both excluded |
| **512** | **0.9434 / 0.9444** | **0.7666 / 0.7703** | **1.0017 / 1.0007** | target retained |

The noisier smaller cases remain recorded, without using them to estimate a
speed difference. On the target, both repeats show substantial regressions.
The 64-row candidate was rejected at the kernel stage. Only the less costly
transposed variant advanced to a full-API check.

## Full API check

The transposed candidate is selected only for M512/N1024/K3072. Other shapes
retain current routing and act as controls. Each fresh process holds two models,
each with two labels, fixed native plans and its own shader library. The baseline
loads the historical shader from `f6e9169`. Each case uses all 24 label
permutations, 24 samples per label and at least one second / three warmup calls.
The second process reverses model allocation and case order.

Together the runs contain **1,152 timed API calls**. Ratios below use pooled
baseline-label and candidate-label medians. They are two observed medians,
not confidence intervals.

| Token lengths | First API ratio | Reverse API ratio | Route |
|---|---:|---:|---|
| 128 | 1.00328 | 1.00300 | unchanged |
| 256 | 0.99786 | 1.00152 | unchanged |
| **512** | **0.99028** | **0.98989** | transposed down projection |
| 16 | 1.00451 | 0.99652 | unchanged |
| 3, 7, 10 | 1.00171 | 0.99579 | unchanged |
| 159, 160 | 0.99989 | 1.00031 | unchanged, execution rows 320 |

Normal command GPU ratios on the target were 0.99054 and 0.98928, consistent
with the API regression. All four six-round time-block ratios in both target
runs were below 1.0. Those blocks are descriptive diagnostics, not independent
replications. Control-case ratios stay near 1.0 and several change direction
between runs; no general speedup is inferred from their small shifts.

Every timed output matches the baseline exactly, including across processes.
The harness and final audit verify native-plan hits with no timed builds and
all dispatch dictionaries after normalizing the replaced kernel name. Selected
calls contain exactly 28 experimental down projections; all other projections,
fused MLP, attention and fallback dispatches retain baseline counts.

All 91 frozen source/helper files matched their hashes after each process and
at final audit. Actual loaded shader hashes and model profile identities match
the intended old/new pair. Sampled memory pressure stayed normal and the harness
verified zero owned active/cache/plan bytes after model close.
System swap did not grow; peak process RSS was about 1,196 MiB including model
loading. The API environment used Python 3.12.13, NumPy 2.5.2 and regex 2025.9.18.

## Correctness and retained evidence

16 tests passed with Metal Shader Validation, each exercising both prototypes.
They cover complete tiles at M64/128/256/512, multiple output-column tiles,
integer inputs, cancellation, large row-dependent values and zero. Tests caught
an incorrect column step in the initial 64-row prototype; it was fixed before
any performance measurement. The failed first log is retained alongside the
successful corrected run.

Every tested candidate array matches the current kernel exactly. Independent
F64 dequantized matmul passes unchanged `atol=5e-5`, `rtol=5e-5`; the maximum
real-weight synthetic-input absolute error was **4.014746e-6**. Each validation
gets a fresh NaN-filled output and trailing sentinel rows. Input and weight
buffers are checked unchanged. This is bounded coverage, not a proof for every
activation, tail or Metal device.

The [machine-readable summary](../benchmarks/native-metal/down-projection-20260913/summary.json)
retains timing samples, controls, fingerprints and raw report hashes. Raw logs,
orchestration and the final audit script are gitignored under
`artifacts/down-projection-20260913/`. The experiment's Python formatting and
lint checks passed. No production host selector or native bridge was changed.
No candidate release wheel, multi-hour soak, non-M1 run or PyPI upload was made.

## Next bounded hypothesis

Keep the current 32-row tile and examine K-loop work separately. A K128 decode
stage could reduce barrier frequency while retaining independent K32 partial
sums, but it doubles shared storage to 16 KiB and can regress occupancy. It
needs its own isolated exact/F64 and full-API experiment. These negative results
do not justify widening tile guards or changing numerical tolerances.
