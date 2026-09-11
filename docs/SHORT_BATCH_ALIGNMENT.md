# Short Qwen batch alignment

The public API keeps its length buckets and output ordering. When the existing
eight-token padding policy cannot fit a short two-input Qwen bucket, the
planner can use the next four-token boundary. Two sequences at that width
still produce a matrix height divisible by eight. The context limit, the
4096-token backend budget, and the limit of twice the shortest input all
remain enforced. BGE and larger buckets retain their existing policy.

For logical lengths `[3, 7, 10]`, the old plan was `([0], 3), ([1, 2], 10)`.
The new plan is `([0], 3), ([1, 2], 12)`: 27 execution tokens instead of 23,
with the same two GPU commands. The second matrix has 24 rows instead of 20.
Its seven projections in each of 28 layers replace a four-row scalar tail
with an eight-row tile. This removes 196 scalar-tail dispatches; the total
number of dispatches is unchanged. The three-token bucket keeps its existing
kernel path.

The implementation changes the execution planner only. It does not introduce
new shaders, weights, tokenization, pooling rules or public API parameters.
Logical token lengths still control attention masks and pooling. Padding can
select different arithmetic kernels, so the numerical gate remains
`atol=5e-6, rtol=1e-4`; bitwise equivalence across the old and new policies is
not required or claimed.

## Experiment and limits

`tools/benchmark_short_batches.py` freezes the planner from commit `26dba97`
and compares it with the current planner through public `encode()`. Identical
old-policy A/B controls and the selected route run in randomized order. Each
label warms for at least three calls and one second. Every measured output
must be bitwise reproducible within its own route and satisfy the unchanged
cross-route tolerance. Normal GPU command timestamps are collected separately
from optional intrusive per-kernel profiling; GPU duration overlaps the
submit/wait duration and must not be added to it.

A broader prototype aligned short buckets of any size by their total matrix
height. Exploratory runs of 14- and 30-input groups were slower with that
policy. The selected implementation is restricted to two-input buckets with
execution width below 128. It does not change grouping or relax resource
limits to obtain the measured benefit.

Two fresh-process qualification runs, numerical checks, and the direct MLX
comparison are recorded with their source hashes in the accompanying
benchmark artifacts. Thermals and clock frequencies are not controlled;
noise-screen failures are retained in the results.

## Paired full-API results

The local Qwen pack was reused on 2026-09-11 on an Apple M1 with 8 GiB memory.
Each fresh process collected 15
samples per label; the second process reversed case order. The screen requires
both old-policy A/B median ratios in `[0.9, 1.1]` and at most 15% median drift
between processes for each route. It is a noise heuristic, not a confidence
interval. Nine of eleven cases pass it.

| Logical lengths | Old/new latency ratio, two runs | Timing screen |
|---|---:|---|
| 3, 7, 10 | 1.182–1.290 | pass |
| 7, 10 | 1.331–1.357 | pass |
| 6, 9 | 1.217–1.411 | pass |
| 7, 11 | 1.449–1.517 | pass |
| 10, 18 | 1.197–1.316 | fail: both A/B controls |
| 30, 58 | 1.106–1.118 | pass |
| 46, 90 | 1.069–1.070 | pass |
| 62, 122 | 1.038–1.059 | pass |
| 7 (unchanged plan) | 1.005–1.065 | fail: first A/B control |
| 4 × 33 (unchanged plan) | 0.996–1.007 | pass |
| 5 × 7, 10 (unchanged plan) | 0.966–0.990 | pass |

For the target mixed batch, this corresponds to **15.4–22.5% less latency**.
The largest cross-policy vector difference across all cases is
`2.1234154701e-7`. Each process checks reproducibility within each route and
cross-route tolerances; this harness does not compare saved vectors between
processes. The separate isolated MLX harness does that comparison.

In the first target-case run, median GPU command time falls from about
176–186 ms (the two old-policy labels) to 136 ms. Command encoding stays around
13.5–13.9 ms. The independent intrusive profile ranks the old 20-row scalar
projection tails among the largest costs; those calls disappear in the new
plan. Intrusive timings must not be treated as normal API wall time.

Reproduce with an existing pack and a fresh output path:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_short_batches.py \
  --model-dir "$QWEN_MODEL_DIR" --samples 15 \
  --case 3,7,10 --case 7,10 --case 6,9 --case 7,11 \
  --case 10,18 --case 30,58 --case 46,90 --case 62,122 \
  --case 7 --case 33,33,33,33 --case 7,7,7,7,7,10 \
  --profile-first-case --output artifacts/short-batches-first-new.json
```

Repeat with `--reverse-cases`, a fresh output path, and without the optional
intrusive profile. The remaining three-token scalar projection path is the
next profiling target; this change does not optimize it.

## Direct MLX comparison

The existing independent MLX 0.32.2 F32 reference graph ran in separate,
sequential processes, with 10 samples per identical A/B label and both engine
orders. Both engines use the current API's identical token IDs and batch
plans, including the new 12-token width for the mixed case. This is a comparison
with that reference graph, not `mlx-embeddings` or a compiled MLX graph.

| Logical lengths | Tessery/MLX latency ratio | Timing screen |
|---|---:|---|
| 3, 7, 10 | 1.486–1.565 | pass |
| 4 × 33 | 1.500–1.525 | pass |
| 7 | 0.343–0.492 | fail: 1.61× Tessery process drift |

Tessery still trails MLX on the target mixed case. The earlier post-fusion
snapshot was 1.889–1.935×, but it used the old planner and was measured in a
different session. The same-session old/new API experiment above establishes
the change's benefit. The noisy seven-token row does not establish a new
advantage over MLX. All cross-engine and cross-process vector checks pass the
existing tolerance.

The maximum cross-engine vector difference in this snapshot is
`4.2095780373e-7`. Python 3.12.13 and NumPy 2.5.2 match between environments;
regex versions differ (2025.9.18 / 2026.9.3), so each comparison also explicitly
verifies identical tokenizer IDs.

Reproduce with `tools/benchmark_mlx_isolated.py --lengths 7 --include-batches
--mlx-mask causal --paired-controls --samples 10`, specifying the local pack,
existing MLX interpreter and a fresh output path. Repeat with
`--engine-order mlx-first --reverse-cases`, then use
`tools/compare_embedding_runs.py` to check the two result files.

## Evidence

Validation passed: 919 tests (94.60% Python coverage), 10 selected Qwen/BGE
full-model checks under Metal Shader Validation, static checks, dependency
policy and frozen input hashes. An isolated installation of the built wheel
passed Qwen/BGE smoke checks and five additional changed-plan cases at output
dimension 64, including permuted input order and comparison to unpadded
forward calls. Trimmed cache and active memory after close were zero. This is
short local qualification, not a new multi-hour soak or PyPI release.

- [First paired run and intrusive profile](../benchmarks/native-metal/short-batches-20260911/first.json)
- [Reversed paired run](../benchmarks/native-metal/short-batches-20260911/repeat.json)
- [Paired repeat screen](../benchmarks/native-metal/short-batches-20260911/summary.json)
- [Tessery-first MLX comparison](../benchmarks/native-metal/short-batches-20260911/mlx-first.json)
- [MLX-first comparison](../benchmarks/native-metal/short-batches-20260911/mlx-repeat.json)
- [MLX repeat screen](../benchmarks/native-metal/short-batches-20260911/mlx-summary.json)
- [Validation and local build hashes](../benchmarks/native-metal/short-batches-20260911/validation.json)
