# Three-row Qwen projections

`linear4_small3` specializes the existing scalar uint4 projection for exactly
three execution rows. Its five admitted `(output channels, input channels)`
pairs are `(1024,1024)`, `(2048,1024)`, `(3072,1024)`, `(1024,2048)` and
`(1024,3072)`. Every other shape retains the previous dispatcher, including
small remainders after full matrix tiles. This changes neither batch planning
nor the public Python API.

One 32-thread SIMD group still computes one output channel. The specialization
uses three accumulators and omits the fourth-row bounds checks. It loads each
BF16 scale and bias once for two consecutive 32-element steps in the same
64-element quantization group. Each lane consumes the same input and decoded
weight sequence as `linear4`, and uses the same final `simd_sum`. It does not
change accumulation precision, reassociate sums, materialize dense weights,
or add workspace buffers.

The host guard is necessary: the shader writes exactly three rows and assumes
the registered 64-element quantization groups. The selected projection shapes
all have input sizes divisible by 64. Random, cancellation and boundary-input
tests compare against an independent F64 product and the previous scalar
kernel, with output sentinels and adjacent two-/four-row fallback cases.

## Method

Preliminary experiments compared the existing incomplete matrix tile and
four-/eight-SIMD-group versions of the old scalar kernel. Those routes did not
show a consistent full-model benefit and are not selected. Removing dynamic
row checks and reusing quantization metadata produced the promising result.

`tools/benchmark_small_projections.py` measures full public `encode()` calls
against the previous scalar dispatch from `1435226`, using the same current
library, weights, tokenizer and batch plans. It includes identical old-path
A/B controls and checks every measured vector for bitwise equality with the
baseline. The two processes also save vectors for an exact repeat check.

Initial 15-sample randomized runs produced noisy A/B controls on short inputs.
The qualification protocol therefore uses 30 samples per label, in shuffled
blocks containing all six permutations of A/B/selected. Every label occupies
each position equally often. Each label warms for at least three calls and one
second, and the second process reverses case order. The timing screen requires
both A/B ratios in `[0.9,1.1]` and at most 15% per-route median drift across
processes. This is a noise heuristic, not a confidence interval. Normal GPU
command timings overlap submit/wait duration and must not be added to it.

Reproduce with the existing local pack and a fresh output path:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_small_projections.py \
  --model-dir "$QWEN_MODEL_DIR" --samples 30 \
  --output artifacts/small-projections-first-new.json
```

Repeat with `--reverse-cases` and a different output path. All local experiments
reuse the existing model packs without downloads.

## Paired full-model results

On the local Apple M1 (8 GiB) and pinned Qwen pack, both balanced runs pass the
timing screen for all seven cases. Every measured vector matches the old
kernel bitwise, including between the two processes.

| Logical lengths | Old/new latency ratio, two runs | Dispatch |
|---|---:|---|
| 3 | 1.495–1.589 | 196 specialized projections |
| 3, 7, 10 | 1.217–1.280 | 196 specialized projections |
| 10, 3, 7 | 1.195–1.305 | 196 specialized projections |
| 2 | 0.991–1.005 | unchanged |
| 4 | 0.900–0.964 | unchanged |
| 7 | 0.976–0.986 | unchanged |
| 3, 3 | 0.986–1.018 | unchanged: six execution rows |

The target mixed batch has **17.8–21.9% less latency** than the preceding
implementation. A single three-token input has **33.1–37.1% less latency**.
The four-token control shows an apparent slowdown despite identical GPU work;
passing this heuristic screen does not establish a causal performance change
for controls. For unchanged shapes both labels use the current dispatcher.

In the first balanced run, the single-input median GPU time falls from
56–57 ms in the two old-policy labels to 34.5 ms; command encoding stays about
5.7 ms. The mixed-batch GPU medians fall from 138–142 ms to 106 ms. These are
normal command timestamps, not intrusive per-kernel profiles.

## Direct MLX comparison

Fresh sequential processes ran the independent MLX 0.32.2 F32 reference graph
in both engine orders, with identical token IDs, plans and paired A/B labels.
This is not `mlx-embeddings` or a compiled MLX graph.

| Logical lengths | Tessery/MLX latency ratio | Timing screen |
|---|---:|---|
| 3, 7, 10 | 1.108–1.153 | pass |
| 4 × 33 | 1.483–1.536 | pass |
| 3 | 1.518–2.899 | fail: 1.91× MLX process drift |
| 7 | 0.368–0.533 | fail: local control and Tessery process drift |

The mixed case still trails MLX by about 11–15% latency in this snapshot.
The earlier 1.486–1.565× snapshot used the preceding kernel revision and a
different measurement session; the paired old/new test above establishes the
benefit of this change. The very short direct comparisons are noisy and do
not support a precise gap or an advantage claim. Cross-engine and cross-process
vector comparisons use the existing `atol=5e-6, rtol=1e-4` gate.

Run `tools/benchmark_mlx_isolated.py --lengths 3 7 --include-batches
--mlx-mask causal --paired-controls --samples 10`, specifying the local model,
MLX interpreter and a fresh output path. Repeat with
`--engine-order mlx-first --reverse-cases`, and check both files with
`tools/compare_embedding_runs.py`.

## Validation and evidence

All 961 tests pass, with 94.64% Python coverage. The 27 selected Shader
Validation cases cover the five projection shapes, cancellation, 32-/64-element
boundaries, output guards, adjacent fallback heights, and three distinct texts
at output dimensions 32 and 1024. Static checks, dependency policy and frozen
model inputs pass.

The built wheel passes isolated Qwen/BGE smoke tests and six additional cases
at dimension 64. Those assert the new kernel's dispatch count, fallback routing,
and bitwise equality with the old scalar kernel. Active and cached GPU memory
are zero after close. The maximum cross-engine vector difference is
`4.2095780373e-7`. This short qualification does not repeat multi-hour soak or
publish a new PyPI version.

- [Balanced first run](../benchmarks/native-metal/small-projections-20260911/balanced-first.json)
- [Balanced reversed run](../benchmarks/native-metal/small-projections-20260911/balanced-repeat.json)
- [Repeat and vector checks](../benchmarks/native-metal/small-projections-20260911/balanced-summary.json)
- [Tessery-first MLX comparison](../benchmarks/native-metal/small-projections-20260911/mlx-first.json)
- [MLX-first comparison](../benchmarks/native-metal/small-projections-20260911/mlx-repeat.json)
- [MLX repeat screen](../benchmarks/native-metal/small-projections-20260911/mlx-summary.json)
- [Validation and build hashes](../benchmarks/native-metal/small-projections-20260911/validation.json)
