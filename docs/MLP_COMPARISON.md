# Direct MLP projection comparison

This follows [mixed Qwen tiles](MIXED_QUANTIZED_TILES.md). It compares the
current Tessery dispatch against public MLX operations using the same verified
first-layer Qwen and BGE weights. No production kernel or public API is changed
by this measurement phase.

## Scope and method

`tools/benchmark_mlp_isolated.py` loads only the selected projection tensors
onto the GPU. It verifies the local pack with the existing profile before
extracting them. Qwen covers gate, up and down; BGE covers intermediate and
output matmul. BGE bias and both models' activation/residual operations are
excluded on both sides. This is resident matrix multiplication, not a full
MLP block, complete embedding request, or general MLX performance claim.

Inputs are deterministic normal-distributed F32 arrays, not captured hidden
states. Rows are projection rows (8/24/40/128/129/264/512), not necessarily
logical tokenizer lengths. These sample both incomplete tiles and the long
projection shapes identified by previous profiling. Only layer zero is
sampled; no claim is made about every layer or all input distributions.

Tessery uses its actual `_linear4`/`_matmul_f32` selector. MLX uses
[`quantized_matmul`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)
with transposed affine uint4 weights and group size 64, or F32 `matmul` for
BGE. Qwen's exact BF16 scale/bias values are promoted to F32 in MLX; Tessery
reads BF16 metadata in its kernel. Packed weight codes are identical. These
metadata-storage differences remain part of the measured implementations.

Every invocation submits a fresh operation and waits for it. MLX calls
[`eval`](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
on a fresh result each time, so the benchmark does not repeatedly time an
already evaluated array. Tessery creates and completes one command per
invocation. Resident inputs/weights, pipeline warmup and host output readback
are outside timing. Allocation/dispatch/synchronization costs inherent to each
engine remain inside; these are wall timings, not cross-engine GPU timestamps.
No graph compilation, lower-precision mode, upstream model implementation or
new runtime dependency is introduced.

Four sequential fresh processes run in Tessery/MLX/MLX/Tessery order. The
second pair reverses the seeded case order. Each case warms for at least
20 calls and 100 ms, then measures 20 samples per control label, with ten
synchronized invocations per sample. Labels A/B invoke the **same callable**
in randomized order. Their ratio detects local timing noise independently of
the actual engine comparison. CPU reference libraries use one thread.

Every output is checked against independent F64 matmul, then Tessery and MLX
outputs are compared in the parent process with unchanged
`atol=5e-5, rtol=5e-5`. Input and weight hashes must match between engines;
profile, runtime source and harness hashes are also checked. Temporary output
arrays used for this comparison are removed after the run, including failures.
No model or package is downloaded.

A separate timing screen requires all four A/B controls within 10% of unity
and each engine's two process medians within 15% (maximum/minimum ≤ 1.15).
These are declared noise-screening heuristics, not statistical confidence
intervals or numerical tolerances. Report status `passed` means correctness
and protocol checks passed; it does not mean every timing screen passed.
Thermals and background activity remain uncontrolled on this single M1 host.

## Results

The host is Apple M1 (8 GB), macOS 26.3. Both interpreters are Python
3.12.13 with NumPy 2.5.2; the installed MLX version is 0.32.2. Runtime source
hashes identify the implementation at commit `061333383d010ff8542fbce53e6a9b82698e5ae4`.

Ratios below are **Tessery time / MLX time** across the two process pairs:
values above one favor MLX. An asterisk marks a case that fails the timing
screen; its direction or range must not be treated as a stable result.

### QWEN

[Raw report](../benchmarks/native-metal/mlp-20260909/qwen.json): 9/21 cases pass the timing screen. All cases pass
F64 and cross-engine numerical checks; maximum cross-engine absolute difference
is 1.9073486e-05.

| Rows | gate_proj | up_proj | down_proj |
| ---: | ---: | ---: | ---: |
| 8 | 0.72–0.73* | 0.77–0.78* | 0.90–0.94* |
| 24 | 0.87–1.08* | 1.01–1.32* | 0.77–1.02* |
| 40 | 0.78–0.93* | 0.96–0.96* | 0.87–1.00* |
| 128 | 1.43–2.15* | 1.38–1.43 | 1.29–1.35 |
| 129 | 1.40–1.46 | 1.25–1.54* | 1.22–1.29 |
| 264 | 1.38–1.40 | 1.39–1.43 | 1.37–1.42 |
| 512 | 1.43–1.58 | 1.48–1.61 | 1.44–1.60* |

### BGE

[Raw report](../benchmarks/native-metal/mlp-20260909/bge.json): 1/14 cases pass the timing screen. All cases pass
F64 and cross-engine numerical checks; maximum cross-engine absolute difference
is 4.196167e-05.

| Rows | intermediate.dense | output.dense |
| ---: | ---: | ---: |
| 8 | 0.60–0.90* | 0.91–1.25* |
| 24 | 0.77–0.87* | 0.85–1.15* |
| 40 | 0.67–0.71* | 0.60–1.16* |
| 128 | 1.76–2.04* | 0.90–1.28* |
| 129 | 1.54–1.78* | 1.38–1.51* |
| 264 | 1.04–1.31* | 0.94–1.20* |
| 512 | 1.72–1.90* | 1.34–1.36 |

## Current full-model stage ranking

Fresh three-sample, 512-token profiles use the current execution policy:
[Qwen](../benchmarks/native-metal/mlp-20260909/qwen-profile.json) and
[BGE](../benchmarks/native-metal/mlp-20260909/bge-profile.json). Instrumented
outputs also pass the existing check against the ordinary public API.

| Share of summed instrumented stage medians | Qwen | BGE |
| --- | ---: | ---: |
| All projections | about 78% | about 60% |
| MLP gate/up or intermediate | about 31% | about 27% |
| MLP down/output | about 16% | about 16% |

Qwen's gate/up shape `(M,N,K) = (512,3072,1024)` runs 56 times per forward;
its down shape runs 28 times. BGE intermediate/output each run 12 times.
These profiles create a separate encoder per dispatch and are intrusive:
they support stage ranking, not precise production fractions or cross-engine
speedup estimates. The direct synchronized operation timings must not be
substituted into these profiles to predict a full-model acceleration.

## Selected next experiment

**Prioritize long Qwen gate/up projections.** Both have the same wide output
shape; the 264-row cases show a stable 1.38–1.43x Tessery/MLX latency ratio,
and the 512-row cases show 1.43–1.61x. Their combined stage share is larger
than the down projection's. The numerical checks pass on the actual local
weight tensors, and these particular timing cases pass the declared screen.

The first candidate is a bounded change to **threadgroup traversal order**
inside the existing 16-by-32 uint4 tile: group a few neighboring row tiles
for each output-channel tile, then compare with the current row-major launch
order. The hypothesis is better cache reuse of packed weight tiles across
row groups. This measurement does **not** establish cache misses, bandwidth,
barriers or arithmetic as the limiting hardware resource; GPU counter work
would be needed to distinguish them.

Keep the current reduction order, tile geometry, BF16 metadata and bounded
remainder handling. Test a small number of row-group widths, with the current
kernel as a same-process control. Reject candidates that fail exact old/new
outputs or unchanged F64 gates. Validate tail offsets with Metal Shader
Validation and require a repeated full-Qwen API gain on identical inputs
before enabling a new selector branch. Changing traversal is an experiment,
not a shipped optimization or a promised gain.

BGE remains the second priority. The 512-row output projection has a stable
1.34–1.36x gap, but only 1/14 BGE cases passes the timing screen. In particular,
the apparent 1.72–1.90x intermediate-projection gap at 512 rows fails the
control screen. It should be remeasured before it drives a kernel rewrite.
Neither model's apparent short-input wins justify a new dispatch guard.

## Validation and reproduction

The full portable suite passes **471 tests** (355 native tests deselected).
The six new benchmark tests check that the timing screen rejects noisy
identical-operation controls, cross-process drift, missing cases and invalid
timings while retaining a stable engine gap. Ruff, mypy, dependency and frozen
input checks pass. The native measurements cover 35 projection/row cases per
engine twice, with independent F64 and cross-engine output checks. No native
runtime source changes, release upload or new model downloads occur in this
phase; it does not repeat or replace multi-hour qualification.

```sh
.venv/bin/python tools/benchmark_mlp_isolated.py \
  --model-dir "$QWEN_MODEL_DIR" --mlx-python "$MLX_PYTHON" \
  --samples 20 --repeats 10 --output artifacts/qwen-mlp-new.json
.venv/bin/python tools/benchmark_mlp_isolated.py \
  --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --mlx-python "$MLX_PYTHON" --samples 20 --repeats 10 \
  --output artifacts/bge-mlp-new.json
.venv/bin/python -m pytest -m 'not metal'
```

Use fresh output paths and existing verified local model directories. The
optional MLX interpreter is development-only; importing `tessery` does not
import MLX. All raw timings, screening failures and provenance hashes are
retained in the linked reports.
