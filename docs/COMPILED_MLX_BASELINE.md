# Compiled MLX reference and backend timings

The isolated comparison now accepts `--mlx-compile` and `--timing-scope backend`.
The default remains the historical uncompiled full `encode()` comparison. MLX is
still a development-only dependency; Tessery runtime and its wheel are unchanged.

Both Qwen and BGE have a pure MLX graph boundary. Token conversion, validation,
mask selection, evaluation and NumPy readback remain outside tracing. Token IDs,
lengths and additive masks are dynamic arrays; weights and model configuration
are fixed for the backend lifetime. Full-length causal Qwen and unmasked BGE
use separate specializations from additive masked inputs.

The cache is keyed by execution thread, array shapes/dtypes, mask kind and output
dimensions. The thread key matters: public `encode()` owns an executor thread,
whereas direct backend measurements run on the benchmark thread. Changing array
values at an existing shape must reuse the graph. Python trace counters reject
retracing of warm entries and any tracing during steady-state measurement.
Closing the reference clears compiled functions before weights.

Each specialization records `first_call_seconds` separately. This is tracing,
compilation **and first execution/readback**, not an isolated compiler duration.
It is excluded from steady-state timings. Warmups still apply independently.
The implementation follows [MLX compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

Backend timing uses the same length buckets, token IDs and pad values as the API,
prepared before timing. It includes forward calls, host/device conversion and
result assembly, but excludes tokenization, admission and batch packing. A
separate exact comparison checks the prepared route against public `encode()`.
The parent validates IDs and plans across engines; comparisons reject mismatched
compile modes, timing scopes and helper hashes. Full API timing still includes
the known regex-version difference between local environments.

Validation on Apple M1 / MLX 0.32.2, 2026-09-12:

- Eleven CPU harness tests passed, including changed-scope identity rejection,
  ragged bucket padding/order, cache reuse and retracing rejection.
- Real Qwen and BGE graphs passed changing IDs/lengths/masks at fixed shape,
  new shape, then old shape again. Trace increments were `[1,0,1,0]`.
  Compiled vs uncompiled maximum absolute errors: Qwen `2.68221e-7`,
  BGE `1.49012e-7` (`atol=5e-6`, `rtol=1e-4`).
- Isolated compiled backend smoke: five Qwen cases and four BGE cases; exact
  repeated outputs and cross-engine tolerance passed. Qwen full API smoke also
  passed five cases. Maximum cross-engine difference was `4.99189e-7` Qwen and
  `2.68221e-7` BGE. These five-sample runs validate the harness; they do not
  establish comparative performance.

Reproduce dynamic-input validation with the existing MLX interpreter:

```sh
PYTHONPATH=src /path/to/mlx-python tools/validate_mlx_compile.py \
  --model-dir /path/to/local-qwen --output artifacts/compiled-validation.json
```

For BGE also pass its cache manifest via `--profile-file`. For a measured run:

```sh
PYTHONPATH=src .venv/bin/python tools/benchmark_mlx_isolated.py \
  --model-dir /path/to/local-qwen --mlx-python /path/to/mlx-python \
  --mlx-mask causal --mlx-compile --timing-scope backend \
  --lengths 3 7 17 24 159 160 256 512 --include-batches \
  --paired-controls --samples 30 --output artifacts/compiled-first.json
```

Repeat in a fresh process with `--engine-order mlx-first --reverse-cases` and a
new output path. Use `tools/compare_embedding_runs.py` on the two reports. Run
API scope separately; do not mix scopes when computing a speed ratio. Preserve
uncompiled baselines in separate outputs. All engines must run sequentially.

In backend scope, the exact API/prepared-route validation also creates an
executor-thread specialization. Allocator diagnostics therefore include that
validation-only graph alongside the directly measured thread's graph; they are
not a minimal standalone compiled-backend memory measurement.

Measured comparison of the subsequently qualified long-Qwen runtime:
[2026-09-13 results](COMPILED_MLX_COMPARISON_20260913.md), with separate API/backend
pairs and all nine cases passing numerical and timing screens.
