# Down projection layout/tile experiment

This branch preserves two unselected prototypes based on `f6e9169` (qualified
runtime `d7edc5b`). Production Python routing remains unchanged. Measurement
helpers install experimental selectors on their own model objects before native
plan capture. Installing this branch does not activate either new kernel.

`linear4_32x32_transposed_k64` transposes the decoded weight tile in shared
memory, retaining the current 32-row tile, 256 threads, 8 KiB shared memory and
two independent row accumulators per SIMD group. It loads the right matrix
without a transpose operation. K32 partial reduction order remains unchanged.

`linear4_64x32_wide_k64` uses 512 threads, 16 SIMD groups, two row fragments per
group and 8 KiB shared memory. The first 256 threads decode weights; every thread
reaches both barriers. The tile covers 64 rows and 32 columns, with row offsets
0..24 and 32..56. The pipeline must support 512 threads. No portable production
fallback is implemented here, and these kernels do not support tails/offsets.

The target is M512/N1024/K3072. Two stable kernel runs found old/new GPU ratios
about 0.944 for the transposed layout and 0.767–0.770 for the 64-row tile.
Both are slower than the current kernel; neither should be enabled on these
results. Smaller exploratory cases and noisy controls are retained in reports.

16 Shader Validation cases exercise both candidates, exact baseline equality,
unchanged F64 tolerances, fresh NaN outputs, trailing sentinels and unchanged
input/weight buffers. Tests caught and led to fixing a column-step error in the
first 64-row prototype before performance measurements.

## Reproduction

Use the existing local Qwen DWQ model and a development Python environment.
Run GPU jobs sequentially, using fresh output files. Reversed runs use the same
source files. Full API defaults to only the transposed candidate, replacing
exactly the M512/N1024/K3072 route. Other shapes are controls. The 64-row
candidate was rejected at the kernel stage and does not need a model run.

```sh
METAL_INFERENCE_TEST=1 MTL_SHADER_VALIDATION=1 PYTHONPATH=src \
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python -m pytest tests/real_metal/test_down_projection.py -q

PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python tools/benchmark_down_projection.py \
  --model-dir /path/to/local-qwen --output /path/to/kernel-first.json

PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python tools/benchmark_down_projection.py --reverse \
  --model-dir /path/to/local-qwen --output /path/to/kernel-repeat.json

PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python tools/benchmark_down_projection_model.py \
  --model-dir /path/to/local-qwen --output /path/to/api-first.json
```

Repeat the API command with `--reverse` and a fresh output path. It uses two
resident models, two labels per model and all 24 label permutations, and reverses
model allocation and case order in the second process. It checks exact vectors,
plan hits without builds, all dispatch counters after target replacement,
memory pressure, bounded swap growth and zero owned GPU resources after close.
The baseline shader is read from pinned git history, so that commit is required.

The primary checkout retains gitignored raw data in
`artifacts/down-projection-20260913/`. The main-branch report is
`docs/DOWN_PROJECTION_EXPERIMENT_20260913.md`, with checked-in timing evidence
under `benchmarks/native-metal/down-projection-20260913/summary.json`.
