# Fused gate/up 32-row experiment

This branch preserves two unselected Metal prototypes and their measurement
harnesses. Production Python routing is unchanged. Neither prototype is enabled
by installing this branch; the benchmark installs a fixed experimental selector
before capturing each candidate model's native plans.

Baseline: `1f324e1` (runtime `d7edc5b`). Date: 2026-09-13.

- `gated4_32x32_k64`: 256 threads, two row fragments per SIMD group. Rejected
  because selected full-API cases were slower by about 4–6% in the initial run.
- `gated4_32x32_wide_k64`: 512 threads, one row fragment per SIMD group.
  Two symmetric forward/reverse runs found old/new API ratios of 1.0048–1.0090
  at selected shapes on Apple M1. Retained for research, deferred for production.
  Pipeline capacity must be checked with fallback before enabling this route on
  other devices; the existing native dispatch rejects unsupported group sizes.

Both use 16 KiB shared memory and preserve the old K32 accumulation sequence.
Selected experimental routing is exactly M128/160/256/512, N3072, K1024.
Do not generalize to tails or use `p.n` as an offset: these kernels have neither
tail masking nor offset support. M16/M24 and all other shapes retain old routing.

## Reproduction

Use an existing local Qwen3-Embedding-0.6B-4bit-DWQ model and Tessery development
environment. Run GPU jobs sequentially. Historical shader loading requires the
baseline commit in git history. Replace interpreter/model/output paths below.

```sh
PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python tools/benchmark_fused32_symmetric.py \
  --model-dir /path/to/local-qwen --samples 24 --output /path/to/first.json

PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python tools/benchmark_fused32_symmetric.py \
  --model-dir /path/to/local-qwen --samples 24 --reverse --output /path/to/repeat.json

METAL_INFERENCE_TEST=1 MTL_SHADER_VALIDATION=1 PYTHONPATH=src \
  /path/to/python -m pytest tests/real_metal/test_fused32.py -q
```

The 24 Shader Validation cases passed for both kernels with exact old/new arrays,
independent F64 checks and output sentinels. Full API timings used warmed native
plans, actual historical/current shader libraries, symmetric duplicated labels,
all 24 label permutations, nine cases and reverse allocation/case order.
The three-model exploratory harness is retained but has asymmetric model reuse;
use the symmetric harness for the small wide-kernel effect.

The main-branch report `docs/FUSED32_EXPERIMENT_20260913.md` and
`benchmarks/native-metal/fused32-20260913/summary.json` contain the decision and
evidence. Raw local artifacts and the read-only review are gitignored under
`artifacts/fused32-20260913/` in the primary checkout. No release qualification,
installed-wheel validation or non-M1 validation was performed for these kernels.
