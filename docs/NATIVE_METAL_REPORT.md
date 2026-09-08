# Native Metal alpha verification — 2026-09-08

The packaged engine ran Qwen3-Embedding-0.6B directly on Metal using the existing
335 MB 4-bit model on macOS 26.3 arm64 / Python 3.12.13. No new model weights were
downloaded. The installed wheel imports no MLX, torch, transformers or Yuri code.

## Validation

Follow-up tensor API validation (same local device): **204 tests passed** with
**95.52%** combined Python statement/branch coverage. Added checks cover chained
device-resident operations with host reads disabled until the final result,
scalar/multidimensional addition and SiLU, matrix transpose, noncontiguous
uploads, copy isolation, cross-runtime rejection, closed resources, parallel
call serialization and cleanup after injected command failures. The full pinned
Qwen3 suite also passed after the buffer lifetime changes. The timings and wheel
hash below describe the earlier 192-test snapshot, not a new performance run.

Initial alpha snapshot:

* 192 tests passed, including actual Metal numerical tests and model inference.
* Python statement/branch combined coverage: 94.95%. Native C++/Metal code is
  checked numerically, not included in that coverage percentage.
* Float32 addition/matmul, uint4 embedding/linear, RMSNorm, RoPE, grouped causal
  attention, SiLU and pooling/projection checked against independent NumPy formulas.
* All 18 frozen tokenizer/mask observations matched the own BPE implementation.
* Real-model tests cover RU and EN-to-RU semantic margins, deterministic repeat,
  batch order, 32/384/1024 dimensions, 511/512/513 boundaries, batch 32 and release
  of temporary GPU buffers. This is short-run resource validation, not a soak.
* Async tests cover admission, overload, cancellation before and during work,
  discarded results, safe errors and lifecycle.
* Wheel installed with `--offline --no-index` using hash-checked staged dependencies.
  Native matmul and real embeddings then ran with `python -I` in a separate venv
  whose path contains spaces/Unicode, with Python socket access denied. This is
  not an OS-level network sandbox certification or a bundled CPython relocation test.
* Mach-O deployment target is 14.0; dylib ID is relative (`@rpath/_native.dylib`).
  The bridge links only Apple system frameworks/libraries. The inspected NumPy
  wheel uses Accelerate; generic NumPy GCC/OpenBLAS notice entries do not match
  libraries present in this target wheel. Full notices are preserved.

## Diagnostic performance

One host, batch 1, 32 tokens, 384 dimensions, two warmup iterations and five warm
samples, measured through the public API in the installed wheel. No concurrent
GPU tests were running. Thermal/power conditions were not controlled.

| Metric | Observation |
|---|---:|
| Model/tokenizer load | 0.609 s |
| First encode | 0.334 s |
| Warm p50 | 0.582 s |
| Warm p95 | 0.606 s |
| Warm throughput | 1.71 texts/s |
| Resident owned GPU weight buffers | 335,218,496 bytes |
| Peak owned GPU buffers for this case | 337,186,244 bytes |

[Raw timings and code hashes](../benchmarks/native-metal/diagnostic-20260908.json).
These are alpha diagnostics, not statistical performance acceptance or parity
with MLX. Cold load, long sequences, batching and memory need broader benchmarks.

## Scope

Working standalone embedding inference and compute primitives, Apache-2.0.
Currently one pinned model; no general autograd/training, generation, HTTP/UDS,
ANN index or embedded CPython distribution. Legacy Yuri space compatibility is
not claimed; the new identifier is `metal-inference-qwen3-f32-v1`.
