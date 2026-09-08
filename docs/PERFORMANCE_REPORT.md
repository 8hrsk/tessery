# Metal performance and reliability observations, 0.4 alpha

Date: 2026-09-08. Device: Apple M1, macOS 26.3 arm64, Python 3.12.13. Existing
Qwen3 and BGE weights were reused in place; no model files were downloaded or copied.

## BGE before/after matrix

Baseline: d562ddd implementation plus command timing counters. Candidate: short-K
tiled F32 matmul, length buckets and common sync/async submission queue. Each cell
has two warmup and seven measured samples in separate sequential processes. These
are diagnostic measurements with uncontrolled power/thermal/background load;
cold load differences are not interpreted as an optimization result. The initial
BGE records contain workspace hashes taken at startup. Later tools also capture
the shader bytes used by the runtime and reject changes under src/metal_inference.

| Batch | Tokens/text | Before p50 (ms) | After p50 (ms) | Before/after |
|---:|---:|---:|---:|---:|
| 1 | 32 | 15.46 | 16.16 | 0.96x |
| 1 | 128 | 113.53 | 78.00 | 1.46x |
| 1 | 512 | 734.01 | 583.95 | 1.26x |
| 4 | 32 | 100.29 | 64.47 | 1.56x |
| 4 | 128 | 422.24 | 280.80 | 1.50x |
| 4 | 512 | 2816.37 | 2271.67 | 1.24x |
| 16 | 32 | 348.96 | 218.69 | 1.60x |
| 16 | 128 | 1723.36 | 1148.60 | 1.50x |
| 16 | 512 | 11523.17 | 9268.67 | 1.24x |
| 32 | 32 | 697.29 | 440.67 | 1.58x |
| 32 | 128 | 3426.06 | 2266.94 | 1.51x |
| 32 | 512 | 23477.32 | 18746.25 | 1.25x |

Eleven of twelve observed cells improve by about 1.24–1.60x. The 1x32 case
does not improve (about 4% slower in this run). Raw samples and p95 are retained;
these runs do not demonstrate speed parity with MLX. A separate randomized paired
matmul experiment isolates the kernel effect on the same runtime.

## Kernel precision gate

The selected aligned short-K tile improves GPU median times in the observed
paired experiment. F32 operands and accumulators are retained. The K=1536
candidate is excluded from automatic routing: one earlier test element exceeded
the fixed atol=5e-5, rtol=5e-5 criterion. The original reduction handles K>512.
The saved BGE reference corpus and generic tensor tests remain separate gates.

## Review disposition

The independent read-only Astra research and Sol medium review are in local
gitignored Markdown files under artifacts/research, and are not included here.

- SOL-1: partial command creation now clears native state; fault-injection regression added.
- SOL-2: length buckets cap padding and restore original row order.
- SOL-3: sync/async callers now share the same submission queue; deterministic ordering and close tests added.
- SOL-4: CPU tokenization cannot be interrupted mid-call. This limitation remains documented.
- Follow-up measurement findings: unavailable timestamps produce null speedup,
  runtime shader/library digests are recorded, and diagnostic checkpoints use atomic replacement.

Detailed reproduction and limits: [performance tools](PERFORMANCE.md).

## Bounded soak and lifecycle

| Model | Completed calls | Duration (s) | Owned bytes during soak | Owned bytes after close | RSS end-start (MiB) |
|---|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | 1250 | 300.12 | 132848640 | 0 | -12.05 |
| Qwen3-Embedding-0.6B-4bit-DWQ | 54 | 300.55 | 335218496 | 0 | -54.48 |

Both runs passed 20 concurrent async rounds with 40 canceled tasks and successful
recovery, then close/rejection/reload checks. Within each run repeated vectors
were bitwise identical. RSS samples fluctuated and finished lower; this is not a
claim that the driver returned all memory or a proof of multi-hour stability.
The final Qwen runner additionally confirmed its source files remained unchanged.

The Qwen latency matrix was deliberately bounded to batches 1/4 at 32 tokens
(five measured samples); its soak also exercised long and mixed-length inputs
and batch 32. The BGE matrix covers all twelve batch/length combinations.

## Raw observations

- [bge-before](../benchmarks/native-metal/performance-20260908/bge-before.json)
- [bge-after](../benchmarks/native-metal/performance-20260908/bge-after.json)
- [qwen-bounded](../benchmarks/native-metal/performance-20260908/qwen-bounded.json)
- [matmul-final](../benchmarks/native-metal/performance-20260908/matmul-final.json)
- [bge-padding](../benchmarks/native-metal/performance-20260908/bge-padding.json)

## Paired padding comparison

Both plans use the same current backend and tokenized inputs, with one warmup
and three measured samples per plan in alternating randomized order. Timing
excludes tokenization/admission and includes output restoration.

| Model/workload | Global positions | Bucket positions | Global median (s) | Bucket median (s) | Speedup | Max vector difference |
|---|---:|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5, 32 texts / max 512 tokens | 16384 | 636 | 6.331 | 0.629 | 10.07x | 8.20e-08 |
| Qwen3-Embedding-0.6B-4bit-DWQ, 8 texts / max 128 tokens | 1024 | 149 | 17.563 | 2.689 | 6.53x | 0.00e+00 |

The synthetic batches contain one long input and repeated short inputs. These
large gains apply to wasted-padding cases, not arbitrary homogeneous traffic.
[Qwen raw comparison](../benchmarks/native-metal/performance-20260908/qwen-padding.json).

## Final validation

The full local suite passed **301 tests** with **95.29% Python statement/branch
coverage**. Ruff, mypy (28 source files), dependency and frozen-input gates passed.
Native kernels are checked numerically; Python coverage does not cover C++/MSL
branches. The installed 0.4.0a1 wheel was tested from a separate environment with
Python isolated mode and network calls blocked: both models, tiled tensor math,
output order, cleanup and absence of unwanted framework imports passed. Wheel
and source archives have generated SPDX metadata.

[Validation record and tested wheel digest](../benchmarks/native-metal/performance-20260908/validation.json).
Hosted CI checks portable behavior; the Metal/model runs described here are local.
