# Profiling and varied stress after the 0.6 qualification

This is an unreleased extension of the 0.6 candidate. It adds GPU profiling,
a measured small-tail optimization and reproducible validation tools. The
multi-hour qualification applied to the earlier `3a7915e` snapshot; it must not
be presented as multi-hour qualification of these changes. No model weights or
runtime dependencies were downloaded. Public imports and embedding contracts
remain unchanged.

## GPU profiling

`MetalRuntime.profile_kernels()` collects a timestamp pair for each dispatch.
Normal inference leaves it disabled and retains one compute encoder per command.
On this Apple M1, stage-boundary sampling is supported; dispatch-boundary
sampling is not. Profiling therefore uses one compute encoder per dispatch,
within the same command buffer. This changes timings: use it to locate expensive
operations, and use unprofiled API measurements to validate optimizations.

The implementation checks hardware support, calibrates GPU ticks with paired
CPU/GPU timestamps before and after completion, and rejects counter errors.
Metal's paired CPU timestamps are nanoseconds, as described in Apple's
[timestamp conversion documentation](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).
The timestamp buffer has 4096 entries, permitting 2048 dispatches per command.
Exceeding this bound aborts the command without publishing partial records.
Completed earlier commands remain available. Unsupported devices raise
`InferenceError` rather than returning made-up timings. Counter storage is
separate from the runtime's model/workspace buffer accounting.

```python
import numpy as np
from tessery import MetalRuntime

with MetalRuntime() as runtime:
    a = np.ones(4096, dtype=np.float32)
    with runtime.profile_kernels() as records:
        result = runtime.add(a, a)
    print(records)  # kernel, shape/dispatch parameters and GPU seconds
```

The context holds the runtime lock. Operations must run on that same thread.
Do not call an `EmbeddingModel` executor from inside its backend's profiling
context. The developer tool below calls the backend directly and handles this
restriction, while separately measuring the normal path:

```sh
python tools/profile_kernels.py --model-dir /path/to/qwen \
  --lengths 7 8 9 31 32 33 128 512 --samples 3 --output artifacts/qwen-profile.json
```

For BGE, add `--profile-file model-manifests/bge-small-en-v1.5-hf-cache.json`
and use the local BGE blob directory. Tools require a new output path and
checkpoint JSON atomically. Reports capture engine source hashes; profiling,
stress and retrieval reports also capture their harness hash.

The final Qwen profile measured about 1.73 s in tiled uint4 linear operations
and 1.56 s in attention at 512 tokens, summed over all layers. At nine tokens,
the small remainder still costs about 36 ms versus 20 ms for complete tiles.
These are intrusive stage timings, not normal end-to-end latencies. They point
to long-sequence attention and short/tail projection overhead as useful targets.
For BGE at 512 tokens, attention took about 379 ms versus 181 ms for both matrix
multiplication kernels combined. Its shorter unaligned inputs also expose the
existing F32 fallback. That fallback's accuracy guard remains in place.

## Small-tail dispatch

For aligned uint4 layouts, complete groups of eight rows still use the tiled
kernel. A final one-to-four-row remainder now uses the original four-row kernel
with a starting-row offset. Five-to-seven-row remainders retain the partial
tile kernel. This avoids the partial tile's shared-memory work for tiny tails.
No model is expanded to F32 and no correctness threshold was relaxed.

`tools/benchmark_tail_dispatch.py` compares the selected route against the
previous 0.6 dispatcher in randomized paired order, using the same model.
This is an incremental comparison against 0.6, not the older untiled baseline.
Controls with unchanged routes are included to expose timing noise.

The final run used three warmups and 30 paired samples per route on an Apple M1:

| Tokens, batch 1 | Previous p50, ms | Selected p50, ms | Previous / selected |
|---|---:|---:|---:|
| 9 | 102.47 | 89.46 | 1.145 |
| 10 | 104.62 | 87.93 | 1.190 |
| 11 | 102.36 | 96.29 | 1.063 |
| 12 | 108.70 | 109.38 | 0.994 |
| 17 | 127.30 | 113.85 | 1.118 |
| 33 | 199.19 | 191.16 | 1.042 |

The unchanged 2/7/8/31/32-token controls ranged from 0.973 to 1.034. The clearest
observed improvements are the one/two-row tails; the four-row tail has no measured
gain in this run. Smaller differences need more controlled measurement. These
are single-host observations with uncontrolled thermal/background load, not
guaranteed speedups. All compared vectors passed the existing tolerance; the
largest absolute difference was 1.94e-7.

## Varied stress and retrieval

`tools/stress_embeddings.py` uses a fixed random seed and batches of 1, 3, 8
and 32 texts. The corpus includes Russian, Chinese, combining accents, emoji,
special tokens, short inputs, tile boundaries and inputs reaching truncation.
It compares ragged batches and restored output order against individual calls.

Each round also gates tokenization to exercise a four-request admission limit:
two requests complete, two are canceled and two overload. Another cancellation
is triggered after entering backend forward, followed by a successful request.
This proves cancellation/recovery during forward activity, not preemption of
an already submitted GPU command. Each trim returns to model-resident bytes
with zero scratch cache. Closed models reject requests and own zero GPU bytes;
models reopen periodically. Reports distinguish reopen counts from closed
lifecycles. This is a bounded correctness stress, not a latency benchmark.

Both final runs passed eight rounds, covering all 15 corpus entries, with eight
forward cancellations, two reopens and three closed lifecycles per model. Every
round passed the two-completed/two-canceled/two-overloaded queue check. After
trim, live model bytes were exactly 335,218,496 for Qwen and 132,848,640 for BGE.
The maximum ragged-batch vector differences were 1.94e-7 and 1.01e-7 respectively.

```sh
python tools/stress_embeddings.py --model-dir /path/to/qwen \
  --rounds 8 --seed 20260909 --output artifacts/qwen-stress.json
python tools/evaluate_retrieval.py --model-dir /path/to/qwen \
  --output artifacts/qwen-retrieval.json
# Future runs can reject rank regressions against a compatible saved baseline:
python tools/evaluate_retrieval.py --model-dir /path/to/qwen \
  --baseline artifacts/qwen-retrieval.json --output artifacts/qwen-retrieval-check.json
```

The hand-authored fixture has eight question/document pairs: four English and
four Russian. Qwen uses all eight; BGE-small-en uses the four English pairs.
Documents and queries use the default empty prefixes. This small set checks
the real embedding/index/search flow. It is not a general retrieval-quality
benchmark or a claim about multilingual BGE. Baseline comparison verifies the
fixture, preprocessing and embedding contract before comparing individual ranks.
Both initial measurement and baseline-check reruns put the expected document
first for every question: Qwen 8/8 and English BGE 4/4 (Recall@1 and MRR both 1).

## Isolated MLX comparison

`tools/benchmark_mlx_isolated.py` starts each engine in a separate process and
uses an existing MLX interpreter only for the development baseline. The library
does not import MLX. The baseline is the same independent Qwen graph built
from public MLX operations as in the earlier comparison, with exact BF16 values
promoted to F32 and an explicit evaluation before host readback. It is not the
upstream mlx-embeddings application or a claim about the best possible MLX graph.

```sh
python tools/benchmark_mlx_isolated.py --model-dir /path/to/qwen \
  --mlx-python /path/to/existing/mlx-environment/bin/python --samples 30 \
  --output artifacts/mlx-isolated.json
```

Both workers use the same verified weights, tokenizer, API queue, batching and
384-dimensional output, three warmups and 30 timed calls per case. The five
cases include seven, eight and 31 tokens, four 33-token texts, and mixed text.
All vectors are compared at the existing `atol=5e-6, rtol=1e-4` threshold.
Process RSS and lifetime peak RSS can now be attributed to one engine each.
Peak RSS includes loading; current RSS is not a sum of allocated Metal buffers
and can change with macOS residency/compression. Allocator counters have
different definitions and cache policies. Imports are excluded from load time;
filesystem caches, background activity and thermals remain uncontrolled.

Final same-host results with MLX 0.32.2 (milliseconds, 30 samples per case):

| Token lengths | Tessery p50 / p95 | MLX p50 / p95 | Tessery / MLX p50 |
|---|---:|---:|---:|
| 7 | 76.46 / 97.57 | 56.09 / 64.15 | 1.36 |
| 8 | 33.65 / 42.00 | 65.74 / 78.36 | 0.51 |
| 31 | 163.78 / 192.04 | 87.09 / 100.17 | 1.88 |
| 33, 33, 33, 33 | 594.41 / 735.96 | 338.21 / 390.62 | 1.76 |
| 3, 7, 10 | 211.23 / 233.37 | 104.02 / 123.32 | 2.03 |

Tessery is faster in one case and slower in four. Maximum vector difference was
4.21e-7. Its process peak RSS was 838.8 MB versus 909.0 MB for the MLX worker,
using decimal MB. Per-case current RSS ranged 429.0–621.5 MB and 443.0–808.7 MB
respectively; Tessery was not lower in every case. These are process snapshots
under macOS memory management, not guaranteed memory requirements or a stable
memory advantage. The separate model/workspace counters remain the useful
invariant for leak and cache-bound checks.

Raw reports, including samples, vectors, source and harness hashes, are in
[profiling-20260909](../benchmarks/native-metal/profiling-20260909/):
[Qwen profile](../benchmarks/native-metal/profiling-20260909/qwen-profile.json),
[BGE profile](../benchmarks/native-metal/profiling-20260909/bge-profile.json),
[paired tail dispatch](../benchmarks/native-metal/profiling-20260909/tail-paired.json),
[isolated MLX](../benchmarks/native-metal/profiling-20260909/mlx-isolated.json),
[Qwen stress](../benchmarks/native-metal/profiling-20260909/qwen-stress.json),
[BGE stress](../benchmarks/native-metal/profiling-20260909/bge-stress.json),
[Qwen retrieval](../benchmarks/native-metal/profiling-20260909/qwen-retrieval-check.json),
and [BGE retrieval](../benchmarks/native-metal/profiling-20260909/bge-retrieval-check.json).

## Verification and next target

The full local suite passed **383 tests, 94.51% coverage**, including native
models, tail routing, profiling overflow/abort/recovery, and portable admission
recovery. Ruff, mypy, dependency policy and frozen-input checks passed.
The 39 targeted uint4/profiler tests also passed with Metal API and Shader
Validation enabled. An isolated installed macOS arm64 wheel passed inference
on both model packs and a timestamp-profiling check. The wheel/source layout
passed `tools/verify_release.py`. This work does not publish a PyPI release.

The next optimization should target long-sequence attention, with bounded
workspace and the same numerical thresholds. For short inputs, investigate
projection/fusion and command overhead. Loading peak RSS is a separate target;
lower steady-state RSS does not imply a lower loading peak. Expand the retrieval
fixture with hard negatives and a larger bilingual corpus before treating it as
a model quality gate. Repeat multi-hour qualification on the eventual release
snapshot after selecting the next kernel changes.
