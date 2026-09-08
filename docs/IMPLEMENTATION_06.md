# Tessery 0.6 candidate: partial tiles, cancellation and qualification

0.6.0a1 is a locally built release candidate, not a published PyPI release.
The public imports remain `from tessery import EmbeddingModel, list_profiles`.
No runtime dependencies were added. Compatibility IDs, token IDs, pooling and
numerical tolerances remain unchanged.

## Arbitrary sequence lengths

The uint4 dispatcher now runs complete 8-row tiles through the existing tiled
kernel and the final partial tile through bounded shared-memory loads/stores.
The partial kernel reserves 6 KiB of threadgroup memory, including decoded weights;
it never expands the entire model to F32. Outputs outside the actual row count
are never stored. Channels still require alignment to 32 and K to 64; unsupported
channel layouts retain the original kernel. Rows 1–4 also retain the original
kernel because the first candidate regressed a two-token request.

On this M1, randomized paired full-Qwen measurements (two warmups, five samples,
identical weights/texts, uncontrolled thermal and background load) gave:

| Batch / tokens | Original uint4 p50, s | Selected dispatcher p50, s | Ratio |
|---|---:|---:|---:|
| 1 / 7 | 0.145495 | 0.075355 | 1.93x |
| 1 / 9 | 0.204919 | 0.096908 | 2.11x |
| 1 / 31 | 0.572557 | 0.152981 | 3.74x |
| 4 / 33 | 2.440783 | 0.596126 | 4.09x |

This compares against the original `linear4` kernel. These unaligned shapes used
that kernel in 0.5. Aligned shapes were already accelerated in 0.5, so their
original-kernel ratios must not be presented as additional 0.6 gains. The
2-token control dispatches the same original kernel on both sides; its timing
variation is measurement noise, not an optimization.

Evidence: [paired measurements](../benchmarks/native-metal/delivery-20260909/quantized-final-paired.json).
The harness also contains isolated direct-kernel diagnostics; full-model results
exercise the actual split dispatcher. Synthetic results pass the existing F64
reference threshold `atol=5e-5, rtol=5e-5`. Model comparisons retain
`atol=5e-6, rtol=1e-4`; maximum difference over the model cases was 2.84e-7.
28 uint4 cases passed with Metal API and Shader Validation enabled, covering
rows 1, 2, 7, 8, 9, 15, 16, 17, 31 and K 64, 1024, 3072 plus an unaligned-channel fallback.

## Cooperative cancellation

The API passes each request's cancellation event into BPE and WordPiece.
Checkpoints cover BPE pair initialization/heap operations, WordPiece cleaning,
segmentation and greedy matching, plus token and text boundaries. Periodic loops
check every 256 operations. No mutable cancellation state is stored on a tokenizer.
An aborted request releases admission in the existing `finally` block and the next
request can run. Tests coordinate cancellation during tokenization and verify
that no GPU forward is called for the canceled request.

This is cooperative cancellation, not a hard real-time deadline: an individual
Unicode/regex/array operation may finish before the next checkpoint. A submitted
Metal command still completes before its buffers are released; its result is discarded.

## Direct MLX comparison

[The comparison harness](../tools/benchmark_mlx.py) builds a separate Qwen graph
from public [MLX operations](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html).
It does not load or copy mlx-embeddings. The installed MLX version was 0.32.2.
Both implementations receive the same verified Qwen weights, tokenizer, queue,
batching, causal mask, last-token pooling and 384-dimensional normalization.
The MLX graph promotes BF16 weight values to F32 to match Tessery's compute
contract. Upload/loading is outside the timed region; each call includes API,
tokenization, graph/command construction, completed GPU execution and CPU output.
Each MLX call creates and evaluates a fresh result, avoiding lazy-result reuse.

| Token lengths | Tessery p50, s | MLX reference p50, s | Tessery / MLX latency |
|---|---:|---:|---:|
| 7 | 0.060332 | 0.043103 | 1.40 |
| 8 | 0.042322 | 0.069154 | 0.61 |
| 31 | 0.216391 | 0.082772 | 2.61 |
| 33, 33, 33, 33 | 0.604018 | 0.337532 | 1.79 |
| 3, 7, 10, mixed text | 0.217353 | 0.101118 | 2.15 |

Evidence: [raw MLX comparison](../benchmarks/native-metal/delivery-20260909/mlx-paired.json).
Two warmups and five randomized paired samples per case; both models are resident
in one process. Every result passed the unchanged vector tolerance, with maximum
absolute difference 4.21e-7. These are single-host diagnostics, not a universal
speed claim or a benchmark of the upstream mlx-embeddings application.

Allocator snapshots are recorded, but are not directly comparable process-memory
measurements: the runtimes have different cache policies, the MLX graph retains
F32 scales/biases, and both models coexist. MLX active/cache bytes were about
373/807 MB; Tessery active bytes about 348 MB, including 12 MB scratch cache.
Process-wide peak memory is not attributable to one engine in this harness.
Further work should target short/tail dispatch overhead and long-sequence
attention, then repeat isolated-process memory measurements and controlled timings.

## Reliability and release boundaries

The final suite passed: **371 tests, 94.69% coverage**, including portable soak
and release-layout checks. Ruff, mypy, dependency policy and frozen-input checks
also passed. An isolated installed wheel passed inference on both model packs.
Short native diagnostics validate repeated vectors, bounded workspace, concurrent
cancellation, recovery, close-to-zero buffers and model reload. The Qwen soak
completed 18 calls over 31.57 seconds, with 64 KiB RSS variation; BGE completed
123 calls over 30.16 seconds. Both closed with zero active/cache GPU bytes. A
35-second portable rehearsal completed 812 iterations with one atomic checkpoint
and about 0.95 MiB retained traced allocations. These bounded
local checks are not multi-hour qualification.

The [Kaggle notebook](../notebooks/kaggle-portable-soak.ipynb) runs the portable
suite and a separate four-hour CPU workload without model downloads. See
[Kaggle instructions and scope](KAGGLE_QUALIFICATION.md). A native Metal soak still
requires Apple Silicon; Kaggle results cannot certify the GPU runtime.

The [publishing workflow](../.github/workflows/publish.yml) is manual and tag-bound.
It checks the native suite, builds a macOS arm64 wheel and source archive, rejects
stale/portable payloads, tests the installed wheel with both real models, then
passes those exact artifacts to a separate OIDC publishing job. PyPI publisher
registration, environment protection and a provisioned native runner are external
prerequisites; the workflow has not published 0.6.0a1.
