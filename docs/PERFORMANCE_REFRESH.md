# Performance refresh after Qwen and BGE projection changes

This report measures the runtime at `bcbf7812e46193e7e301925e2f10977ee60fc56d`
on the local Apple M1, macOS 26.3, on 2026-09-09. The work changes developer
measurement tools and documentation, not production inference kernels or
public APIs. No model or dependency downloads were needed.

## What the instruments now measure

`tools/profile_kernels.py` now applies the current execution padding policy
by default and records both logical token count and execution width. It checks
the resulting direct-backend vector against `EmbeddingModel.encode` before
profiling. Previously it always sent raw logical widths directly to the
backend, which ceased to represent API dispatch after execution alignment was
introduced. `--execution-policy raw` preserves that diagnostic mode explicitly.
The refreshed selected-policy profiles matched public API vectors exactly.

The profiler additionally groups timings by `(kernel, rows, cols, K)`, making
individual projection shapes visible. Five samples follow two backend warmups
and one public-API call. Normal backend wall times are recorded separately.
These controls exclude tokenization and admission; the isolated comparison
below includes them. The timestamp profiler creates an encoder per dispatch
and perturbs execution. Kernel shares below are proportions of summed kernel
medians, useful for prioritization, not an end-to-end speedup forecast.

`tools/benchmark_mlx_isolated.py` now supports the local BGE pack as well as
Qwen, and records profile identity, token-ID hashes and execution plans. It
rejects different preprocessing/plans or incompatible vector outputs. Every
comparison uses separate fresh worker processes so the two engines are not
resident together. Reversing engine order tests sensitivity to run order; the
samples are not interleaved request-by-request across engines.

The MLX baseline is an independently written graph using public MLX 0.32.2
operations, not upstream mlx-embeddings and not a claim about every optimized
MLX configuration. Runtime code never imports it. Both engines use the same
verified weights, tokenizer implementation, API admission, batch planner and
384-dimensional outputs. Qwen retains uint4 weights and promotes BF16 metadata
to F32; BGE uses its original F32 weights. No compiled whole-model graph or
lower-precision baseline is tested.

The new BGE graph uses bias-aware matrix multiplication, official
[layer normalization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.layer_norm.html),
[scaled dot-product attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html),
and erf-form GELU with [mlx.core.erf](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html).
It uses bidirectional attention, absolute positions, token type zero and L2
normalization of the encoder CLS state, excluding the optional tanh pooler.
Both BGE workers validate all four frozen CPU-reference batches before timing.

Qwen requests the MLX causal fast path when all logical lengths equal the
execution width; otherwise it constructs an additive causal/padding mask.
BGE uses no attention mask for full lengths and an additive key-padding mask
otherwise. Both honor actual sequence lengths. MLX chooses its own SDPA
implementation (`force_fused` remains its default). Dense masks can affect
performance, particularly around padded inputs; this comparison deliberately
holds Tessery's current batching policy constant across both engines.

Both interpreters are Python 3.12.13 with NumPy 2.5.2. Existing regex versions
differ (Tessery 2025.9.18, MLX environment 2026.9.3); exact token IDs are checked,
but tokenizer timing is not isolated, so this is a confound for short API calls.
No environment packages were changed. Thermals, background work, filesystem
caches and macOS memory residency remain uncontrolled.

## Updated kernel profiles

| Model | Logical / execution tokens | Projection share | Attention share |
| --- | ---: | ---: | ---: |
| QWEN | 128 / 128 | 89.0% | 5.4% |
| QWEN | 129 / 129 | 79.7% | 15.8% |
| QWEN | 512 / 512 | 80.7% | 14.6% |
| BGE | 128 / 128 | 72.0% | 15.3% |
| BGE | 129 / 136 | 41.3% | 53.8% |
| BGE | 512 / 512 | 57.3% | 35.2% |

At 512 tokens, Qwen's two MLP expansion projections account for about 545 ms
of profiled stage time and the MLP output for about 277 ms. BGE's expansion
accounts for about 29 ms and its MLP output for about 18 ms. These numbers are
medians grouped by projection dimensions across layers; summing medians from
different groups is not identical to taking the median of their per-sample sum.

The boundary at 129 logical tokens is significant. Qwen stays at width 129,
using the old eight-row projection tiles plus scalar tails and scalar
attention. BGE pads to width 136, which still fails the attention tile guard
(sequence length must be divisible by 32). Its profiled attention stage rises
from about 3.56 ms at 128 tokens to 23.28 ms at 129 logical tokens. Corresponding
unprofiled backend medians were 28.49 ms and 47.08 ms. This is a dispatch-boundary
observation, not a controlled before/after optimization experiment.

At 33 logical tokens, Qwen executes width 40 and falls back entirely to the
older eight-row projection kernel because 40 is not divisible by 16. That is
a separate opportunity from attention's sequence alignment restriction.

Raw profiles: [Qwen](../benchmarks/native-metal/refresh-20260909/qwen-profile.json)
and [BGE](../benchmarks/native-metal/refresh-20260909/bge-profile.json).

## Isolated Qwen comparison

Each cell reports Tessery / MLX p50 latency in milliseconds. The two columns
are separate complete runs in opposite engine orders, with 15 timed calls
per case per engine after three warmups. Ratios above one mean Tessery is
slower. All 14 matched cases passed the existing vector gate; the maximum
absolute difference was 6.04e-7 (rounded up).

| Logical lengths | Tessery first: T / M, ms | MLX first: T / M, ms | T / M ratio range |
| --- | ---: | ---: | ---: |
| 7 | 33.30 / 62.66 | 34.87 / 65.65 | 0.53–0.53 |
| 31 | 98.08 / 76.80 | 110.16 / 82.52 | 1.28–1.33 |
| 128 | 400.52 / 281.42 | 402.50 / 282.68 | 1.42–1.42 |
| 129 | 642.25 / 341.79 | 643.81 / 333.80 | 1.88–1.93 |
| 512 | 1757.77 / 1092.26 | 1781.38 / 1080.47 | 1.61–1.65 |
| 4 × 33 | 510.28 / 338.92 | 506.59 / 336.10 | 1.51–1.51 |
| 3, 7, 10 | 203.88 / 100.31 | 214.78 / 98.54 | 2.03–2.18 |

The winner is unchanged by engine order for all seven cases. Tessery is faster
on seven-token input, and MLX is faster on the other six. At 128 tokens the
ratio is about 1.42; at 512 it is 1.61–1.65. The small ratio range at those
lengths supports the direction of this result, not a universal performance
guarantee or a statistical confidence interval.

At 129 tokens, MLX uses a causal mask because Qwen does not add execution
padding there. At seven and 31 tokens, padding means the reference takes the
additive-mask branch. Thus this is a comparison of explicitly documented
matched workloads, not MLX's best separately tuned batching strategy.

Reports: [Tessery first](../benchmarks/native-metal/refresh-20260909/qwen-tessery-first.json),
[MLX first](../benchmarks/native-metal/refresh-20260909/qwen-mlx-first.json).

## Isolated BGE comparison

The same two-order, 15-sample procedure applies. Both workers passed the four
frozen CPU-reference batches: maximum absolute error was 2.35e-7 for Tessery
and 3.71e-7 for the MLX graph (rounded up). Across the timed BGE cases, the
largest cross-engine difference was 3.69e-7. The numerical gates remain
`atol=5e-6, rtol=1e-4`.

| Logical lengths | Tessery first: T / M, ms | MLX first: T / M, ms | T / M ratio range |
| --- | ---: | ---: | ---: |
| 7 | 5.99 / 5.94 | 6.01 / 5.93 | 1.01–1.01 |
| 31 | 9.63 / 6.07 | 9.94 / 7.41 | 1.34–1.59 |
| 128 | 27.37 / 17.97 | 26.91 / 18.45 | 1.46–1.52 |
| 129 | 51.89 / 23.75 | 50.63 / 23.71 | 2.13–2.19 |
| 512 | 125.16 / 80.59 | 119.54 / 78.54 | 1.52–1.55 |
| 4 × 33 | 45.32 / 14.69 | 33.52 / 24.63 | 1.36–3.08 |
| 3, 7, 10 | 16.76 / 13.17 | 17.22 / 13.05 | 1.27–1.32 |

At seven tokens the difference is about 1%, insufficient to claim a meaningful
winner. MLX is faster in the other six measured cases in both orders. At 128
and 512 tokens the ratio is approximately 1.46–1.52 and 1.52–1.55 respectively.
The four-by-33 batch varies markedly: 1.36–3.09. The direction is consistent,
but **a stable threefold gap is not established**. Profile and retime that
specific batch before selecting an optimization on the strength of its first
run. The batch-one profiles do not directly attribute that batch's bottleneck.

Reports: [Tessery first](../benchmarks/native-metal/refresh-20260909/bge-tessery-first.json),
[MLX first](../benchmarks/native-metal/refresh-20260909/bge-mlx-first.json).

## Priorities supported by these measurements

Follow-up status: priorities 1 and 2 are now implemented and validated in
[attention tails](ATTENTION_TAILS.md) and [mixed projections](MIXED_QUANTIZED_TILES.md).
The ranking below records the evidence at the time of this refresh.

1. **Handle incomplete attention tiles.** Qwen width 129 and BGE width 136
   fall back to scalar attention. Support bounded query/key tails in the
   existing tiled algorithm, with masked loads and stores, while preserving
   causal/GQA/BERT semantics, numerical gates and bounded workspace. Validate
   63/64/65, 127/128/129 and 511/512, plus unequal-length batches. Compare full
   API latency on identical inputs; do not assume extra padding is free.
2. **Use mixed 16-row and eight-row Qwen projection tiles.** Keep the new
   kernel for complete 16-row regions of widths such as 40 and 129, then
   compute only the remainder using qualified smaller kernels. Preserve
   reduction order and verify tail offsets with shader validation. This
   targets an observed fallback without redesigning the entire matmul.
3. **Focus further long-input work on MLP projections.** Projections still
   dominate both models, especially Qwen. Compare each projection shape
   directly against MLX before selecting another tile/layout experiment.
   Gate/up weight loading and reuse, and BGE's expansion/down-projection
   throughput, are concrete targets. Earlier rejected tile sizes should not
   be assumed faster without new same-input measurements.
4. **Treat bias/activation/residual fusion as a secondary experiment.** BGE
   has 72 separate bias dispatches per forward, but at 512 tokens their
   measured kernel share is only about 5%; GELU is about 1%. Fusion may reduce
   command encoding and memory traffic, particularly for short inputs, but
   these timings do not justify promising a large full-model gain from fusion
   alone. Intrusive per-dispatch profiling overstates some small-stage costs.

This is a prioritization of future experiments. No production optimization was
implemented in this measurement refresh, and the results do not establish
performance superiority over MLX as a whole. All timing cases use synthetic
repeated-token texts plus each model's special tokens; this is not a retrieval
quality evaluation. The existing frozen BGE corpus serves a correctness role.

## Memory and validation limits

Each report includes separate-process current and lifetime peak RSS, model load
time, and engine allocator counters. Lifetime peaks include loading, warmup,
and (for BGE) frozen-reference validation. MacOS compression/residency can make
current RSS lower than logical model storage. Different allocator/cache policies
and the expanded Qwen BF16 metadata preclude treating RSS snapshots as an
intrinsic memory advantage. These runs are not a memory leak qualification or
a replacement for a final multi-hour release soak.

The six final reports include all raw samples and both comparison workers;
intermediate worker files are not duplicated in git. Their source hashes
match the unchanged production runtime and their harness/reference hashes
match the final developer tools. The five-sample BGE preflight is excluded
from final performance tables.

The full local suite passed **614 tests with 94.55% coverage**. Ruff formatting,
Ruff lint, mypy, installed dependency policy and frozen input checks passed.
Both selected-policy profiling runs passed; a separate raw-policy smoke at
nine tokens passed for each model. No correctness threshold was relaxed.
Runtime/package code is unchanged, so this refresh does not rebuild or publish
a wheel or reset the prior release-qualification status.

## Reproduction

```sh
.venv/bin/python tools/profile_kernels.py --model-dir "$QWEN_MODEL_DIR" \
  --lengths 7 9 31 33 128 129 512 --samples 5 --output artifacts/qwen-profile-new.json
.venv/bin/python tools/profile_kernels.py --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --lengths 7 9 31 33 128 129 512 --samples 5 --output artifacts/bge-profile-new.json
.venv/bin/python tools/benchmark_mlx_isolated.py --model-dir "$QWEN_MODEL_DIR" \
  --mlx-python "$MLX_PYTHON" --samples 15 --lengths 7 31 128 129 512 \
  --include-batches --mlx-mask causal --engine-order tessery-first \
  --output artifacts/qwen-compare-new.json
.venv/bin/python tools/benchmark_mlx_isolated.py --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --mlx-python "$MLX_PYTHON" --samples 15 --lengths 7 31 128 129 512 \
  --include-batches --engine-order tessery-first --output artifacts/bge-compare-new.json
```

Set the model variables to existing verified local packs and `MLX_PYTHON` to
an existing development interpreter. Repeat both comparisons with
`--engine-order mlx-first` and different output paths. Case-order seed is 94;
reports require new paths and are checkpointed atomically. The library itself
continues to depend only on its pinned NumPy and regex packages.

The first follow-up, bounded attention query/key tails, is implemented and
measured in the [attention-tail report](ATTENTION_TAILS.md).
