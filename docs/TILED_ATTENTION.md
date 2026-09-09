# Tiled attention and independent review

This change extends the unreleased 0.6 candidate after `fd8a7b9`. It targets the
long-sequence attention bottleneck identified by the previous GPU profile.
The independent review ran alongside implementation, read source and existing
measurements, and wrote only its gitignored Markdown report. It did not run GPU
jobs or download models. The runtime remains independent of MLX and uses only
the existing local Qwen/BGE packs.

## Selected kernel

`attention_tiled` evaluates eight queries against a block of 32 keys using
F32 SIMD-group matrix operations. It performs a stable online softmax update
per key block, rescales the accumulated output and evaluates the probability × V
product with SIMD-group matrices. Four SIMD groups cooperate in one 128-thread
threadgroup. The declared shared arrays total 9312 bytes before compiler
alignment. There is no sequence-squared buffer or new global workspace.

The dispatcher selects it only for sequence widths divisible by 32 and at least
64, with head dimension 32 or 128. The original attention kernel remains the
fallback. Alignment makes full eight-query and 32-key memory reads safe; actual
lengths still mask padding, including queries beyond the actual input length.
Causal and bidirectional masks and grouped-query head mapping are preserved.
The kernel is intended for finite internal tensors from the verified model packs;
it does not promise that masked NaN/Inf values behave like skipped memory loads.

An initial four-way split-key experiment did not show a consistent performance
benefit and was discarded. It is not a selectable runtime path.

## Numerical and lifecycle checks

The added native tests cover both model head dimensions, causal and
bidirectional attention, GQA, batch two, sequence widths 64/128/512 and actual
lengths 1/31/32/33/511/512. Outputs for padded query rows are also checked.
Independent float64 references retain `atol=2e-6, rtol=2e-5`. A separate case
has logits near ±1131, where naive exponentiation would overflow, and very large
finite V values in masked positions. Tests with Metal API/Shader Validation
passed. Route tests cover 32/63/64/65/127/128/129/511/512 and unsupported head
dimensions 64/256. Full-model comparisons retain `atol=5e-6, rtol=1e-4`.

The full suite passed **414 tests, 94.48% coverage**. Dependency policy,
frozen-model input checks, Ruff and mypy passed. Compatibility IDs, tokenization,
pooling and normalization contracts are unchanged. Earlier multi-hour results
belong to their earlier source snapshot; they are not qualification of this kernel.
The final 20 native attention cases passed with Metal API/Shader Validation,
including uniform logits and a strong maximum shift between key blocks. Four
seeded stress rounds per model exercised batches 1/3/8/32, four cancellations
during forward, queue overload/recovery, one reopen and two closes. Trim returned
to resident model bytes, with zero scratch cache; close owned zero GPU buffers.
RAG baseline checks remained Qwen 8/8 and English BGE 4/4. An isolated installed
wheel passed both models and explicitly verified tiled dispatch on 128-token
inputs, as well as the extreme-input cosine fix.

## Measuring the change

`tools/benchmark_attention.py` measures the old and new kernels against F64.
`tools/benchmark_attention_model.py` compares the actual public embedding API
with the previous attention path and the selected dispatcher on identical input.
Both use paired randomized order, three warmups, atomic JSON reports and source
and harness hashes. The model tool includes a 33-token fallback control, aligned
64/128/512-token inputs and a ragged 128/127/33-token batch. These are unprofiled
latencies. Thermal and background load are uncontrolled on this single Apple M1.
Kernel speedups must not be presented as whole-model speedups.

Raw [measurement reports](../benchmarks/native-metal/attention-20260909/) include
[kernel timings](../benchmarks/native-metal/attention-20260909/kernels-final.json),
[Qwen API timings](../benchmarks/native-metal/attention-20260909/qwen-model.json),
[BGE API timings](../benchmarks/native-metal/attention-20260909/bge-model.json),
both MLX process orders, stress and retrieval checks, with raw samples and hashes.

Full-model p50 measurements (15 paired samples, milliseconds):

| Model | Token lengths | Previous | Selected | Speedup |
|---|---|---:|---:|---:|
| Qwen | 33, fallback control | 201.46 | 193.37 | 1.04× |
| Qwen | 64 | 272.39 | 253.94 | 1.07× |
| Qwen | 128 | 584.18 | 510.34 | 1.14× |
| Qwen | 512 | 3427.95 | 2183.69 | 1.57× |
| Qwen | 128,127,33 | 1351.90 | 1201.51 | 1.13× |
| BGE | 33, fallback control | 31.50 | 31.89 | 0.99× |
| BGE | 64 | 34.19 | 31.02 | 1.10× |
| BGE | 128 | 80.67 | 56.66 | 1.42× |
| BGE | 512 | 582.72 | 229.02 | 2.54× |
| BGE | 128,127,33 | 180.44 | 136.27 | 1.32× |

The maximum embedding differences were 2.09e-7 for Qwen and 1.04e-7 for BGE.
The unchanged 33-token controls measure noise, not a speedup. Direct attention
kernel measurements at 512 tokens improved 7.36× for Qwen's dimensions and 8.17×
for BGE's; other model work explains the smaller end-to-end gain.

```sh
python tools/benchmark_attention.py --samples 15 --output artifacts/attention-kernels.json
python tools/benchmark_attention_model.py --model-dir /path/to/qwen \
  --samples 15 --output artifacts/attention-qwen.json
# For BGE, supply its local directory and --profile-file when using the cache manifest.
```

The isolated MLX harness also accepts `--lengths`, `--mlx-mask` and
`--engine-order`. The optional causal string mask is used only when all rows
have full sequence length; ragged inputs still use the dense padding/causal mask.
This follows the [MLX SDPA API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).
Runs can reverse engine order to expose process-order bias. The baseline remains
an independent F32 graph built from public MLX operations, not an upstream
mlx-embeddings benchmark or a search over all MLX implementation choices.

Two fresh-process comparisons used MLX 0.32.2, the causal string mask, 15 samples
per engine/case and three warmups. The process order was reversed in the second
run. Every vector comparison passed; maximum absolute difference was 6.03e-7.

| Tokens | Tessery / MLX p50, Tessery first | Tessery / MLX p50, MLX first |
|---|---:|---:|
| 128 | 1.806 | 1.871 |
| 512 | 1.969 | 1.917 |

Tessery remains slower than this MLX graph in both long-sequence cases. The
before/after gains above are genuine improvements to Tessery, not evidence of
general superiority over MLX. Quantized projection work remains a major target
after reducing attention cost. The two runs expose order sensitivity but do not
remove all thermal/background bias. MLX dtype, graph compilation, forced-fused
choices and BGE comparisons still need their own controlled study.

## Review fix and remaining opportunities

The reviewer reproduced a separate public `cosine_search` issue: finite float32
vectors with components near `3e38` overflowed intermediate norms and produced
NaN scores, while tiny vectors could underflow. The normal embedding path is
preserved. Extreme inputs now use scaled float64 normalization; zero vectors
are still rejected. Regression tests cover `3e38` and `1e-40` values and ranking.

The most useful remaining experiments are alignment-aware padding for short
inputs, full F32 tiles plus a bounded tail for BGE, larger quantized projection
tiles, and fusion where measurements show dispatch overhead. Last-layer
specialization to compute only the pooled query is promising but needs its own
correctness and performance study. These are hypotheses, not measured speedups.

Follow-up: bounded alignment-aware padding is now implemented and measured in
[the execution padding report](ALIGNED_BATCHING.md). The other items remain
future work.

The original aligned kernel remains in use. A subsequent
[bounded-tail implementation](ATTENTION_TAILS.md) covers verified unaligned
sequence widths without changing the execution padding policy.
