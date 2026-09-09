# Full embedding comparison after MLP fusion

Measured offline on 2026-09-09 on the local Apple M1. The comparison snapshot
is runtime `1b553f0b87bf15729ecd04b413d19d3a434c00ee`, after bounded attention
tails, mixed projection tiles and [128/512-row MLP fusion](FUSED_MLP.md).
Later extension measurements below are separate from this snapshot.

## Method and limits

Both models use their existing verified local packs. MLX **0.32.2** runs in
its existing separate Python interpreter; Tessery's runtime has no MLX
dependency. The baseline is our independent model graph built from public
MLX operations, **not mlx-embeddings**. Both graphs compute in F32; Qwen's
BF16 quantization metadata is exactly promoted to F32 for the MLX reference.
The BGE graph uses F32 addmm, layer normalization, erf GELU and SDPA. There is
no `mx.compile` or lower-precision baseline in this comparison.

Each engine runs alone in a fresh process. The first pair is Tessery then
MLX, the second MLX then Tessery with case order reversed. Both use the same
Tessery tokenizer/admission/batching API wrapper. Token ID hashes, execution
plans, profile identity and output dimensions must match. MLX uses the
causal attention fast path for fully unpadded Qwen batches and an additive
mask otherwise; BGE uses no mask for unpadded batches and an additive mask
for padding.

The refreshed harness adds **identical-operation A/B controls**: ten samples
per label, randomized label order, after at least three warmups and 100 ms.
Reported medians pool both labels. Timing includes the complete synchronous
`encode()` call, including tokenization, execution planning and host output.
Loads, warmups and vector checks are excluded. CPU reference thread limits
are one; thermals, clocks and file caches remain uncontrolled.

All per-process outputs must be bit-identical across repeated calls.
Cross-engine and cross-process output comparisons retain `atol=5e-6,
rtol=1e-4`. BGE also checks the four frozen CPU reference batches in every
worker. `tools/compare_embedding_runs.py` rejects different source/model/
input identities and missing or duplicate cases, then uses the established
noise screen: all four A/B ratios within 0.90–1.10 and both engines' process
medians within 15%. This is a heuristic screen, not a confidence interval.
Numerical success is reported separately from timing evidence.

## Results

Ratios are **Tessery latency / MLX latency** across both engine orders.
Below 1 favors Tessery; above 1 favors MLX. Batch labels list logical token
lengths, not padded matrix sizes. These measured cases do not establish
performance on other models, Apple GPUs, precisions or compiled MLX graphs.

| Logical lengths | Qwen ratio | Qwen screen | BGE ratio | BGE screen |
| --- | ---: | --- | ---: | --- |
| 7 | 0.818–0.823 | pass | 1.190–1.967 | fail |
| 128 | 1.331–1.346 | pass | 1.498–1.637 | pass |
| 129 | 1.522–1.540 | pass | 1.495–1.562 | pass |
| 256 | 1.497–1.506 | pass | 1.548–1.576 | pass |
| 512 | 1.554–1.555 | pass | 1.547–1.566 | pass |
| 4 × 33 | 1.477–1.481 | pass | 1.673–1.713 | pass |
| 3, 7, 10 | 1.889–1.935 | pass | 1.557–1.721 | fail |

All seven Qwen cases pass the screen. Tessery is about **1.22x faster** on
the measured single seven-token input, while MLX is faster on every other
Qwen case. In particular, the mixed short batch loses despite the single
short-input win; a general short-input advantage is not established.

Five of seven BGE cases pass. MLX is faster in those five. The seven-token
case has a 1.53x Tessery process drift plus a failed A/B control; the mixed
short batch also fails a local control. Those rows remain in the evidence
but should not drive a dispatch change. At 128 tokens, BGE's 13.3% process
drift is close to the declared threshold, so its ratio range is relatively
wide even though it passes.

## 256-row extension experiment

The existing parallel fused kernel is tested at 256 rows against the current
unfused path, with two identical baseline labels, three warmup rounds and
15 paired randomized samples per route. Two fresh processes give speedups
of **1.0570x and 1.0563x** for complete public `encode()` calls. A/B control
ratios are 1.0008 and 1.0029. All 108 calls, including warmups, produce
bit-identical vectors. The candidate code itself is unchanged; this measures
whether extending its selection to another height helps the complete model.
The two prototype reports retain the pre-extension runtime source hashes.

## Reproduction and evidence

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_mlx_isolated.py \
  --model-dir "$QWEN_MODEL_DIR" --mlx-python "$MLX_PYTHON" \
  --mlx-mask causal --paired-controls --samples 10 \
  --lengths 7 128 129 256 512 --include-batches \
  --output artifacts/qwen-after-fusion-new.json
```

Repeat with `--engine-order mlx-first --reverse-cases` and a fresh output
path. BGE uses its local directory plus
`--profile-file model-manifests/bge-small-en-v1.5-hf-cache.json`. Summarize each
pair with `tools/compare_embedding_runs.py FIRST.json REPEAT.json --output
NEW-SUMMARY.json`. The older unpaired mode remains available; its reports
cannot be admitted by this repeat-screen tool.

- [Qwen first pair](../benchmarks/native-metal/post-fusion-20260909/qwen-first.json)
- [Qwen reversed pair](../benchmarks/native-metal/post-fusion-20260909/qwen-repeat.json)
- [Qwen repeat screen](../benchmarks/native-metal/post-fusion-20260909/qwen-summary.json)
- [BGE first pair](../benchmarks/native-metal/post-fusion-20260909/bge-first.json)
- [BGE reversed pair](../benchmarks/native-metal/post-fusion-20260909/bge-repeat.json)
- [BGE repeat screen](../benchmarks/native-metal/post-fusion-20260909/bge-summary.json)

- [256-row first prototype](../benchmarks/native-metal/post-fusion-20260909/fusion-256-first.json)
- [256-row repeat prototype](../benchmarks/native-metal/post-fusion-20260909/fusion-256-repeat.json)
