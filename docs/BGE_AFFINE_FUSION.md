# BGE affine projection fusion

This isolated prototype fuses the bias addition into the existing BGE F32
matrix multiplication for the three previously qualified `(N,K)` pairs:
`(384,384)`, `(1536,384)` and `(384,1536)`. Complete eight-row tiles preserve
independent K32 partial sums; residual rows use the same scalar SIMD reduction.
Both epilogues add bias after the completed F32 sum. Other shapes keep the
separate matmul and bias launches. GELU remains the existing erf formulation.

The tile epilogue repeats 32 bias values across an eight-row, 1 KiB shared
matrix and uses one barrier. This avoids assumptions about which SIMD lane
owns each matrix element. No model or workspace buffers are added. A full
BGE forward removes 72 `add_bias` launches per execution bucket.

## Evidence on Apple M1

The baseline is `368d00a`. `tools/benchmark_bge_fusion.py` switches between the
old route under two identical control labels and the fused route, then checks
**exact array equality for every embedding**. Each block contains all six label
permutations. Fresh second processes reverse case order; 95% intervals for
mean latency saved bootstrap entire permutation blocks, not correlated calls.
The initial pass used 30 samples per label and 0.3-second warmup; targeted
repeats used 60 samples per label and one-second warmup. All labels also warm
for at least three calls. Native execution plans are disabled explicitly when
available because a cached plan bypasses the experimental route selector.

A case passes the performance screen only if both identical A/B controls are
within 10%, each label's cross-process median drift is at most 15%, and both
block intervals exclude zero in the favorable direction. These are local
measurements under uncontrolled thermals, not universal speed guarantees.

| Logical lengths | Old/new p50 ratio | Evidence |
|---|---:|---|
| 3 | 1.115–1.258 | Still noisy; no performance claim |
| 7 | 1.130–1.210 | Initial paired pass |
| 17 | 1.053–1.118 | Positive intervals, but 15.2% drift fails screen |
| 24 | 1.099–1.113 | Targeted paired pass |
| 159 | 1.042–1.043 | Targeted paired pass |
| 160 | 1.043–1.057 | Targeted paired pass |
| 161 | 1.043–1.047 | Initial paired pass |
| 256 | 1.034–1.036 | Initial paired pass |
| 512 | 1.023–1.029 | Initial paired pass |
| 4 × 33 | 1.037–1.057 | Initial paired pass |
| 3, 7, 10 | 1.062–1.074 | Targeted paired pass |

The 17- and 24-token cases share the same 24-row execution kernel; the
remaining noisy API timings are not evidence of a shape-specific regression.
They nevertheless do not justify a performance claim for those inputs.
Native-plan interaction needs separate joint qualification before promotion.

Validation: 231 runtime unit tests; 44 affine kernel cases covering scalar,
full tile, mixed tail and fallback shapes against the old route and independent
F64 products; exact output sentinels; seven existing BGE model tests including
frozen CPU embeddings, persisted retrieval and HTTP. All 44 affine cases also
pass Metal Shader Validation. Ruff and strict mypy pass.

Compact evidence is in `benchmarks/native-metal/bge-affine-20260912/summary.json`.
Full samples, vectors, normal command counters and frozen initial harness are
in the ignored `artifacts/bge-fusion/` directory of the experiment worktree.

```sh
PYTHONPATH=src python tools/benchmark_bge_fusion.py \
  --model-dir "$BGE_MODEL_DIR" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --samples 60 --output artifacts/bge-fusion/new-first.json
# Run a fresh process with --reverse-cases and another output filename.
```
