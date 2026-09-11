# Fused MLP for 160 and 24 execution rows

The Qwen uint4 dispatcher now admits two additional `(M,N,K)` shapes:
`(160,3072,1024)` and `(24,3072,1024)`. The public API, tokenizer, batch planner,
weight representation, precision and workspace budget are unchanged.

At 160 rows the existing `gated4_16x32_k64` fuses gate projection, up projection
and SiLU. At 24 rows it processes the first complete 16 rows, followed by a new
`gated4_8x32` for the final eight. The latter uses four SIMD groups, 8 KiB of
threadgroup memory and independent F32 partial sums for successive K32 blocks.
It loads each left matrix tile once for both projections and writes the fused
result directly to the gate output. Prefix and suffix write disjoint ranges.
The existing up scratch remains allocated; this change does not claim a
reduction in retained workspace memory.

All other execution heights and projection dimensions retain their old paths.
The guard concerns **execution rows**, not logical token counts: existing
padding can map a 17-token input to 24 rows, 159 tokens to 160, and four
33-token inputs to four 40-token rows, totaling 160.

## Paired measurements

`tools/benchmark_batched_mlp.py` compares full public `encode()` calls with the
previous unfused route at the selected height. Both old-policy A/B labels use
the same implementation. Each label warms for at least three calls and one
second. Eighteen samples per label are balanced over all six ordering
permutations; the second fresh process reverses case order. Every measured
vector must equal the old result bitwise. Timings include normal runtime
counters, actual token IDs, batch plans and source/shader fingerprints.

The timing screen requires both A/B ratios within `[0.9,1.1]` and no more than
15% per-label median drift between processes. All cases below passed. This is
a noise heuristic, not a confidence interval; power/thermals are uncontrolled.
Results describe the local Apple M1 and pinned Qwen pack only.

| Logical input lengths | Old/new latency ratio, two runs |
|---|---:|
| 160 | 1.051–1.061 |
| 4 × 33 | 1.054–1.059 |
| 5 × 32 | 1.052–1.060 |
| 8 × 20 | 1.056–1.060 |
| 161, unchanged control | 0.997–1.006 |
| 24 | 1.043–1.053 |
| 2 × 12 | 1.049–1.066 |
| 3 × 8 | 1.046–1.059 |
| 3, 7, 10 | 1.032–1.033 |
| 25, unchanged control | 0.998–1.010 |

The four-by-33 case has **5.1–5.5% less latency**; the mixed short batch has
**3.1–3.2% less latency** than the preceding checkout. Saved vectors also
match between processes. An exploratory all-eight-row variant used one
launch but repeated weight decoding across three tiles; it gave weaker
results for the homogeneous 24-row cases and was not selected. It has only
one exploratory run, so no definitive cross-variant ranking is claimed.

Raw evidence and paired summaries are in
`benchmarks/native-metal/batched-mlp-20260912/`. The 160-row evidence used the
previous production shader; the 24-row evidence appended the new kernel.
Their host override was then replaced by the guarded production dispatcher.
Installed-wheel checks separately verify production dispatch and old-policy
output equality, including padded inputs and dimension truncation.

Reproduce against the current shader with a fresh output path:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_batched_mlp.py \
  --model-dir "$QWEN_MODEL_DIR" --rows 24 --samples 18 \
  --output artifacts/batched-mlp-first-new.json
```

Repeat with `--reverse-cases`; use `--rows 160` for the other guard. Candidate
shader injection is only for entry points absent from the production shader.

## Qualification scope

Kernel tests compare independent F64 products, exact old-route outputs and
untouched output sentinels. Metal Shader Validation additionally checks the
minimal eight-row input and an offset suffix. Real-model checks cover both
Qwen and BGE, cancellation, queue overload, varied batches, trim and reload.

The long-run harness now records actual batch plans, per-case completed calls
and kernel dispatch deltas. Its twelve-case cycle includes 160/24 execution
rows, the three-row specialization, long inputs and neighboring widths. A
short rehearsal validates this reporting; only completed long reports can
establish the requested two hours per Metal model and four hours of portable
CPU stress on Kaggle. Kaggle does not qualify Metal kernels. The earlier
September 9 run belongs to an older commit and cannot qualify this revision.


The initial complete local test run after updating the dispatch-count fixture
passed **991 tests**, with **94.62% coverage**; the focused Shader Validation run
passed **34 tests**. Installed-package tests compare 17 logical layouts at
64/384/1024 dimensions (51 cases) with the old policy, and the saved 384-D
baseline vectors with the installed implementation. Both Qwen and BGE passed
eight seeded stress rounds, eight in-forward cancellations, two reloads and
three closed lifecycles. Thirty-second rehearsals completed 156 Qwen and 1,200
BGE calls, visiting every workload case and preserving bitwise repeatability,
live buffer ownership and workspace bounds. These rehearsals are not the
multi-hour qualification.
