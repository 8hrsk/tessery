# Rejected experiment: eight-row fused MLP with K64 decoding

Baseline `368d00a`; isolated branch `perf/gated8-k64`, Apple M1, 2026-09-12.
Keep the production `gated4_8x32` route. This experiment did not improve full
`encode()` latency and must not be enabled in production.

The prototype keeps 128 threads and decodes 16 values per thread using one BF16
metadata pair, halves threadgroup barriers and retains independent K32 partial
sums in their original order. Its two decoded-weight arrays grow from 8 to
16 KiB combined. No host route is changed: the development harness substitutes
only the M24 eight-row suffix, retaining the 16-row prefix and every other shape.

Ten Shader Validation tests passed: complete-tile bounds at offsets 0 and 16,
random signed inputs, saturation, N32/96/3072 and K64/256/1024, with exact equality
to the old shader. An initial compile rejected a variable named `half` (a Metal
type name); the tested version uses `word`.

Two fresh-process full-API runs used 18 samples per label, both baseline labels
and the candidate, balanced permutations, at least one second warmup per route,
and reversed case order in the second process. All vectors were exactly equal
and identical A/B controls passed their existing ±10% screening threshold.

| Logical lengths | Baseline/candidate, first | Repeat |
|---|---:|---:|
| `[24]` | 0.9763 | 0.9644 |
| `[12,12]` | 0.9800 | 0.9662 |
| `[8,8,8]` | 0.9691 | 0.9615 |
| `[3,7,10]` | 0.9679 | 0.9498 |
| `[25]`, unchanged control | 0.9810 | 1.0090 |

All targeted medians are worse in both runs. The untouched control demonstrates
that small measured differences include noise; these runs support rejecting the
optimization, not a portable claim of an exact regression percentage. Normal
first-run single24 GPU medians were about 34.0 ms baseline and 35.2 ms candidate.
Larger shared-memory use or altered decode access patterns could offset fewer
barriers; occupancy counters were not collected, so the mechanism is unproven.

Raw ignored outputs are `artifacts/gated8-k64-first.json` and
`artifacts/gated8-k64-repeat.json` in the experiment worktree. They include source
and harness hashes, token identity, execution plans, vectors, all timings and
normal runtime diagnostic samples.
