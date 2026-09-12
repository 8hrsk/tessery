# Deferred experiment: BGE attention shared-memory specialization

Baseline `368d00a`; branch `perf/bge-attention32`, Apple M1, 2026-09-12.
The broad dim32 specialization is not selected for production in this cycle.

`attention_tiled` reserves `state[8*128]` and `partial[8*128]` even for BGE's
32-dimensional heads. The isolated candidate changes only those declarations to
`8*32`, keeping all indexing and arithmetic unchanged. Combined declared shared
arrays shrink from 9312 to 3168 bytes. Existing `attention_tail_32` already uses
correctly sized arrays and was left unchanged. The benchmark alone selects the
candidate at dim32; production dispatch is untouched.

Six Shader Validation tests passed with exactly equal outputs: seq64/128/512,
causal and bidirectional masks, ragged lengths1/seq-1, grouped KV heads. The
full-model measurements used two fresh processes, reversed case order,18 samples
per label, identical baseline A/B controls and balanced six permutations after
at least one second warmup for every route. All embeddings were exactly equal.

| Logical lengths | Baseline/candidate, first | Repeat |
|---|---:|---:|
| `[24]`, unchanged | 0.9964 | 1.0121 |
| `[64]` | 1.0321, failed control | 0.9569 |
| `[128]` | 1.0146 | 1.0021 |
| `[160]` | 1.0096 | 0.9973 |
| `[512]` | 1.0287 | 1.0185 |
| `4×33`, unchanged | 0.9947 | 0.9988 |
| `[159,160]` | 1.0329 | 1.0071 |

The broad route lacks consistent benefit.512 remains a possible future target,
but its small apparent gain needs stronger evidence. As a diagnostic, grouping
paired log ratios into the three six-permutation blocks of each run gives
approximate Student-t95% intervals of `[1.0125,1.0351]` and `[0.9925,1.0462]`
for the geometric baseline/candidate ratio. These very small-sample intervals
are not a universal performance guarantee; the second includes no improvement.
Do not promote the specialization based on these medians alone.

Raw ignored evidence is in `artifacts/attention32-first.json` and
`artifacts/attention32-repeat.json` in the isolated worktree, with hashes, IDs,
plans, vectors, raw timings and normal runtime counters. No occupancy counters
were collected, so lower declared shared memory is not proof of higher achieved
occupancy. The production kernel remains unchanged.
