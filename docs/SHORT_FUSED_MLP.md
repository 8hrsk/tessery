# M16 fused MLP qualification with native plans

Baseline `3f18160`, isolated branch `perf/short-fused-mlp`, Apple M1, 2026-09-12.
Select the existing fused16 kernel for `(rows,cols,k)=(16,3072,1024)`; leave M8 on
its previous route. The candidate changes no shaders or accumulation order.

The benchmark holds two separately loaded models. Each captures its fixed route
before the first forward, and measurements require native plan hits. Baseline
A/B labels use the same baseline instance; the candidate has its own weights and
plan cache. No route is mutated after capture. Calls run sequentially. The tested
baseline revision already includes native execution plans, so these gains are
incremental to the reduction in Python dispatch overhead.

Two fresh-process runs used18 samples per label, balanced six permutations,
at least one second warmup per route and reversed case order. All vectors were
exactly equal within and across processes; source hashes, token IDs and plans
matched. All identical-control screens passed. Target M16 calls replace56
projection dispatches plus28 SiLU dispatches with28 fused dispatches.

| Logical lengths | Execution rows | Baseline/candidate, first | Repeat |
|---|---:|---:|---:|
| `[7]` | 8 | 1.0057 | 0.9982 |
| `[8]` | 8 | 0.9963 | 1.0011 |
| `[9]` | 16 | 1.0723 | 1.0317 |
| `[12]` | 16 | 1.0474 | 1.0488 |
| `[16]` | 16 | 1.0372 | 1.0516 |
| `[3]`, unchanged | 3 | 0.9886 | 0.9926 |
| `[24]`, unchanged | 24 | 1.0083 | 1.0105 |
| `[3,7]` | 3 + 8 | 0.9958 | 0.9973 |
| `[7,7]` | 16 | 1.0516 | 1.0375 |
| `[3,7,10]`, unchanged | 3 + 24 | 0.9964 | 1.0035 |

For each of the four M16 cases in each process, group paired log ratios into
three blocks of six permutations. Approximate Student-t95% ratio intervals all
have lower bounds above1.014 (minimum1.0141). These small-sample intervals assume
independent blocks and are diagnostics, not a cross-device guarantee. Together
with both process repeats and unaffected controls, the evidence supports the
narrow M16 guard. M8 gains are indistinguishable from noise and are not selected.

The existing M16 shader was previously qualified as the prefix of M24. Here the
full Qwen model adds independent coverage for all16-row execution layouts above,
with exact embeddings. Final integration still needs its normal native tests,
Shader Validation and installed-wheel qualification; this report is not a release.

Evidence: ignored `artifacts/short-fused-first.json` and
`artifacts/short-fused-repeat.json` in this worktree, including source/harness
hashes, all timings and diagnostics. The checked-in harness explicitly forces
the historical unfused M8/M16 baseline before capture so it remains usable after
integrating M16. That equivalent baseline wrapper was added after measurement;
the original records retain the hash of the measured harness.
