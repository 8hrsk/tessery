# Post-attention residual and RMSNorm experiment

Decision: **keep this prototype outside production**. It removes 28 Qwen
kernel launches and preserves both residual state and normalized output, but
the complete API gain is small and inconsistent before native-plan integration.
Only the 512-token case has positive mean-latency intervals in both processes,
with a p50 old/new ratio near 1.0017. That does not justify a broad default route.

The experiment is isolated on `perf/qwen-residual-norm`, based on `368d00a`.
For hidden width 1024, each SIMD lane computes and writes its rounded F32
residual sum, accumulates squares in the existing order, then applies the
existing RMSNorm weight scaling. The later MLP residual skip still consumes
the updated state. Only post-attention residual plus normalization is fused;
next-layer input normalization and the final normalization remain separate.
Other hidden widths retain the original two kernels.

Two fresh complete-API runs each use 24 samples per label, all six permutations
of two identical old controls and the candidate, at least three warmups and
one second per label. The second process reverses workload order. Every
measured vector equals the old route exactly, including between processes.
The 95% interval resamples four complete permutation blocks per run; that is
limited statistical power for such small effects. Plans are disabled to avoid
cached replay bypassing the experimental route selector. Thermals are uncontrolled.

| Logical lengths | Old/new p50 latency ratio | Positive mean-saved CI in both runs |
|---|---:|---|
| 3 | 1.013–1.020 | No |
| 7 | 1.000–1.007 | No |
| 17 | 0.988–0.997 | No |
| 24 | 0.996–1.002 | No |
| 160 | 1.002–1.003 | No |
| 512 | 1.0017–1.0018 | Yes |
| 4 × 33 | 1.002–1.005 | No |
| 3, 7, 10 | 1.007–1.014 | No |

Kernel validation covers random, cancellation and zero data at one through
4096 execution rows, plus a nonqualified hidden-width fallback. Both output
buffers must equal the old path and preserve sentinel suffixes; normalized
results also match an independent F64 reference within existing tolerances.
All 18 kernel cases and 26 existing Qwen model tests pass. The 18 kernel cases
also pass Metal Shader Validation; 205 runtime unit tests, Ruff and mypy pass.

Compact evidence is in
`benchmarks/native-metal/residual-norm-20260912/summary.json`; full samples,
vectors and normal runtime counters remain in ignored `artifacts/residual-norm/`.

A stronger future fusion hypothesis is Q/K RMSNorm plus RoPE. At head dimension
128, each lane of a 32-lane group owns values at `lane`, `lane+32`, `lane+64`
and `lane+96`; both RoPE pairs stay within that lane after the RMS reduction.
This could remove 56 launches and intermediate tensor writes per model forward.
It needs separate exact numerical and position checks, identical transcendental
operations, and full-API measurement with the native execution plan enabled.
No such prototype or performance result is claimed here.
