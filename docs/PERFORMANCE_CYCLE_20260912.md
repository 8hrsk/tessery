# Performance cycle: 2026-09-12

This cycle executes the first bounded experiments in stages 0–5 of
[the performance plan](PERFORMANCE_PLAN.md). Baseline: `368d00a`; measured
candidate: `6a59b21`. Subsequent report-only commits do not change that source.
Experiments used separate branches/worktrees and the existing local Qwen3 0.6B
DWQ and BGE-small weights. No model download was needed.

## Accepted changes

| Stage | Result | Details |
|---|---|---|
| 0 | Fresh profiles, compiled MLX reference, immutable revision comparison | Dynamic masks/lengths are graph inputs; compilation is outside warm timing; retraces are rejected. API and pretokenized backend scopes are separate. |
| 1 | Native prepared execution plans | One C ABI call re-encodes the captured operations with current buffers; bounded four-plan cache shares the 64 MiB retention budget with scratch. Arithmetic is unchanged. |
| 2 | BGE matmul+bias | Preserve F32 reduction order; guard rows >= 8 and the three validated projection shapes. Removes 72 bias dispatches per forward. [Evidence](BGE_AFFINE_FUSION.md). |
| 3 | Qwen M16 fused MLP | Reuse the existing kernel only at the verified M16 shape; M8 remains unchanged. [Evidence](SHORT_FUSED_MLP.md). |
| 4 | Attention shared-memory experiment evaluated | No production change: the repeated full-model result was not convincing. [Evidence](ATTENTION32_EXPERIMENT.md). |
| 5 | Index token reuse and cached-norm partial top-k | Preserve public encode overrides, cancellation, stable ties, score ordering and SQLite round trips. [Token reuse](INDEX_TOKEN_REUSE.md), [search](RETRIEVAL_PERFORMANCE.md). |

Execution plans retain operation metadata and pipelines, not activation buffers.
Replay checks current buffer sizes and ownership. Trim, close and command errors
clear retained plans. `MemoryStats.plan_cache_bytes` reports native plan metadata
separately; it is not a measurement of all Python/driver overhead.

Review also caught two measurement/API hazards: helper-mutating benchmarks now
bypass cached plans, and index token reuse respects subclass, instance and class
overrides of public `encode()`. Revision benchmarks verify the actual imported
source paths and hash sources/helpers before and after each subprocess.

## Full API comparison with the original revision

Apple M1; 30 samples per identical A/B label; four fresh processes in
baseline/candidate/candidate/baseline order, with reversed case order. Source,
dependency, profile, token IDs and batching identities were checked. Output vectors
match **exactly** for both models in every measured case.

The table reports baseline latency divided by candidate latency: above 1 is
faster. Ranges are the two observations, **not confidence intervals**. The screen
requires identical controls within 10% and process drift within 15%; it detects
large noise but cannot establish small improvements. Thermals were uncontrolled.

| Token lengths | Qwen speed ratio | BGE speed ratio |
|---|---:|---:|
| 3 | 1.167–1.220 | 1.595–1.598 |
| 7 | 1.159–1.170 | Excluded: 33.6% candidate process drift |
| 16 | 1.211–1.245 | 1.325–1.326 |
| 24 | 1.131–1.139 | 1.213–1.363 |
| 160 | 1.029–1.031 | 1.098–1.099 |
| 512 | 1.008–1.008 | 1.051–1.052 |
| 4 × 33 | 1.027–1.030 | 1.106–1.118 |
| 3, 7, 10 | 1.180–1.183 | 1.228–1.234 |

For example, the first Qwen mixed-batch median changed from 58.90 to 49.91 ms;
BGE at three tokens changed from 7.45 to 4.67 ms. Qwen at 512 tokens changed
from 536.02 to 531.88 ms: this small observation is not a robust performance claim.
Qwen passed 8/8 timing screens; BGE passed 7/8.

Cached-norm search alone, excluding embedding inference, improved by 4.44–4.52x
at 10,000 × 1024 vectors and k=5, and by 5.79–6.08x at 10,000 × 384. A separate
CPU index-building experiment used actual tokenizers with a fake embedding
backend; its speedup must not be presented as full Metal indexing throughput.
The actual Metal index check preserved vectors/chunks/metadata/query results and
SQLite reloads, while reducing tokenized texts from 24 to 12 for each model.

## Comparison with compiled MLX

The reference is an independent F32 graph using MLX 0.32.2 `mx.compile`, not the
mlx-embeddings package or every possible MLX configuration. Model inputs and batch
plans match. Compilation/first evaluation is recorded separately; steady-state
retraces are rejected. The API test includes different regex versions (Tessery
2025.9.18, reference 2026.9.3), so it does not isolate GPU-kernel performance.
The backend-scope harness exists for that separate question.

Here ratios are **Tessery latency / MLX latency**: below 1 is faster for Tessery.

| Token lengths | Qwen ratio | BGE ratio |
|---|---:|---:|
| 3 | 1.513–1.564 | 1.003–1.019 |
| 7 | 0.744–0.750 | Excluded: process drift |
| 16 | 0.657–0.671 | 1.036–1.056 |
| 24 | 0.938–0.959 | 1.105–1.143 |
| 160 | 1.412–1.417 | 1.535–1.564 |
| 512 | 1.620–1.623 | 1.650–1.653 |
| 4 × 33 | 1.405–1.411 | 1.752–1.770 |
| 3, 7, 10 | 1.039–1.047 | Excluded: MLX process drift |

All numerical/identity checks passed. Qwen passed 8/8 timing screens and BGE 6/8.
Tessery has a measured advantage for several short Qwen inputs; long inputs still
favor this MLX reference. Host-side plans cannot eliminate the long-input gap.

## Rejected experiments and remaining work

- [Eight-row K64 MLP suffix](GATED8_K64_EXPERIMENT.md): correct, but slower in all
  four targeted full-API cases. Additional shared memory did not pay off.
- [Residual + RMSNorm](RESIDUAL_NORM_EXPERIMENT.md): correct, but no useful stable
  full-API advantage that justifies the extra production route.
- [BGE attention shared-memory reduction](ATTENTION32_EXPERIMENT.md): promising
  isolated observations did not survive the repeated uncertainty check.
- M8 fused MLP: near parity; the original route remains.

These shaders stay in experimental branches. This cycle does not implement every
future hypothesis in the plan. The next bounded kernel experiments are Q/K
RMSNorm+RoPE and a small large-M GEMM tile search. ICB replay, pooled-last-layer
specialization, asynchronous preparation and a separate FP16 quality profile
remain later options, subject to correctness and full-API evidence.

## Validation and qualification boundary

- Full selected-source suite: **1173 passed, 1 skipped**, coverage **94.58%**.
- Focused Metal Shader Validation: **93 passed, 1 skipped**.
- Ruff, formatting, mypy, dependency/input checks and diff checks passed.
- Diagnostic/revision-tool tests passed after the last diagnostic-only edit.
- Short sequential 30-second rehearsals passed: Qwen 199 calls, BGE 1556 calls,
  all 14 scenarios, bounded execution plans and scratch, zero retained bytes
  after close. These are launch rehearsals, not long-run qualification.
- Wheel/sdist verification, isolated installed-wheel smoke on both models and
  installed dependency-policy checks passed. Local artifacts retain the logs.

The first Linux CI/Kaggle attempt exposed a test-collection import that depended
on running `python -m pytest` with the repository root on `sys.path`. The ranking
test now owns its small fake model; ordinary `pytest` collects it without that
assumption. This follow-up changes tests/documentation only, leaving the measured
library, wheel bytes and running native qualification snapshot unchanged. Kaggle
records its corrected test-suite commit separately from the native snapshot.

The prior 2h Qwen + 2h BGE and 4h Kaggle runs passed for **368d00a only**. New long
runs use an immutable source/wheel snapshot and must complete before this candidate
is called long-run qualified. Kaggle exercises portable CPU behavior, not Metal.
Publication to PyPI is a separate release step.

Compact measured results and hashes are checked in at
[`benchmarks/native-metal/performance-cycle-20260912/summary.json`](../benchmarks/native-metal/performance-cycle-20260912/summary.json).
Raw JSON/logs and local qualification state are gitignored under
`artifacts/performance-20260912/` and
`artifacts/qualification-performance-20260912/` respectively.
