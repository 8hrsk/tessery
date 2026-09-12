# Long Qwen projections: 32-row weight reuse

Baseline `e49b39c`; candidate runtime source `6bc4035`; Apple M1, 2026-09-12.
Selected for the four guarded execution heights after the controls below.
The two-hour qualification of this runtime passed: 48,772 Qwen calls across
16 scenarios in 7200.104 seconds, with exact repeatability, bounded caches and
zero active/cached runtime bytes after close. Sources, wheel, loaded shader/native
library, supervisor and exact kernel counters were independently verified.
The qualified commit is `d7edc5bcda8926b411f5dff36b90ec8446b9187d`; local final reports
are in `artifacts/qualification-large-m-20260912/`.
This is the next bounded experiment from the performance plan. Existing local
Qwen3-Embedding-0.6B DWQ weights were reused without downloads.

## Selected change

`linear4_32x32_k64` keeps 256 threads, eight SIMD groups and the same 8 KiB
decoded-weight tile as `linear4_16x32_k64`. Each SIMD group computes two separate
8x8 row fragments, reusing decoded weights and the right-hand matrix fragment.
The output tile covers 32 rather than 16 rows. Independent K32 partial sums and
their accumulation order remain unchanged. Extra accumulators can affect
register pressure; achieved occupancy was not measured.

The host guard accepts only execution heights **128, 160, 256 and 512**, and the
five previously verified Qwen projection shapes. There is no new tail path.
Every other height/shape retains the original route, including M16/M24 fused MLP.
Logical sequence length and execution height differ: `4x33` executes as M160,
whereas `[159,160]` executes as M320 and stays on the old route.

## Measurements

Two microbenchmark processes covered 20 M/N/K combinations each, checking against
an independent F64 reference with the existing tolerances. Every output matched
the old tile exactly. Most kernel-only speed ratios were around 1.16–1.18 versus
the existing 16-row tile. The M512/N2048 case was inconsistent between runs;
these isolated timings alone are not grounds for production selection.

Full API comparisons used two fixed model instances with independent native-plan
caches, identical baseline A/B labels, 18 samples per label, three balanced
six-permutation blocks, and at least one second of warmup per label. A fresh
second process reversed case order. Every timed call checked exact embeddings,
plan-cache hits and the expected 140 candidate dispatches per selected batch.

Ratios below are old latency / candidate latency, so values above 1 are faster.

| Input lengths | First process | Reverse process |
|---|---:|---:|
| 128 | 1.0850 | 1.0818 |
| 160 | 1.0838 | 1.0833 |
| 256 | 1.0849 | 1.0852 |
| 512 | 1.0456 | 1.0600 |
| 4x33 | 1.0766 | 1.0768 |
| 24, unchanged | 0.9801 | 1.0031 |
| 159,160, unchanged | 0.9974 | 1.0003 |

All identical A/B controls passed the existing 10% noise screen. Approximate
Student-t95 intervals on three block-mean log ratios had lower bounds above 1
for every selected case in both runs (minimum 1.031 for M512). These small-sample
intervals are not portable guarantees. Variation in the untouched controls shows
that benchmark conditions can affect a few percent of timing; thermal and
allocation effects were not independently established.
The paired medians and geometric intervals describe different estimators.

The measured prototype sampler and shader bytes are preserved in ignored
artifacts and match the hashes in both raw reports. After selecting the host
guard, the development sampler explicitly restores the old 16-row dispatch on
its baseline instance and adds helper/dependency/profile provenance checks.
It must not silently compare the new route against itself in future runs.

A separate four-process revision comparison checked the unpatched runtime
against `e49b39c`, using a common immutable sampler and source/helper fingerprints.
All vectors and provenance checks passed. However, baseline timing changed
11–13% while candidate timing stayed nearly constant. At 512 tokens the two
baseline/candidate ratios were 1.0766 and 0.9551; at 128 they were 1.0762 and
0.9718. The existing 15% drift screen passes these cases, but opposite effect
directions are not evidence of a production improvement. This contradictory run is retained as a limitation; its aggregate medians alone
cannot establish improvement.

## Shader-library and allocation controls

Two additional processes used three fixed model instances: old shader with old
16-row routing, new shader with old routing, and new shader with the new route.
The old model also had identical A/B labels. Every one of the 24 permutations of
four labels was measured once per size, after warmup; the second process reversed
both allocation and case order. Each runtime recorded the shader bytes actually
loaded, immutable source/helper hashes, identical model/profile identities,
exact vectors and 140 expected projection dispatches with warm native-plan hits.

| Input length | Old version / new version, first | Reverse allocation | New-library old / new route, first / reverse |
|---|---:|---:|---:|
| 128 | 1.0827 | 1.0808 | 1.0776 / 1.0792 |
| 512 | 1.0475 | 1.0482 | 1.0749 / 1.0781 |

The full API improvement survives the old shader and reversed allocation controls:
about **8% at 128 tokens** and **4.8% at 512** on this M1. Earlier paired runs also
support the M160/M256 guard and the four-by33 batch; the three-library diagnostic
itself measured only 128/512. This is a local steady-state result, not a device-wide
performance guarantee or a new comparison against MLX.

At 512, old and new shader libraries on the old route had nearly equal GPU time
(about 518 ms), while the new route took about 479 ms. The new-library old-route
instance had about 14 ms more API latency outside recorded GPU/encoding time.
That overhead persists in reversed allocation order and remains unexplained; it
must not be presented as a proven shader-library effect. Total old-version versus
new-version API ratios above include it. The earlier independent-process drift
remains a warning against comparing isolated process medians without interleaved
controls. No GPU clock or thermal telemetry was available to prove its cause.

Both runs kept normal system memory pressure, with no growth from the initial
swap usage. The models reused local weights and closed all runtime allocations.
The controlled result supports the narrow guard; it does not justify expanding
it to additional shapes or changing numerical tolerance.

Machine-readable controls and raw artifact hashes are recorded in
[`summary.json`](../benchmarks/native-metal/library-control-20260912/summary.json).

## Validation and rejected attention experiment

- Final selected-source suite: **1351 passed, 1 skipped**, coverage **94.59%**.
  This includes direct M128/M160/M256/four-by33 model regressions and five
  shader-library diagnostic unit tests.
- Expanded raw-kernel and model Shader Validation: **40 passed**.
- Raw tests cover random, zero and cancellation-heavy inputs. Each kernel gets a
  newly initialized output plus a guard row, so missing writes cannot inherit
  baseline values. Model checks include ragged inputs and neighboring sizes.
- The original per-layer projection test counts both large tiles; separate guard
  checks ensure complete coverage and preserve nonselected fallback shapes.
- Ruff, formatting, mypy and dependency/input checks passed.

The [attention traversal experiment](QWEN_ATTENTION_KVGROUP_EXPERIMENT.md) passed
CPU mapping and Shader Validation but showed no useful full-API improvement.
Its shader remains in a separate experimental branch. Only its report is retained
with the selected change.

The previous multi-hour qualification belongs to the previous runtime. This
change passed its own pinned native qualification described above. A 60-second rehearsal passed
406 calls across 16 scenarios, including each selected height; exact kernel
counters were independently checked. The installed wheel passed the standard
Qwen/BGE smoke check and seven dedicated candidate cases (two repeats each):
exact vectors, 1680 new-kernel dispatches, warm plan hits and zero cached/active
runtime bytes after close. Every installed package member matches the wheel and
source checkout; see
[`validation-summary.json`](../benchmarks/native-metal/large-m-20260912/validation-summary.json). Final installation and qualification launch details are
recorded under `artifacts/qualification-large-m-20260912/`. The CPU-only Kaggle
soak does not exercise this Metal kernel. The subsequent [compiled MLX comparison](COMPILED_MLX_COMPARISON_20260913.md)
measures this qualified runtime and records its remaining long-input gap.

Machine-readable paired evidence:
[`paired-summary.json`](../benchmarks/native-metal/large-m-20260912/paired-summary.json).
Raw runs are gitignored under `artifacts/performance-next-20260912/` in the main
checkout. Experiments were isolated in separate worktrees; no model downloads or
production rollback were needed.
