# Mixed quantized projection tiles

This extends the [larger Qwen projection kernel](QUANTIZED_TILES.md) after
[bounded attention tails](ATTENTION_TAILS.md). The previous dispatcher used
the 16-row kernel only when the entire projection row count was divisible by
16. Otherwise all rows used the older eight-row path.

For the same five verified `(N,K)` projection shapes, the dispatcher now covers
all complete 16-row groups first, at most one complete eight-row group next,
and a bounded one-to-seven-row tail last. The eight-row kernel accepts a
starting-row offset in the existing `Params.n` field. Zero is the default,
preserving previous callers. The large kernel still starts at row zero; the
scalar/partial kernels already supported offsets. No ABI fields, global
workspace, model formats, attention or execution-padding policies change.

Examples use **projection rows**, which equal batch size times execution width:

| Rows | Selected decomposition |
| ---: | --- |
| 17 | 16 + scalar tail 1 |
| 24 | 16 + 8 |
| 31 | 16 + 8 + bounded tail 7 |
| 40 | 32 + 8 |
| 129 | 128 + scalar tail 1 |
| 264 | 256 + 8 |

Intervals are disjoint and cover every output row. One-to-four-row tails keep
the scalar reduction; five-to-seven-row tails keep the partial tile. Complete
regions retain the same 32-product F32 partial-sum order as the previous
kernels. Other projection dimensions and inputs with fewer than 16 rows keep
the earlier route. No weights are expanded or copied into a new model pack.

## Measurement method

The baseline dispatcher is preserved in the developer benchmark and comes
from `7673c19e9f483b5bad6426f3e8258e958529b2af`. It includes the existing 16-row
fast path for aligned inputs and all current attention optimizations. These
are paired before/after Tessery measurements, not a new MLX comparison.

Both harnesses use three warmups and 15 measured pairs in seeded randomized
order on Apple M1, macOS 26.3. GPU jobs run sequentially; thermals and background
activity are uncontrolled. Reports include raw samples and source/harness
hashes. The kernel harness also hashes the module holding the baseline route.

The [direct kernel report](../benchmarks/native-metal/mixed-20260909/kernels.json)
checks eleven row counts and all five Qwen projection shapes against independent
F64 matmul with unchanged `atol=5e-5, rtol=5e-5`. Selected and baseline outputs
match exactly. Each measured command repeats its complete route ten times;
reported GPU and wall times are divided by ten. This measures repeated dispatch
cost, not isolated-request latency. Numerical validation runs outside the timing
loop, and the CPU reference is limited to one library thread.

Initial single-invocation experiments repeatedly showed a slowdown for the
264-row down projection, but neighboring tests also showed large apparent
changes for **identical** 256/272-row routes. Moving validation outside the loop
alone did not fix those controls. With repeated dispatches, the 264-row down
projection instead improved in both the exploratory and final runs. These observations do not identify a
hardware cause or justify a shape-specific fallback. Small matrices can still
lose time to extra dispatches. The full-model paired measurements below are the
application-level acceptance check; kernel timings are diagnostic evidence, not
a promise that every projection becomes faster.

The final repeated-dispatch run still has noisy controls: the unchanged
8-row routes span 0.84–1.15x, while 256-row controls span 0.975–1.001x and
272-row controls span 0.987–1.062x. The 264-row down projection measures 1.13x
in that run. Consequently the direct timings do not establish a stable gain
for each individual shape. No thresholds were relaxed and no kernel guard was
chosen from an isolated timing anomaly.

## Full-model results

Both runs use the same local Qwen3 0.6B uint4 pack and complete `encode` calls,
including tokenization, batching, all layers and readback. The logical/execution
length distinction is unchanged: 17→24, 33→40, 65→72, while 129 and 257 stay
unpadded. Eight equal 33-token inputs produce 264 projection rows.

Reports: [first run](../benchmarks/native-metal/mixed-20260909/model.json),
[repeat](../benchmarks/native-metal/mixed-20260909/model-repeat.json).
Times below are median milliseconds; speedup is previous/selected.

| Logical tokens | First: previous → selected | First speedup | Repeat: previous → selected | Repeat speedup |
| --- | ---: | ---: | ---: | ---: |
| 7 | 34.28 → 34.31 | 0.999x | 33.36 → 33.08 | 1.008x |
| 16 | 48.96 → 49.04 | 0.998x | 54.42 → 47.02 | 1.157x |
| 17 | 94.47 → 84.80 | 1.114x | 104.05 → 88.10 | 1.181x |
| 33 | 159.64 → 144.77 | 1.103x | 168.35 → 142.87 | 1.178x |
| 65 | 289.09 → 238.42 | 1.213x | 298.12 → 243.55 | 1.224x |
| 128 | 399.79 → 393.66 | 1.016x | 411.01 → 407.66 | 1.008x |
| 129 | 592.16 → 489.42 | 1.210x | 605.21 → 497.97 | 1.215x |
| 257 | 1161.21 → 946.96 | 1.226x | 1200.31 → 972.11 | 1.235x |
| 8 × 33 | 1044.47 → 849.43 | 1.230x | 1070.46 → 841.79 | 1.272x |

All before/after output arrays in both reports match exactly (`max_abs_error = 0`).

The first run's unchanged 7/16/128-token controls are within about 2% of
unity. The repeat has a 1.157x result for the unchanged 16-token control,
so the short-input gains cannot be treated as precise stable improvements.
The 128-token control remains near unity in both runs. These controls and
uncontrolled thermals limit causal attribution; no new MLX speedup ratio is
claimed from these Tessery-only measurements.

## Validation

The full native/portable suite passes **820 tests**, with **94.60%** Python
coverage. Ruff formatting/lint, mypy, dependency policy and frozen-input checks
also pass. Coverage does not measure Metal/C++ branches.

**106 tests pass with Metal API and Shader Validation enabled.** They cover
all remainders at rows 17–31, longer 40/129/257-row inputs, 4095-row boundaries,
all five projection pairs and cancellation-heavy products. Outputs are checked
against F64 and the previous eight-row/scalar route without relaxing tolerance.
NaN output sentinels detect missing stores; an offset-only test proves that the
eight-row kernel preserves surrounding rows. Host tests verify disjoint complete
coverage through 4096 rows, and full Qwen tests confirm all seven projections
per layer use the large kernel at logical lengths 17/33/128/129.

The [Qwen stress run](../benchmarks/native-metal/mixed-20260909/qwen-stress.json)
passes four rounds with batches 1/3/8/32, four forward cancellations, one reload
and two closed lifecycles. Batched/independent vectors match exactly. After
trimming, active bytes return to the 335,218,496-byte weight baseline and cached
bytes are zero; closing releases all owned buffers.

The native wheel and source distribution pass package-layout verification.
The wheel was installed offline into a separate interpreter and passed Qwen/BGE
smokes. An isolated installed-package check confirms 196 large-tile dispatches
plus 196 eight-row dispatches at logical length 17, and 196 large-tile plus
196 scalar-tail dispatches at length 129, with finite unit-normalized vectors
and zero active buffers after close.

The native wheel and source distribution remain a local `0.6.0a1` candidate.
These short tests do not replace a multi-hour qualification of this exact
revision or establish results on other Apple GPUs. No PyPI release is published
as part of this optimization.

## Reproduction

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 METAL_INFERENCE_TEST=1 \
  .venv/bin/python -m pytest tests/real_metal/test_kernels.py \
  -k 'mixed_quantized or eight_row_offset' -s
METAL_INFERENCE_TEST=1 .venv/bin/python -m pytest --cov
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_mixed_quantized_tiles.py \
  --output artifacts/mixed-kernels-new.json
.venv/bin/python tools/benchmark_mixed_quantized_model.py \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/mixed-model-new.json
.venv/bin/python tools/stress_embeddings.py --rounds 4 \
  --model-dir "$QWEN_MODEL_DIR" --output artifacts/mixed-stress-new.json
```

Use existing verified local model paths and fresh output paths. No models or
runtime dependencies are downloaded by these commands.

The next profiling target is the remaining MLP projection cost: refresh direct
same-input comparisons with MLX before selecting gate/up reuse or another tile
layout. The noisy isolated controls here make measurement stability part of
that work, rather than evidence for a more aggressive dispatch guard.
