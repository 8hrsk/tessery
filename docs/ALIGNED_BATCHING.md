# Execution padding for Metal tiles

Tessery can pad an execution bucket to the next multiple of eight tokens. This
avoids slow projection tails and, at suitable widths, enables tiled attention.
Tokenization, real sequence lengths, masks, pooling, output order and public
embedding contracts remain unchanged. The caller does not need a new API flag.

## Selection and bounds

Bucket membership still comes from the existing stable length planner. Extra
padding is considered only for the supported Qwen uint4 and BERT F32 adapters,
at widths of at least five tokens. It never exceeds the caller's `max_length`,
twice the shortest real length in the bucket, or the backend's 4096 padded-token
budget. Padding uses the tokenizer's own pad ID, with unchanged real lengths.

Projection rows are `batch_size * width`. An already aligned matrix keeps its
original shape unless the next eight-token boundary also enables tiled
attention (at least 64 tokens and a multiple of 32). For Qwen widths of 128 or
more, padding is selected only at these attention boundaries: the small
projection tail alone did not justify a broader policy in the initial study.
BERT can also benefit from projection alignment at larger widths because its
unaligned F32 matrices otherwise use the scalar path.

This is a conservative heuristic measured on M1 with the existing Qwen3 0.6B
uint4 and BGE-small F32 packs. It is not an autotuner or a speed guarantee for
every model profile or Apple device. In particular, it does not round every
sequence up to 32: that can add enough work to erase the kernel advantage.

## Reproduction

`tools/benchmark_alignment.py` compares the previous and selected plans through
the complete synchronous `encode()` API, including tokenization and admission.
Every pair uses exactly the same texts and checks all output components at
`atol=5e-6, rtol=1e-4`. The old plan still uses the current kernels, isolating
the execution padding change. Case metadata records actual lengths and bucket
shapes, and reports contain raw timing samples and source/harness hashes.

```sh
python tools/benchmark_alignment.py --model-dir /path/to/qwen \
  --samples 15 --lengths 3 4 5 6 7 8 9 12 15 31 33 63 65 127 129 255 511 \
  --output artifacts/alignment-qwen.json
# BGE: add --profile-file when using the local HF cache manifest.
# --explore compares unrestricted bounded multiples of eight and thirty-two.
```

Each case uses three warmups and randomized paired order. Power, thermals and
background activity are uncontrolled; identical-plan controls measure this
noise and must not be described as algorithmic improvements. The measurements
compare Tessery before/after, not MLX. Existing MLX comparisons in
[the attention report](TILED_ATTENTION.md) retain their original scope and date.

## M1 results, 2026-09-09

The final runs used 15 paired samples for each of 20 cases per model, with the
same existing local model bytes. Reports:
[Qwen](../benchmarks/native-metal/alignment-20260909/qwen.json) and
[BGE](../benchmarks/native-metal/alignment-20260909/bge.json).

| Model | Real tokens | Previous p50, ms | Selected p50, ms | Speedup |
|---|---:|---:|---:|---:|
| Qwen | 7 | 73.41 | 35.71 | 2.06x |
| Qwen | 31 | 167.99 | 136.15 | 1.23x |
| Qwen | 63 | 335.37 | 252.56 | 1.33x |
| Qwen | 127 | 652.56 | 514.21 | 1.27x |
| Qwen | 255 | 1447.62 | 1054.47 | 1.37x |
| Qwen | 511 | 3535.66 | 2215.00 | 1.60x |
| BGE | 7 | 9.11 | 9.01 | 1.01x |
| BGE | 31 | 27.85 | 18.83 | 1.48x |
| BGE | 63 | 56.29 | 31.14 | 1.81x |
| BGE | 127 | 123.66 | 59.90 | 2.06x |
| BGE | 255 | 271.22 | 129.57 | 2.09x |
| BGE | 511 | 713.31 | 234.35 | 3.04x |

The mixed `[127, 65, 33]` case measured 1.13x for Qwen and 1.23x for BGE.
Maximum absolute vector differences across all measured pairs were 2.09e-7
and 1.49e-7 respectively, within the unchanged 5e-6/1e-4 model tolerances.
Every added position is masked and the real token lengths stay unchanged.

Small timings are noisy: identical-plan controls ranged from 0.895x (BGE,
eight rows of 33 tokens) to 1.275x (BGE, four tokens). Neither is an algorithmic
change. Qwen's unchanged 129-token case measured 1.001x. The weak 1.01x result
for BGE at seven tokens and 1.03x for Qwen at 65 tokens should not be interpreted
as established improvements. The stronger long-input gains warrant the bounded
policy, but these single-host runs do not establish universal performance.

## Validation and remaining work

The complete native/local suite passed: **441 tests, 94.52% coverage**. New
tests compare aligned public API outputs against unaligned real-model forwards
at short, ragged and 511-token inputs. Portable tests exercise bucket identity,
stable ordering, caller context limits, the shortest-row padding bound and
the 4096-token budget, including the 9-by-455 case that cannot be rounded up.
API tests check real token IDs, unchanged lengths and use of the tokenizer's
pad ID. Ruff, mypy, dependency policy and frozen input checks also passed.

Four seeded stress rounds per model covered batch sizes 1/3/8/32, four forward
cancellations, queue overload/recovery, one reopen and two closed lifecycles.
See [Qwen stress](../benchmarks/native-metal/alignment-20260909/qwen-stress.json)
and [BGE stress](../benchmarks/native-metal/alignment-20260909/bge-stress.json).
Trimmed active allocations returned to 335,218,496 and 132,848,640 bytes
respectively, with zero cached bytes; close released all active allocations.
The new macOS arm64 wheel was installed in a separate temporary environment
and passed isolated Python smoke checks with both models.

This is a short regression/stress qualification. The older multi-hour runs do
not qualify this changed batching policy. The next performance target is
projection work: larger quantized tiles for Qwen and full F32 tiles with a
bounded tail for BGE, with the same numerical gates. MLX comparisons need a
fresh run before attributing any change in the cross-engine gap to this work.
