# Performance and reliability diagnostics

Tessery uses its own Metal runtime and kernels. These tools require local weights
and the native macOS arm64 build; they do not download models or import MLX.
All results are single-host diagnostics, not production qualification or measured
performance parity with MLX. Run GPU workloads sequentially. Power, thermal state
and other applications can affect timings; raw samples are more useful than one
best number.

## Reproduce an end-to-end run

```sh
.venv/bin/python tools/build_native.py
.venv/bin/python tools/diagnose_metal.py \
  --model-dir /absolute/local/model \
  --profile qwen3-embedding-0.6b-dwq \
  --output artifacts/performance/qwen-run.json --label local-run \
  --iterations 15 --soak-seconds 300
```

For the existing BGE cache use its blobs directory and
`--profile-file model-manifests/bge-small-en-v1.5-hf-cache.json` instead of
`--profile`. Output files must be new. `artifacts/` is gitignored. The tool
atomically replaces partial matrix/soak checkpoints and marks ordinary failures in the JSON; externally
killed or hung processes can leave `status: running`, which is not a passed run.
A soak checkpoint is sampled about every ten seconds, after a complete request.

The matrix covers batch 1/4/16/32 and verified actual sequence widths 32/128/512.
Use `--batch-sizes`, `--lengths`, `--iterations`, `--warmup` to bound a run. The
profile determines output dimensions and maximum length. Each row records load
separately, excludes warmup, reports raw end-to-end latency and p50/p95, and
computes throughput from all completed texts divided by elapsed sample time.
Small sample counts make tail percentiles especially uncertain.

The soak cycles short, long, mixed-length and batch-32 requests. It checks repeated
outputs, finite normalized embeddings through the backend, and exact recovery of
owned Metal-buffer bytes after each call. Separate async rounds exercise concurrent
requests, cancellation and recovery. After close, owned buffers must reach zero,
new requests must fail, and a fresh model must load and run. Deterministic tests
cover admission saturation, queued/in-flight cancellation, mixed sync/async order,
and close while work is queued. A five-minute run cannot establish hours-long
stability or recovery from every hardware fault.

## Counter meanings

`MetalRuntime.diagnostics()` returns a snapshot of cumulative allocation counts,
allocated bytes, encoded dispatches by kernel, successful completed commands,
command encoding time, submit/wait time and GPU command duration. GPU duration is
read after completion using Apple's [command-buffer timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime).
`gpu_timed_commands` distinguishes available timings from absent ones.

Encoding includes Python, driver work and allocations inside a command. Submit/wait
includes GPU execution and scheduling: **do not add it to GPU time**. Dispatch
counts include encoded work subsequently aborted, while command timing counts
only successful completed commands. This is not per-kernel GPU counter sampling.
The extra counters are lightweight but timings are from an instrumented runtime.
The runtime records a digest of the exact shader bytes supplied to compilation,
plus a digest of the library file checked before and after loading. The diagnostic
also records its runner digest and rejects workspace source changes during a run.
These checks catch ordinary concurrent edits; they are not signed binary attestations.
Unavailable GPU durations produce null speedup rather than a fabricated zero or NaN.

`active_bytes` and `peak_bytes` count requested owned Metal-buffer bytes, excluding
Python/driver objects and caches. Peak is cumulative for the runtime, not reset for
each matrix row. Current process RSS comes from macOS `ps`; process peak RSS from
`getrusage` is a lifetime maximum. Neither is a substitute for owned-buffer
accounting or a direct measure of GPU-only memory. RSS fluctuations alone are not
proof of a leak, and zero owned buffers after close does not imply zero process RSS.

## Implemented optimizations and bounds

Aligned F32 matmul uses an original 8x32 output tile with four SIMDgroups and
F32 matrix operands/accumulators. It reads existing transposed-layout weights
without a new copy. Routing requires M divisible by 8, N divisible by 32 and K
divisible by 8, with K <= 512. Smaller, unaligned or longer-K products retain the
original SIMD reduction. Long-K tiled accumulation was evaluated but not enabled
after a numerical acceptance failure. FP16 conversion and fast math are not used.
The design uses Apple's [SIMDgroup matrix primitives](https://developer.apple.com/videos/play/tech-talks/10858/),
not source copied from another tensor runtime.

Length buckets use a stable ascending sort, cap padding width at twice the
shortest input in the bucket and honor the backend's padded-token budget. Output
rows are scattered back to the caller's original order. No text, tokenization,
truncation, pooling or normalization rule changes. Batch composition changes, so
numerical regression checks cover saved reference vectors and mixed batches.
For one length-512 input and 31 length-2 inputs, work falls from 16,384 padded
positions to 574. This arithmetic reduction is not itself a timing claim.

Sync and async inference now share one executor submission queue; synchronous
callers cannot bypass already queued async work by racing for the GPU lock.
Closing cancels queued sync calls with `ClosedError`. Cancellation during CPU
normalization/tokenization remains cooperative only at the surrounding boundaries;
it does not interrupt a tokenizer call or an in-flight Metal command instantly.
Partial native command creation cleans up its state so the runtime can be retried.
A fault-injection test verifies cleanup of a partially begun command.

## Isolate the effects

`tools/benchmark_matmul.py --output artifacts/performance/matmul.json` alternates
the two kernels in randomized order with two warmup and twenty measured samples.
It compares with an independent NumPy float64 product and records absolute error.
The longer-K candidate is evidence for future work, not the selected default.

`tools/benchmark_padding.py` takes the same model/profile/output arguments as the
main diagnostic. It alternates historical global padding and new length buckets
on the **same current backend**, checks vector agreement and original order, and
records planned padded positions. Its defaults are 32 texts with one length-512
input; `--batch-size 8 --long-tokens 128` bounds the heavier Qwen comparison.
Timing excludes tokenization and admission,
so it isolates batch planning from the simultaneous matmul change.

Future work includes long-K accuracy-preserving tiling, quantized tiled GEMM,
tiled attention, controlled per-kernel profiling and a longer soak on another Mac.
Local research/review reports are deliberately kept outside Git under
`artifacts/research/`.

## 0.5 workspace and uint4 changes

`active_bytes` now includes cached scratch; `cache_bytes` is its retained subset.
Soak checks live bytes (`active_bytes - cache_bytes`) and the cache budget after
every request, allowing bounded cache contents to change with shapes. Disable
the cache with `EmbeddingModel.load(..., workspace_limit_bytes=0)` or trim via
`model.trim_memory()`. Buffer reuse happens after GPU completion/readback.

`tools/benchmark_quantized.py --model-dir /absolute/qwen --output NEW.json`
compares the original and tiled uint4 kernels against float64, then runs paired
full-model comparisons on the same loaded Qwen weights. It excludes two warmups
and records 20 kernel samples or five full-model samples per route. Both routes
use the same batching and workspace configuration. No weights are downloaded.
See the [0.5 delivery report](IMPLEMENTATION_05.md) for measured bounds.
