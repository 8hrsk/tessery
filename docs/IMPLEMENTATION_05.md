# Tessery 0.5 implementation and validation

The four stages were implemented in order: uint4 optimization, bounded workspace
reuse, persisted retrieval, and a local HTTP embeddings service. Existing Qwen
and BGE weights were reused in place. No new model was downloaded or copied.

## UInt4 results

On this Apple M1/macOS 26.3 host, the new original kernel decodes a 32x32 weight
block into 4 KiB threadgroup memory and computes an 8x32 output tile. Inputs,
decoded weights, and accumulation remain float32. Partial sums restart every
32 K elements before accumulation into the final fragment; no FP16 or full-model
F32 expansion is used. The route currently requires M divisible by 8, N divisible
by 32, and K divisible by 64. Other shapes retain the original four-row reduction.
These alignment conditions limit which real requests benefit.

Paired full-model measurements (two warmups, five samples per route):

| Batch / tokens per text | Original p50, s | Tiled p50, s | Speedup |
| --- | ---: | ---: | ---: |
| 1 / 8 | 0.1531 | 0.0443 | 3.46x |
| 1 / 32 | 0.5632 | 0.1285 | 4.38x |
| 4 / 32 | 2.2133 | 0.5096 | 4.34x |
| 1 / 128 | 2.2802 | 0.5722 | 3.98x |

Maximum full-model vector difference was 2.84e-7 (acceptance fixed at absolute
5e-6 plus relative 1e-4). The vectors are not bitwise identical to the old kernel.
The existing compatibility ID remains; this is numerical qualification on the
local pack, not proof of cross-device bitwise equivalence.

Four paired kernel shapes showed GPU median speedups of 3.62x, 4.77x, 5.33x and
4.33x. Both original and candidate were checked against float64 reference using
absolute 5e-5 plus relative 5e-5. Maximum candidate absolute error was 8.20e-5
on K=3072 (within the combined tolerance). A prior eight-row dot-product
experiment preserved bits but did not provide a stable benefit and was removed.

[Raw paired samples](../benchmarks/native-metal/delivery-20260908/quantized-paired.json)
include hashes of loaded shader/native bytes. These timings were captured before
the workspace-cache change, so both routes used the same uncached allocator.
Power, thermals and other desktop activity were uncontrolled. Results are not a
universal latency claim, and no direct MLX speed comparison was performed.

## Workspace reuse

The default 64 MiB cache retains only idle, exactly sized forward buffers.
IDs/lengths are uploaded anew. A repeated one-bucket request allocated only two
new buffers after warmup in the real Qwen test. The lease spans command completion
and output readback; errors discard unsubmitted commands before scratch returns
to the cache. Live buffers never alias each other. Old cache entries are evicted
when retention would exceed budget; oversized allocations are not retained.

`active_bytes` includes cached scratch; `cache_bytes` is its subset. Statistics
wait for a completed forward or trim. `trim_memory()` and `close()` release cached
allocations. Setting `workspace_limit_bytes=0` disables retention. Tests exercise
reuse, changed sizes, eviction, partial command abort, recovery and close.

Two-minute mixed-shape soak runs checked exact repeatability, live buffer counts
and cache bounds after every completed request:

| Model | Calls | Elapsed, s | Live owned bytes | Maximum sampled cache | Active/cache after close |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 uint4 | 23 | 124.56 | 335,218,496 | 66,746,880 | 0 / 0 |
| BGE F32 | 491 | 120.54 | 132,848,640 | 9,850,368 | 0 / 0 |

Both passed ten concurrency rounds (20 canceled tasks), recovery and fresh-model
reload. Qwen's cache evicted buffers as shapes changed; live weights stayed fixed.
The mixed workload includes long, unaligned inputs that retain the old uint4
kernel, so these call counts are not the aligned-matrix benchmark above.

[Qwen raw observations](../benchmarks/native-metal/delivery-20260908/qwen-workspace.json)
and [BGE observations](../benchmarks/native-metal/delivery-20260908/bge-workspace.json)
include process RSS separately from owned Metal buffers. Neither showed positive
end-to-start RSS growth, but a two-minute single-host run is not a long-term leak
proof. Qwen was recorded before the follow-up lock around public memory-stat
reads; BGE and the final tests include that change. Both runners verified their
source files remained unchanged during each run. The earlier 0.4 five-minute
runs remain historical evidence, not measurements of this new allocator.

## Retrieval and HTTP

The [RAG example](RAG.md) builds an immutable SQLite snapshot, reloads it, and
retrieves source-labelled passages. A real BGE test ranks France above gardening
for a capital-of-France query. Format validation rejects modified records and
mismatched model/revision/dimensions/max-length contracts. Snapshot publication
is atomic and refuses to overwrite an existing file. This is exact search for
small corpora; generation and ANN are outside this delivery.

The [HTTP API](HTTP_API.md) binds only loopback, bounds both connections and model
admission, supports optional bearer tokens, and cancels inference on response
timeout. A real BGE test requires HTTP vectors to equal direct API vectors bitwise.
Other tests cover malformed JSON/framing, authentication, host/origin restrictions,
timeouts, overload, recovery and safe error payloads. No background server is
installed. HTTP remains a trusted-local-app alpha, not a public production service.

CPU tokenization and submitted Metal work are not forcibly interrupted. Long-running
multi-host qualification, unaligned uint4 tiling, richer tensor operations, generation,
ANN and an external-service production lifecycle remain future work.

## Final validation

333 tests passed with 94.61% Python statement/branch coverage, including both local
models, native kernels/tensors, cache lifecycle, index persistence and HTTP sockets.
Ruff, formatting, strict mypy, frozen-input hashes and dependency policy passed.
Coverage does not measure Metal/C++ branches. No runtime dependencies were added.

A native 0.5.0a1 wheel was installed in a separate Python environment and exercised
on both models: embeddings, two-upload warm workspace reuse, trim/close, SQLite
retrieval round trip and HTTP equality. External DNS/connections were blocked by
a Python audit hook; only numeric loopback was allowed for HTTP. Wheel and source
archive have SPDX sidecars. Native builds and source archives remain local ignored
artifacts; no release tag or public binary release is implied.
