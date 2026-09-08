# Standalone API, 0.3 alpha

```python
from metal_inference import EmbeddingModel, MetalRuntime, cosine_search
model = EmbeddingModel.load(
    "/absolute/local/model", dimensions=384, max_length=512, max_pending=8,
)
vectors = model.encode(["текст", "text"], dimensions=1024)
model.close()
```

`load(..., profile="qwen3-embedding-0.6b-dwq")` accepts a built-in profile name or
an immutable `ModelProfile`. Omitted dimensions and max length use profile defaults.
It verifies consumed bytes against the profile's hashes and uploads
that validated snapshot into owned Metal buffers. Model code is never imported.
No pickle/torch loader, downloader, plugin registry, remote code hook, tokenization
cache or network library is involved.

`encode(texts, *, dimensions=None)` returns an ordered NumPy float32 matrix.
Empty batches return `(0, dimensions)` without GPU work. Whitespace-only strings,
invalid Unicode, more than 32 texts or 1 MiB total UTF-8 input are rejected.
A single pretokenizer segment is bounded to 64 KiB to constrain BPE resource use.
For the built-in Qwen3 profile, dimensions must be integers 32..1024, excluding
booleans, and max length is 1..512 including its special suffix. MRL truncation
precedes L2. BGE uses exactly 384 dimensions and 2..512 tokens, including CLS/SEP;
it pools the first encoder token and applies L2. No prompt prefix is added implicitly.

`encode_async` uses a private executor and bounded admission shared with sync
calls. Overload raises `OverloadError`. Canceling an await prevents queued work;
already submitted GPU work finishes, its result is discarded, and its slot is
released. Impose a deadline with:

```python
import asyncio
async with asyncio.timeout(5):
    vectors = await model.encode_async(["text"])
```

`close()` is idempotent, rejects subsequent calls, cancels queued async jobs and
waits for current GPU work before freeing model buffers. It cannot instantly
interrupt Metal. For asynchronous cleanup use `await asyncio.to_thread(model.close)`.
A context manager provides deterministic ownership.

`descriptor` exposes model/tokenizer revisions, dimension range, pooling,
quantization/storage/compute types, manifest digest and compatibility ID.
Compute is float32. The original Qwen3 profile retains `metal-inference-qwen3-f32-v1`.
Other profiles derive identifiers from their architecture, model metadata,
tokenizer/pooling contract and artifact digests. Changing physical filenames does
not change this identifier. Store the chosen output dimension, max length and
application query-prefix policy alongside vectors; those call options are not
encoded in the profile ID. Legacy Yuri embedding-space compatibility is not claimed.

`health()`, `memory_stats()` and `warmup()` report readiness, owned GPU buffer
bytes and perform an initial forward. `peak_bytes` is not process RSS or driver
memory. `cache_bytes=0` describes tensor caches; compiled pipelines are retained.

`cosine_search(query, documents, k=5)` returns `SearchHit(index, score)` records
from an existing embedding matrix. Ties keep document order. This small helper
uses NumPy and owns no persistent index or database.

## Metal tensors

`MetalRuntime.tensor(array)` uploads a nonempty NumPy float32 array, including
noncontiguous arrays and scalars, into an owned contiguous allocation. The
input is copied, with no implicit dtype conversion. `Tensor.shape`, `dtype`
and `nbytes` describe the allocation. Create tensors through the runtime;
the `Tensor` constructor and raw buffers/dispatch are internal interfaces.

* `a + b`: elementwise addition with identical shapes, without broadcasting.
* `a @ b`: matrix multiplication of `[M,K]` and `[K,N]` tensors.
* `a.transpose()`: matrix transpose into a new contiguous allocation.
* `a.silu()`: elementwise `x * sigmoid(x)` for any supported shape.
* `a.numpy()`: an independent, writable NumPy float32 copy.

All operands must belong to the same runtime. Operations preserve inputs and
return new tensors; no intermediate result is copied to the host. Matrix
multiplication transposes its right operand on Metal and releases that temporary
after execution. Commands execute eagerly and synchronize before returning;
this is not a lazy graph, asynchronous GPU queue or kernel fusion API.

```python
import numpy as np
from metal_inference import MetalRuntime

with MetalRuntime() as gpu:
    with gpu.tensor(np.ones((3, 4), np.float32)) as x:
        with x.transpose() as weights:
            with (x @ weights).silu() as result:
                host = result.numpy()
```

Tensor `close()` is idempotent, and tensor context managers free their allocation.
Closing a runtime frees all its remaining buffers and invalidates its tensors.
Subsequent computation/read raises `ClosedError`; mismatched shapes, dtypes or
runtimes raise `InferenceError`. Calls on one runtime are serialized, including
resource release. Garbage collection also releases unreferenced tensors; use
explicit cleanup when deterministic lifetime matters. Allocation size is bounded
to 2 GiB per buffer and by available Metal memory. Empty tensors, broadcasting,
views, arbitrary strides, mixed precision and autodiff are not implemented.

`MetalRuntime.add` and `MetalRuntime.matmul` still accept NumPy inputs and return
synchronized host arrays for callers that prefer a single-operation interface.

## CLI

* `profiles`: list built-in profile names and their data-only manifests.
* `inspect --model-dir /absolute/model`: SHA-256/size validation.
* `embed --model-dir /absolute/model --dimensions 384`: read a JSON string array
  from stdin, write JSON embeddings to stdout. Optional `--input FILE` and
  `--output FILE`; existing output files are refused.
* `benchmark --model-dir /absolute/model --batch-size 1 --tokens 32 --iterations 10
  --warmup 2`: synthetic direct-API diagnostic with raw timing samples.

`inspect`, `embed` and `benchmark` accept either `--profile NAME` or
`--profile-file FILE`. Defaults preserve the original Qwen3 behavior. `inspect`
now returns a `profile` object with artifact filenames, sizes and hashes.
See [model profiles](MODEL_PROFILES.md) for manifest trust and supported formats.

Invoke commands with `metal-inference` or `python -I -m metal_inference`.
Exit codes: 0 success; 2 invalid arguments, model/input/I/O or inference error.
Runtime failures contain stable safe codes. `embed` intentionally emits vectors
to the caller-selected destination. The library does not log/persist prompts or
vectors. HTTP/UDS, TLS and a Go supervisor are not part of this alpha.
