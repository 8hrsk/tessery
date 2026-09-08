# Standalone API, 0.2 alpha

```python
from metal_inference import EmbeddingModel, MetalRuntime, cosine_search
model = EmbeddingModel.load(
    "/absolute/local/model", dimensions=384, max_length=512, max_pending=8,
)
vectors = model.encode(["текст", "text"], dimensions=1024)
model.close()
```

`load` verifies consumed bytes against registered immutable hashes and uploads
that validated snapshot into owned Metal buffers. Model code is never imported.
No pickle/torch loader, downloader, plugin registry, remote code hook, tokenization
cache or network library is involved.

`encode(texts, *, dimensions=None)` returns an ordered NumPy float32 matrix.
Empty batches return `(0, dimensions)` without GPU work. Whitespace-only strings,
invalid Unicode, more than 32 texts or 1 MiB total UTF-8 input are rejected.
A single pretokenizer segment is bounded to 64 KiB to constrain BPE resource use.
Dimensions must be integers 32..1024, excluding booleans. Configured max length
is an integer 1..512, including the special suffix. MRL truncation precedes L2.

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
Compute is float32, with the new space identifier `metal-inference-qwen3-f32-v1`.
Legacy Yuri embedding-space compatibility is not claimed.

`health()`, `memory_stats()` and `warmup()` report readiness, owned GPU buffer
bytes and perform an initial forward. `peak_bytes` is not process RSS or driver
memory. `cache_bytes=0` describes tensor caches; compiled pipelines are retained.

`cosine_search(query, documents, k=5)` returns `SearchHit(index, score)` records
from an existing embedding matrix. Ties keep document order. This small helper
uses NumPy and owns no persistent index or database.
`MetalRuntime.add` and `MetalRuntime.matmul` provide general float32 GPU operations
and return synchronized host arrays. Low-level buffers/dispatch are internal,
not a stable general tensor API.

## CLI

* `inspect --model-dir /absolute/model`: SHA-256/size validation.
* `embed --model-dir /absolute/model --dimensions 384`: read a JSON string array
  from stdin, write JSON embeddings to stdout. Optional `--input FILE` and
  `--output FILE`; existing output files are refused.
* `benchmark --model-dir /absolute/model --batch-size 1 --tokens 32 --iterations 10
  --warmup 2`: synthetic direct-API diagnostic with raw timing samples.

Invoke commands with `metal-inference` or `python -I -m metal_inference`.
Exit codes: 0 success; 2 invalid arguments, model/input/I/O or inference error.
Runtime failures contain stable safe codes. `embed` intentionally emits vectors
to the caller-selected destination. The library does not log/persist prompts or
vectors. HTTP/UDS, TLS and a Go supervisor are not part of this alpha.
