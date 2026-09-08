# Local HTTP API

The standalone `metal-inference serve` command loads one model once and serves
embeddings to trusted applications on the same Mac. It binds **127.0.0.1 only**.
This alpha uses Python's standard-library HTTP server with bounded worker threads;
it is not an internet-facing production server, TLS endpoint or process supervisor.
Python documents that boundary in its [HTTP server reference](https://docs.python.org/3.12/library/http.server.html).
No external server package or model download is required.

```sh
metal-inference serve --model-dir /absolute/model --port 8765
```

Existing BGE cache:

```sh
metal-inference serve \
  --model-dir "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json
```

Ctrl-C or SIGTERM stops accepting requests, waits for request workers, and closes
the model. It waits for an already submitted GPU command to finish. The command
prints the listening address to stderr; it does not log prompts, vectors, tokens
or request traces. It stays in the foreground, with no automatic startup service.

## Routes

| Method/path | Result |
| --- | --- |
| `GET /health` | Loaded/ready state and compatibility ID; 503 when not ready |
| `GET /v1/models` | Descriptor of the single loaded model |
| `GET /v1/memory` | Owned active/peak bytes and retained workspace cache bytes |
| `POST /v1/embeddings` | Ordered embeddings with model and compatibility ID |

```sh
curl http://127.0.0.1:8765/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["Hello","Привет"],"dimensions":384}'
```

The request accepts `input` (a string or at most 32 strings), optional `dimensions`,
and optional `model` (must equal the loaded model ID). Unknown fields, duplicate
JSON keys, non-finite numbers and unsupported dimensions are rejected. BGE always
uses 384 dimensions; Qwen supports 32..1024.

The response resembles the common embeddings API shape, with `object: "list"`,
`model`, `compatibility_id`, and `data` records containing `object: "embedding"`,
`index`, and `embedding` (JSON floats). This is a bounded API, not a promise of
complete OpenAI API compatibility: no token-array inputs, base64 encoding, usage
accounting, model switching, generation, streaming or remote index management.
The [RAG library](RAG.md) manages local document indexes separately.

## Limits and failure behavior

Defaults: `--max-pending 8` model jobs, `--max-connections 16` HTTP workers,
`--request-timeout 30` seconds for inference including the model queue, and
`--workspace-limit-mib 64` cached GPU scratch. Use `--workspace-limit-mib 0` to
release scratch after every forward. Connection limits apply before spawning a
worker; excess connections/jobs receive 429 and `Retry-After: 1`.

Body size is at most 2 MiB JSON and model input at most 1 MiB decoded UTF-8.
One `Content-Length` is required, `Content-Type` must be `application/json`, and
chunked transfer/`Expect: 100-continue` are unsupported. Connections close after
one response. Socket inactivity timeout is `min(10, request_timeout)` seconds;
it is separate from the inference timer. A trickling client is not subject to a
hard total network deadline, so this remains a trusted-local-app service.

Inference timeout returns 504 and cancels queued work. GPU work already submitted
finishes and its result is discarded; it is not forcibly terminated. CPU
tokenization is not instantly interrupted. Shutdown may outlast the response
timeout while such work drains. Errors contain stable codes, not exception text:
400 invalid input, 401 unauthorized, 403 forbidden, 404 missing route, 413 oversized
body, 415 wrong media type, 417 unsupported expectation, 429 overloaded,
500 inference failure, 504 deadline exceeded.

Optional authentication: `--token-file /absolute/token.txt`, containing an ASCII
bearer token of 16..4096 printable non-whitespace characters (a trailing newline is allowed).
Send `Authorization: Bearer TOKEN` on every route. Tokens are not printed. Without
this option, local processes can call the service. Host must be
`127.0.0.1:PORT` or `localhost:PORT`; browser `Origin` requests are rejected and
CORS is not enabled. Do not expose the service through a forwarding proxy.

## Embedding in another Python application

```python
from metal_inference import EmbeddingModel
from metal_inference.server import EmbeddingServer

with EmbeddingModel.load('/absolute/model') as model:
    with EmbeddingServer(model, port=8765) as server:
        server.serve_forever()
```

The caller owns the model. If a separate thread runs `serve_forever`, call
`shutdown` from a different thread, then `server_close`, and finally close the
model. This follows Python's [socket server lifecycle](https://docs.python.org/3.12/library/socketserver.html).

Memory statistics wait for an in-progress forward or cache trim to finish, so
the active/cache counts describe the same completed state.
