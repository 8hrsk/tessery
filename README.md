# Tessery — Metal Inference

An independent **Apache-2.0** inference engine for Apple Silicon, with its own
Objective-C++ Metal runtime, GPU kernels, SafeTensors reader, BPE and WordPiece tokenizers.
It does **not** use MLX, PyTorch, MPS, transformers or a hosted model service.
Yuri is not required.

**Status: working 0.3 alpha.** Qwen3 uint4 and BERT float32 adapters share the
same Metal runtime. Verified profiles cover Qwen3-Embedding-0.6B and
BGE-small-en-v1.5. Float32 tensors support GPU addition, matrix
multiplication, transpose and SiLU. This is an inference-focused foundation;
autograd, training, a general lazy tensor graph, generation and HTTP serving
are not implemented.

## Embeddings and retrieval

Requirements: macOS 14+, Apple Silicon, Python 3.12+. Xcode Command Line Tools
are required to build from source; a prebuilt native wheel needs no compiler.
No model is downloaded automatically.

```python
from metal_inference import EmbeddingModel, cosine_search

with EmbeddingModel.load("/absolute/path/to/Qwen3-Embedding-0.6B-4bit-DWQ") as model:
    vectors = model.encode([
        "What is the capital of France?",
        "Париж — столица Франции.",
        "Бананы растут в тропиках.",
    ])
    print(vectors.shape)  # (3, 384), numpy float32, unit L2
    print(cosine_search(vectors[0], vectors[1:], k=1))
    print(model.memory_stats())
```

Qwen3 output dimensions: 32..1024. Sequence limit: 1..512. Batch: up to 32 texts.
Large padded batches are split to bound temporary memory. Weights and tokenizer
load once. GPU forwards are serialized with bounded admission.
`encode_async` supports asyncio cancellation and `asyncio.timeout` deadlines.

## Model profiles

```python
from metal_inference import EmbeddingModel, list_profiles

print(list_profiles())
with EmbeddingModel.load("/absolute/path/to/bge-small-en-v1.5", profile="bge-small-en-v1.5") as model:
    vectors = model.encode(["A question", "A relevant passage"])
    print(vectors.shape)  # (2, 384), normalized CLS embeddings
```

BGE is an English BERT encoder; its native 384 dimensions are preserved, with a
2..512 token sequence limit including CLS/SEP. An explicit JSON profile can pin
another compatible set of weights/configuration/tokenizer without changing the
engine. Unknown architecture/tokenizer/pooling combinations are rejected.
See [profile format and existing cache reuse](docs/MODEL_PROFILES.md).

## General compute

```python
import numpy as np
from metal_inference import MetalRuntime

with MetalRuntime() as gpu:
    a = gpu.tensor(np.ones((2, 64), dtype=np.float32))
    b = gpu.tensor(np.ones((64, 3), dtype=np.float32))
    bias = gpu.tensor(np.ones((2, 3), dtype=np.float32))
    result = ((a @ b) + bias).silu()
    print(result.numpy())  # one explicit copy back to the host
```

Each operation completes synchronously; intermediate tensors stay in Metal
memory. Inputs are preserved, and results own separate allocations. Use tensor
context managers or `close()` to free allocations early; closing the runtime
frees every remaining owned buffer. NumPy-in/NumPy-out `gpu.add` and `gpu.matmul`
remain available. See the [tensor API](docs/API.md#metal-tensors) for limits.

## Build and run

```sh
uv sync --locked --group dev
uv run metal-inference inspect --model-dir /absolute/model
printf '["Привет", "Hello"]' | uv run metal-inference embed --model-dir /absolute/model
uv run metal-inference benchmark --model-dir /absolute/model --tokens 32 --iterations 10
uv build
```

`inspect` verifies all three consumed artifacts. Additional model-directory files
are ignored and never imported. Built-in profiles and caller-selected manifests
pin all consumed bytes. The wheel contains our native bridge and kernels, not model
weights or an embedded Python interpreter.

```sh
uv run ruff check .
uv run mypy
uv run pytest -m 'not metal'
METAL_INFERENCE_TEST=1 uv run pytest --cov
```

Native tests require both local models; paths and profile overrides are described
in [model profiles](docs/MODEL_PROFILES.md). Tests never download weights.

See [API](docs/API.md), [architecture/status](docs/STATUS.md),
[BGE validation](docs/BGE_VALIDATION.md), [earlier measurements](docs/NATIVE_METAL_REPORT.md),
and [provenance](PROVENANCE.md).
The old `yuri_mlx_embeddings` namespace contains the earlier protocol codec only.
It is not a dependency of `metal_inference`. Legacy Yuri vector reuse requires
its own compatibility report; Go fixtures do not block this standalone engine.
