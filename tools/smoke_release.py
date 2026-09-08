"""Run with an isolated wheel interpreter (-I) on a qualified Apple Silicon host."""

import importlib.metadata
import os
from pathlib import Path

import numpy as np

from tessery import EmbeddingModel, ModelProfile, __version__, list_profiles

assert __version__ == importlib.metadata.version("tessery")
assert list_profiles()
root = Path(__file__).resolve().parents[1]
models = [
    (
        os.getenv(
            "METAL_INFERENCE_MODEL_DIR",
            str(Path.home() / ".mlx-serve/models/Qwen3-Embedding-0.6B-4bit-DWQ"),
        ),
        "qwen3-embedding-0.6b-dwq",
    ),
    (
        os.getenv(
            "METAL_INFERENCE_BGE_MODEL_DIR",
            str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs"),
        ),
        ModelProfile.from_file(
            os.getenv(
                "METAL_INFERENCE_BGE_PROFILE_FILE",
                str(root / "model-manifests/bge-small-en-v1.5-hf-cache.json"),
            )
        ),
    ),
]
for directory, profile in models:
    with EmbeddingModel.load(directory, profile=profile) as model:
        texts = ["Paris is in France.", "A short test."]
        first = model.encode(texts)
        np.testing.assert_array_equal(first, model.encode(texts))
        assert np.isfinite(first).all()
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-5)
        model.trim_memory()
        assert model.memory_stats().cache_bytes == 0
    assert model.memory_stats().active_bytes == 0
print({"wheel_smoke": "passed", "version": __version__, "models": len(models)})
