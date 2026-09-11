"""Execution-only padding must preserve masks, pooling and embedding contracts."""

import os
from pathlib import Path

import numpy as np
import pytest

from metal_inference.batching import length_batches
from tessery import EmbeddingModel, ModelProfile

pytestmark = [
    pytest.mark.metal,
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in local models"),
]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=["qwen", "bge"])
def model(request):
    if request.param == "qwen":
        directory = os.getenv(
            "METAL_INFERENCE_MODEL_DIR",
            str(Path.home() / ".mlx-serve/models/Qwen3-Embedding-0.6B-4bit-DWQ"),
        )
        profile = "qwen3-embedding-0.6b-dwq"
    else:
        directory = os.getenv(
            "METAL_INFERENCE_BGE_MODEL_DIR",
            str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs"),
        )
        profile = ModelProfile.from_file(
            os.getenv(
                "METAL_INFERENCE_BGE_PROFILE_FILE",
                str(ROOT / "model-manifests/bge-small-en-v1.5-hf-cache.json"),
            )
        )
    with EmbeddingModel.load(directory, profile=profile) as loaded:
        yield loaded


@pytest.mark.parametrize(
    "lengths",
    [
        [7],
        [9, 12],
        [31, 33],
        [63, 65],
        [127, 129],
        [511],
        [3, 7, 10],
        [10, 3, 7],
        [6, 9],
        [7, 11],
        [62, 122],
    ],
)
def test_aligned_api_matches_unaligned_same_tokens(model, lengths):
    special = 2 if model.descriptor.architecture == "bert_f32" else 1
    texts = [" token" * (n - special) for n in lengths]
    ids, actual = model._tokenizer.batch(texts, max_length=model.max_length)
    assert actual.tolist() == lengths
    expected = np.empty((len(texts), model.dimensions), np.float32)
    for rows, width in length_batches(actual, model._backend.max_padded_tokens):
        expected[rows] = model._backend.forward(
            np.ascontiguousarray(ids[rows, :width]), actual[rows], dimensions=model.dimensions
        )
    output = model.encode(texts)
    np.testing.assert_allclose(output, expected, atol=5e-6, rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1, atol=1e-6)
