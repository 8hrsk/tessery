import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from metal_inference.tokenizer import QwenTokenizer
from metal_inference.weights import read_json
from tessery import EmbeddingModel, cosine_search

pytestmark = [
    pytest.mark.metal,
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in pinned model"),
]
ROOT = Path(__file__).resolve().parents[2]
MODEL = os.getenv(
    "METAL_INFERENCE_MODEL_DIR",
    str(Path.home() / ".mlx-serve/models/Qwen3-Embedding-0.6B-4bit-DWQ"),
)


@pytest.fixture(scope="module")
def model():
    with EmbeddingModel.load(MODEL) as model:
        yield model


def test_frozen_token_ids_and_masks():
    tokenizer = QwenTokenizer(read_json(MODEL, "tokenizer.json"))
    corpus = json.loads((ROOT / "benchmarks/corpus.seed-v2.json").read_text())
    cases = {r["id"]: r["text"] for r in corpus["cases"]}
    for observation in json.loads(
        (ROOT / "benchmarks/observations/local-20260905/vectors.json").read_text()
    ):
        ids, lengths = tokenizer.batch(
            [cases[key] for key in observation["case_ids"]], max_length=512
        )
        np.testing.assert_array_equal(ids, observation["input_ids"])
        np.testing.assert_array_equal(
            np.arange(ids.shape[1])[None, :] < lengths[:, None], observation["attention_mask"]
        )


def test_semantics_repeat_and_no_framework(model):
    texts = [
        "What is the capital of France?",
        "Париж — столица Франции.",
        "Бананы растут в тропиках.",
        "Какая столица Франции?",
    ]
    vectors = model.encode(texts)
    assert vectors.shape == (4, 384) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2] + 0.2
    assert vectors[3] @ vectors[1] > vectors[3] @ vectors[2] + 0.2
    np.testing.assert_array_equal(vectors, model.encode(texts))
    assert cosine_search(vectors[0], vectors[1:3], k=1)[0].index == 0
    assert not any(
        name.split(".")[0] in {"mlx", "mlx_embeddings", "torch", "transformers", "tokenizers"}
        for name in sys.modules
    )


def test_dimensions_batch_order_and_memory(model):
    texts = ["a", "Короткий текст.", "Text with several words and punctuation!"]
    native = model.encode(texts, dimensions=1024)
    short = model.encode(texts, dimensions=32)
    np.testing.assert_allclose(
        short, native[:, :32] / np.linalg.norm(native[:, :32], axis=1, keepdims=True), atol=1e-6
    )
    reversed_vectors = model.encode(list(reversed(texts)), dimensions=1024)
    np.testing.assert_array_equal(native, reversed_vectors[::-1])
    assert (model.memory_stats().active_bytes - model.memory_stats().cache_bytes) == 335218496
    for _ in range(3):
        model.encode(texts)
    assert (model.memory_stats().active_bytes - model.memory_stats().cache_bytes) == 335218496


def test_boundaries_and_batch32(model):
    output = model.encode([" token" * n for n in (510, 511, 512)])
    assert output.shape == (3, 384)
    np.testing.assert_allclose(output[1], output[2], atol=1e-6)
    vectors = model.encode(["text"] * 32)
    assert vectors.shape == (32, 384)
    np.testing.assert_allclose(vectors, np.repeat(vectors[:1], 32, axis=0), atol=1e-6)


@pytest.mark.parametrize("tokens", [17, 33, 128, 129])
def test_large_tile_covers_all_seven_projections_per_layer(model, tokens):
    runtime = model._backend.runtime
    kernel = "linear4_16x32_k64"
    before = runtime.diagnostics()["dispatches"].get(kernel, 0)
    model.encode([" token" * (tokens - 1)])
    after = runtime.diagnostics()["dispatches"][kernel]
    assert after - before == 7 * model._backend.layers


def test_workspace_reuses_scratch_but_refreshes_inputs(model):
    texts = ["Alpha", "Beta"]
    expected = model.encode(texts)
    runtime = model._backend.runtime
    before = runtime.diagnostics()["allocations"]
    np.testing.assert_array_equal(model.encode(texts), expected)
    assert runtime.diagnostics()["allocations"] - before == 2  # IDs and lengths only.
    assert 0 < model.memory_stats().cache_bytes <= 64 * 1024 * 1024
    model.trim_memory()
    assert model.memory_stats().cache_bytes == 0
    assert model.memory_stats().active_bytes == 335218496
    np.testing.assert_array_equal(model.encode(texts), expected)


def test_unaligned_attention_uses_bounded_tile_per_layer(model):
    runtime = model._backend.runtime
    kernel = "attention_tail_128"
    before = runtime.diagnostics()["dispatches"].get(kernel, 0)
    model.encode([" token" * 128])
    assert runtime.diagnostics()["dispatches"][kernel] - before == model._backend.layers
