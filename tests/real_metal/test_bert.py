import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from metal_inference.errors import InvalidInputError
from metal_inference.weights import read_json
from metal_inference.wordpiece import WordPieceTokenizer
from tessery import EmbeddingModel, ModelProfile, cosine_search, get_profile

pytestmark = [
    pytest.mark.metal,
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in native GPU and local BGE tests"
    ),
]
ROOT = Path(__file__).resolve().parents[2]
PROFILE = ModelProfile.from_file(
    os.getenv(
        "METAL_INFERENCE_BGE_PROFILE_FILE",
        str(ROOT / "model-manifests/bge-small-en-v1.5-hf-cache.json"),
    )
)
MODEL = os.getenv(
    "METAL_INFERENCE_BGE_MODEL_DIR",
    str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs"),
)


@pytest.fixture(scope="module")
def reference():
    return json.loads(
        (ROOT / "benchmarks/observations/bge-small-en-v1.5/reference.json").read_text()
    )


@pytest.fixture(scope="module")
def model():
    with EmbeddingModel.load(MODEL, profile=PROFILE) as model:
        yield model


def test_wordpiece_matches_frozen_reference(reference):
    tokenizer = WordPieceTokenizer(read_json(MODEL, "tokenizer.json", profile=PROFILE))
    for row in reference["token_cases"]:
        assert tokenizer.encode(row["text"], max_length=row["max_length"]) == row["ids"]
    for row in reference["batches"]:
        ids, lengths = tokenizer.batch(row["texts"], max_length=512)
        np.testing.assert_array_equal(ids, row["ids"])
        np.testing.assert_array_equal(
            np.arange(ids.shape[1])[None, :] < lengths[:, None], row["mask"]
        )


def test_full_forward_matches_independent_cpu_reference(model, reference):
    assert model.descriptor.architecture == "bert_f32"
    assert model.descriptor.pooling == "cls"
    assert model.descriptor.quantization == "none"
    assert model.descriptor.compatibility_id == get_profile("bge-small-en-v1.5").compatibility_id
    assert model.health().compatibility_id == model.descriptor.compatibility_id
    for row in reference["batches"]:
        output = model.encode(row["texts"])
        expected = np.array(row["vectors"], np.float32)
        np.testing.assert_allclose(output, expected, atol=5e-6, rtol=1e-4)
        assert np.min(np.sum(output * expected, axis=1)) > 0.99999
    assert not any(
        name in sys.modules
        for name in ("mlx", "mlx_embeddings", "torch", "transformers", "tokenizers")
    )


def test_semantics_limits_and_memory(model, reference):
    texts = reference["batches"][0]["texts"]
    output = model.encode(texts)
    assert cosine_search(output[0], output[1:], k=1)[0].index == 0
    assert output[0] @ output[1] > output[0] @ output[2] + 0.1
    assert model.encode([]).shape == (0, 384)
    for dims in (32, 383, 385, True):
        with pytest.raises(InvalidInputError):
            model.encode(["hello"], dimensions=dims)
    resident = model.memory_stats().active_bytes - model.memory_stats().cache_bytes
    for _ in range(3):
        np.testing.assert_array_equal(model.encode(texts), output)
        assert (model.memory_stats().active_bytes - model.memory_stats().cache_bytes) == resident
    batch = model.encode(["hello"] * 32)
    np.testing.assert_allclose(batch, np.repeat(batch[:1], 32, axis=0), atol=1e-6)
    assert (
        (model.memory_stats().active_bytes - model.memory_stats().cache_bytes)
        == resident
        == 132848640
    )


def test_bert_truncation_keeps_sep_and_padding_mask(model):
    texts = [" token" * n for n in (509, 510, 511)]
    ids, lengths = model._tokenizer.batch(texts, max_length=512)
    assert lengths.tolist() == [511, 512, 512]
    assert [int(ids[i, int(n) - 1]) for i, n in enumerate(lengths)] == [102] * 3
    out = model.encode(texts)
    np.testing.assert_allclose(out[1], out[2], atol=1e-6)
    assert (model.memory_stats().active_bytes - model.memory_stats().cache_bytes) == 132848640


def test_persisted_retrieval_and_http_use_same_real_model(model, tmp_path):
    import http.client
    import threading

    from tessery import DocumentIndex
    from tessery.server import EmbeddingServer

    documents = {
        "france.txt": "Paris is the capital of France.",
        "garden.txt": "Bananas grow in warm tropical climates.",
    }
    index = DocumentIndex.build(model, documents)
    target = tmp_path / "retrieval.sqlite"
    index.save(target)
    loaded = DocumentIndex.load(target)
    assert (
        loaded.search(model, "What is the capital of France?", k=1)[0].chunk.source == "france.txt"
    )
    expected = model.encode(["Paris is the capital of France."])
    with EmbeddingServer(model, port=0) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request(
                    "POST",
                    "/v1/embeddings",
                    json.dumps({"input": documents["france.txt"]}),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                assert response.status == 200
                actual = json.loads(response.read())["data"][0]["embedding"]
                np.testing.assert_array_equal(np.asarray(actual, np.float32), expected[0])
            finally:
                connection.close()
        finally:
            server.shutdown()
            worker.join(timeout=5)
            assert not worker.is_alive()
