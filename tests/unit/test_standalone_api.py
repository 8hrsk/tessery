import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from metal_inference.api import EmbeddingModel
from metal_inference.errors import (
    ClosedError,
    ConfigurationError,
    InferenceError,
    InvalidInputError,
    OverloadError,
)
from metal_inference.retrieval import cosine_search


class Tokenizer:
    def batch(self, texts, *, max_length):
        return np.array([[int(t)] for t in texts], dtype=np.uint32), np.ones(len(texts), np.uint32)


class Backend:
    max_padded_tokens = 2
    runtime = SimpleNamespace(active_bytes=100, peak_bytes=200)

    def __init__(self):
        self.calls = []
        self.closed = False
        self.entered = threading.Event()
        self.release = None

    def forward(self, ids, lengths, *, dimensions):
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5)
        assert not self.closed
        self.calls.append(ids[:, 0].tolist())
        output = np.zeros((len(ids), dimensions), np.float32)
        for i, value in enumerate(ids[:, 0]):
            output[i, int(value)] = 1
        return output

    def close(self):
        self.closed = True


@pytest.fixture
def model():
    backend = Backend()
    model = EmbeddingModel(backend, Tokenizer(), 384, 512, 2)
    yield model
    if backend.release:
        backend.release.set()
    model.close()


def test_order_subbatches_and_lifecycle(model):
    assert model.encode([]).shape == (0, 384)
    assert model._backend.calls == []
    vectors = model.encode(["3", "1", "2"], dimensions=1024)
    assert vectors.dtype == np.float32
    assert vectors.argmax(axis=1).tolist() == [3, 1, 2]
    assert model._backend.calls == [[3, 1], [2]]
    assert model.health().ready
    assert model.memory_stats().active_bytes == 100
    model.close()
    model.close()
    assert not model.health().ready
    with pytest.raises(ClosedError):
        model.encode([])


@pytest.mark.parametrize(
    "texts,dims",
    [
        ("text", 384),
        (b"text", 384),
        (None, 384),
        ([" "], 384),
        (["\ud800"], 384),
        ([4], 384),
        (["0"] * 33, 384),
        (["x" * 1048577], 384),
        (["я" * 600000], 384),
        (["0"], True),
        (["0"], 31),
        (["0"], 1025),
    ],
)
def test_input_errors(model, texts, dims):
    with pytest.raises(InvalidInputError):
        model.encode(texts, dimensions=dims)


@pytest.mark.parametrize(
    "options",
    [
        {"dimensions": True},
        {"dimensions": 31},
        {"max_length": 513},
        {"max_length": False},
        {"max_pending": 0},
        {"max_pending": 65},
    ],
)
def test_load_option_validation(options):
    with pytest.raises(ConfigurationError):
        EmbeddingModel.load("/not-read", **options)


def test_overload_and_queued_cancellation(model):
    async def run():
        backend = model._backend
        backend.release = threading.Event()
        first = asyncio.create_task(model.encode_async(["0"]))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        queued = asyncio.create_task(model.encode_async(["1"]))
        await asyncio.sleep(0)
        with pytest.raises(OverloadError):
            await model.encode_async(["2"])
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await asyncio.sleep(0)
        backend.release.set()
        assert (await first).shape == (1, 384)
        assert backend.calls == [[0]]
        assert (await model.encode_async(["2"])).argmax() == 2
        assert (await model.encode_async([])).shape == (0, 384)

    asyncio.run(run())


def test_inflight_cancellation_discards_and_releases(model):
    async def run():
        backend = model._backend
        backend.release = threading.Event()
        job = asyncio.create_task(model.encode_async(["0"]))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        backend.release.set()
        result = await model.encode_async(["1"])
        assert result.argmax() == 1

    asyncio.run(run())


def test_safe_failure(model, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("secret text and /secret/path")

    monkeypatch.setattr(model._backend, "forward", fail)
    with pytest.raises(InferenceError) as error:
        model.encode(["0"])
    assert str(error.value) == "inference_failed"
    assert error.value.__suppress_context__


def test_cosine_lookup():
    docs = np.array([[1, 0], [0, 1], [1, 0]], np.float32)
    hits = cosine_search(np.array([2, 0], np.float32), docs, k=3)
    assert [h.index for h in hits] == [0, 2, 1]
    assert [h.score for h in hits] == [1, 1, 0]
    assert cosine_search(np.ones(2, np.float32), np.empty((0, 2), np.float32)) == []
    for query, documents, k in [
        (np.ones((1, 2)), docs, 1),
        (np.zeros(2), docs, 1),
        (np.ones(2), np.zeros((1, 2)), 1),
        (np.ones(2), docs, 0),
        (np.full(2, np.nan), docs, 1),
    ]:
        with pytest.raises(InvalidInputError):
            cosine_search(query, documents, k=k)
