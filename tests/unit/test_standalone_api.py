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
    def batch(self, texts, *, max_length, canceled=None):
        return np.array([[int(t)] for t in texts], dtype=np.uint32), np.ones(len(texts), np.uint32)


@pytest.mark.parametrize("scale", [3e38, 1e-40])
def test_cosine_search_extreme_finite_vectors(scale):
    query = np.array([scale, scale], np.float32)
    documents = np.array([[scale, scale], [-scale, -scale], [scale, -scale]], np.float32)
    with np.errstate(all="raise"):
        hits = cosine_search(query, documents)
    assert [h.index for h in hits] == [0, 2, 1]
    np.testing.assert_allclose([h.score for h in hits], [1, 0, -1], atol=1e-15)


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


def test_length_buckets_restore_original_output_order(model):
    class VariableTokenizer:
        def batch(self, texts, *, max_length, canceled=None):
            lengths = np.array([3, 8, 2, 8, 2], np.uint32)
            ids = np.zeros((5, 8), np.uint32)
            ids[:, 0] = [int(t) for t in texts]
            return ids, lengths

    model._tokenizer = VariableTokenizer()
    model._backend.max_padded_tokens = 16
    shapes = []
    forward = model._backend.forward

    def record(ids, lengths, **kw):
        shapes.append(ids.shape)
        assert ids.flags.c_contiguous
        return forward(ids, lengths, **kw)

    model._backend.forward = record
    output = model.encode(["3", "1", "2", "4", "5"])
    assert output.argmax(axis=1).tolist() == [3, 1, 2, 4, 5]
    assert shapes == [(3, 3), (2, 8)]


@pytest.mark.parametrize(
    "limit, budget, expected_width", [(512, 4096, 8), (7, 4096, 7), (512, 7, 7)]
)
def test_execution_padding_keeps_real_lengths_and_uses_tokenizer_pad_id(
    model, limit, budget, expected_width
):
    class PaddingTokenizer:
        pad_id = 99

        def batch(self, texts, *, max_length, canceled=None):
            assert max_length == limit
            return np.array([[3, 4, 5, 6, 7, 8, 9]], np.uint32), np.array([7], np.uint32)

    model._tokenizer = PaddingTokenizer()
    model.max_length = limit
    model._backend.max_padded_tokens = budget
    original = model._backend.forward

    def forward(ids, lengths, **kw):
        assert ids.shape == (1, expected_width)
        assert ids.dtype == np.uint32 and ids.flags.c_contiguous
        assert lengths.tolist() == [7]
        assert ids[0, :7].tolist() == [3, 4, 5, 6, 7, 8, 9]
        if expected_width == 8:
            assert ids[0, 7] == 99
        return original(ids, lengths, **kw)

    model._backend.forward = forward
    assert model.encode(["3"]).argmax(axis=1).tolist() == [3]


def test_sync_does_not_bypass_admitted_async(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    backend = Backend()
    backend.release = threading.Event()
    model = EmbeddingModel(backend, Tokenizer(), 384, 512, 4)
    sync_submitted = threading.Event()
    submit = model._executor.submit

    def observed_submit(function, snapshot, *args):
        result = submit(function, snapshot, *args)
        if snapshot == ["2"]:
            sync_submitted.set()
        return result

    monkeypatch.setattr(model._executor, "submit", observed_submit)

    async def run():
        with ThreadPoolExecutor(max_workers=1) as callers:
            first = asyncio.create_task(model.encode_async(["0"]))
            assert await asyncio.to_thread(backend.entered.wait, 2)
            queued = asyncio.create_task(model.encode_async(["1"]))
            await asyncio.sleep(0)
            sync = callers.submit(model.encode, ["2"])
            assert await asyncio.to_thread(sync_submitted.wait, 2)
            backend.release.set()
            await asyncio.gather(first, queued)
            assert (await asyncio.wrap_future(sync)).argmax() == 2
            assert backend.calls == [[0], [1], [2]]

    try:
        asyncio.run(run())
    finally:
        backend.release.set()
        model.close()


def test_close_cancels_queued_sync_as_closed(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    backend = Backend()
    backend.release = threading.Event()
    model = EmbeddingModel(backend, Tokenizer(), 384, 512, 4)
    admitted = threading.Event()
    shutdown = threading.Event()
    submit = model._executor.submit
    native_shutdown = model._executor.shutdown

    def observed_submit(function, snapshot, *args):
        result = submit(function, snapshot, *args)
        if snapshot == ["1"]:
            admitted.set()
        return result

    def observed_shutdown(**kw):
        shutdown.set()
        native_shutdown(**kw)

    monkeypatch.setattr(model._executor, "submit", observed_submit)
    monkeypatch.setattr(model._executor, "shutdown", observed_shutdown)
    with ThreadPoolExecutor(max_workers=3) as callers:
        try:
            first = callers.submit(model.encode, ["0"])
            assert backend.entered.wait(2)
            queued = callers.submit(model.encode, ["1"])
            assert admitted.wait(2)
            closing = callers.submit(model.close)
            assert shutdown.wait(2)
            with pytest.raises(ClosedError):
                queued.result(timeout=2)
            backend.release.set()
            first.result(timeout=2)
            closing.result(timeout=2)
            assert backend.calls == [[0]]
        finally:
            backend.release.set()
            model.close()


def test_async_cancellation_during_tokenization_releases_worker(model):
    from metal_inference.cancellation import checkpoint

    entered, release = threading.Event(), threading.Event()
    original = model._tokenizer.batch

    def batch(texts, *, max_length, canceled=None):
        if texts == ["1"]:
            entered.set()
            assert release.wait(5)
            checkpoint(canceled)
            pytest.fail("Canceled tokenization reached encoding")
        return original(texts, max_length=max_length, canceled=canceled)

    model._tokenizer.batch = batch

    async def exercise():
        job = asyncio.create_task(model.encode_async(["1"]))
        assert await asyncio.to_thread(entered.wait, 5)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        release.set()
        result = await model.encode_async(["2"])
        assert result.argmax(axis=1).tolist() == [2]
        assert model._backend.calls == [[2]]

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_stress_harness_queue_overload_and_recovery(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    stress = importlib.import_module("stress_embeddings")
    with EmbeddingModel(Backend(), Tokenizer(), 384, 512, 4) as model:
        result = asyncio.run(stress.queue_stress(model, "1"))
        assert result == {"canceled": 2, "overloaded": 2, "completed": 2}
        assert model.encode(["2"]).argmax(axis=1).tolist() == [2]
