import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from metal_inference.api import EmbeddingModel
from metal_inference.cancellation import checkpoint
from metal_inference.errors import CanceledError, ClosedError, InvalidInputError, OverloadError
from metal_inference.index import DocumentIndex


class CountingTokenizer:
    pad_id = 0
    vocab_size = 128

    def __init__(self):
        self.text_count = 0
        self.calls = []

    def batch(self, texts, *, max_length, canceled=None):
        checkpoint(canceled)
        self.calls.append(list(texts))
        self.text_count += len(texts)
        rows = [[1, *[ord(c) % 126 + 2 for c in text]][:max_length] for text in texts]
        ids = np.zeros((len(rows), max(map(len, rows))), np.uint32)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
        return ids, np.asarray([len(row) for row in rows], np.uint32)


class Backend:
    max_padded_tokens = 4096
    vocab_size = 128
    runtime = SimpleNamespace(active_bytes=0, peak_bytes=0)

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def forward(self, ids, lengths, *, dimensions):
        self.entered.set()
        assert self.release.wait(5)
        self.calls.append((ids.copy(), lengths.copy()))
        output = np.zeros((len(ids), dimensions), np.float32)
        for i, length in enumerate(lengths):
            output[i, int(np.sum(ids[i, :length])) % dimensions] = 1
        return output

    def close(self):
        pass


@pytest.fixture
def model():
    backend = Backend()
    model = EmbeddingModel(backend, CountingTokenizer(), 384, 64, 2)
    yield model
    backend.release.set()
    model.close()


def build(model, reuse):
    model._reuse_index_tokens = reuse
    return DocumentIndex.build(
        model,
        {"one": "Paris hello 世界 " * 60, "two": "Other document content " * 40},
        chunk_chars=70,
        overlap_chars=6,
        document_prefix="doc:",
        query_prefix="query:",
    )


def test_index_reuses_accepted_rows_exactly_with_prefix_and_truncation(model, tmp_path):
    original = build(model, False)
    old_calls = model._tokenizer.text_count
    old_forwards = list(model._backend.calls)
    model._backend.calls.clear()
    model._tokenizer.text_count = 0
    actual = build(model, True)
    assert len(actual.chunks) > 32
    assert model._tokenizer.text_count == old_calls - len(actual.chunks)
    assert actual.chunks == original.chunks
    assert len(old_forwards) == len(model._backend.calls)
    for expected, observed in zip(old_forwards, model._backend.calls, strict=True):
        np.testing.assert_array_equal(expected[0], observed[0])
        np.testing.assert_array_equal(expected[1], observed[1])
    np.testing.assert_array_equal(actual._vectors, original._vectors)
    assert actual._metadata == original._metadata
    target = tmp_path / "prepared.sqlite"
    actual.save(target)
    loaded = DocumentIndex.load(target)
    assert loaded.search(model, "Paris") == actual.search(model, "Paris")
    assert model._tokenizer.calls[-1] == ["query:Paris"]
    assert all(len(ids) <= 32 for ids, _ in model._backend.calls)


@pytest.mark.parametrize("budget", [0, 128])
def test_token_cache_cap_falls_back_without_changing_chunks_or_vectors(model, monkeypatch, budget):
    import metal_inference.index as module

    original = build(model, False)
    monkeypatch.setattr(module, "_MAX_PREPARED_TOKEN_BYTES", budget)
    actual = build(model, True)
    assert actual.chunks == original.chunks
    np.testing.assert_array_equal(actual._vectors, original._vectors)


@pytest.mark.parametrize(
    "texts,rows,cap",
    [
        ([""], [np.ones(1, np.uint32)], 64),
        (["valid"] * 33, [np.ones(1, np.uint32)] * 33, 64),
        (["valid"], [], 64),
        (["valid"], [np.ones(3, np.float32)], 64),
        (["valid"], [np.ones((1, 3), np.uint32)], 64),
        (["valid"], [np.zeros(0, np.uint32)], 64),
        (["valid"], [np.ones(65, np.uint32)], 64),
        (["valid"], [np.array([128], np.uint32)], 64),
        (["valid"], [np.ones(1, np.uint32)], True),
    ],
)
def test_prepared_input_validates_text_and_token_limits(model, texts, rows, cap):
    with pytest.raises(InvalidInputError):
        model._encode_prepared(texts, rows, max_length=cap)
    assert not model._backend.calls


def test_prepared_input_retokenizes_after_changed_cap(model):
    texts = ["longish text"]
    ids, lengths = model._tokenizer.batch(texts, max_length=64)
    model.max_length = 4
    expected = model.encode(texts)
    calls = model._tokenizer.text_count
    actual = model._encode_prepared(texts, [ids[0, : lengths[0]]], max_length=64)
    np.testing.assert_array_equal(actual, expected)
    assert model._tokenizer.text_count == calls + 1


def test_prepared_input_snapshot_admission_fifo_and_close(model, monkeypatch):
    texts = ["first"]
    ids, lengths = model._tokenizer.batch(texts, max_length=64)
    rows = [ids[0, : lengths[0]]]
    expected = model.encode(texts)
    backend = model._backend
    backend.calls.clear()
    backend.entered.clear()
    backend.release.clear()
    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(model.encode, ["occupy"])
        assert backend.entered.wait(2)
        # The private input path participates in the same admission limit.
        model._admission.acquire()
        try:
            with pytest.raises(OverloadError):
                model._encode_prepared(texts, rows, max_length=64)
        finally:
            model._admission.release()
        queued = threading.Event()
        original_submit = model._executor.submit

        def submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            queued.set()
            return future

        monkeypatch.setattr(model._executor, "submit", submit)
        second = callers.submit(model._encode_prepared, texts, rows, max_length=64)
        assert queued.wait(2)
        rows[0].fill(0)
        texts[0] = "mutated after snapshot"
        backend.release.set()
        first.result(timeout=2)
        np.testing.assert_array_equal(second.result(timeout=2), expected)
        assert len(backend.calls) == 2
    model.close()
    with pytest.raises(ClosedError):
        model._encode_prepared(texts, rows, max_length=64)


def test_prepared_worker_cancellation_releases_admission(model):
    ids, lengths = model._tokenizer.batch(["cancel"], max_length=64)
    canceled = threading.Event()
    canceled.set()
    future = model._submit(["cancel"], 384, canceled, (ids, lengths, 64))
    with pytest.raises(CanceledError):
        future.result(timeout=2)
    assert not model._backend.calls
    model._encode_prepared(["cancel"], [ids[0]], max_length=64)
    assert len(model._backend.calls) == 1


def test_probe_cap_change_and_restore_disables_reuse(model, monkeypatch):
    original_batch = model._tokenizer.batch
    probes = 0

    def change_cap(texts, *, max_length, canceled=None):
        nonlocal probes
        probes += 1
        if probes == 1:
            model.max_length = 32
        elif probes == 2:
            model.max_length = 64
        return original_batch(texts, max_length=max_length, canceled=canceled)

    monkeypatch.setattr(model._tokenizer, "batch", change_cap)
    expected = build(model, False)
    probes = 0
    model.max_length = 64
    monkeypatch.setattr(
        model, "_encode_prepared", lambda *a, **kw: pytest.fail("mixed probe caps reused")
    )
    actual = build(model, True)
    assert actual.chunks == expected.chunks
    np.testing.assert_array_equal(actual._vectors, expected._vectors)
