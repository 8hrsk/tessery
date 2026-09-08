import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from metal_inference import DocumentIndex, ModelDescriptor, read_documents
from metal_inference.errors import InvalidInputError, ManifestError


class Model:
    descriptor = ModelDescriptor()
    dimensions = 32
    max_length = 64
    _tokenizer = SimpleNamespace(
        batch=lambda texts, max_length: (
            None,
            np.array([min(len(t) + 1, max_length) for t in texts]),
        )
    )

    def encode(self, texts):
        vectors = np.zeros((len(texts), self.dimensions), np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0 if "Paris" in text else 1] = 1
        return vectors


@pytest.fixture
def model():
    return Model()


def test_snapshot_round_trip_search_order_and_contract(tmp_path, model):
    index = DocumentIndex.build(
        model, {"fr.txt": "Paris is in France.", "food.txt": "Bananas are yellow."}
    )
    target = tmp_path / "search Ω # ?.sqlite"
    index.save(target)
    original = target.read_bytes()
    loaded = DocumentIndex.load(target)
    assert loaded.chunks == index.chunks
    hit = loaded.search(model, "Paris", k=1)[0]
    assert hit.chunk.source == "fr.txt" and hit.score == 1
    with pytest.raises(FileExistsError):
        index.save(target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
    model.max_length = 32
    with pytest.raises(ManifestError):
        loaded.search(model, "Paris")
    model.max_length = 64
    model.descriptor = replace(model.descriptor, revision="different")
    with pytest.raises(ManifestError):
        loaded.search(model, "Paris")


def test_chunks_cover_original_and_do_not_saturate_tokenizer(model):
    text = "abc Paris xyz " * 70
    index = DocumentIndex.build(
        model, {"doc": text}, chunk_chars=90, overlap_chars=20, document_prefix="doc:"
    )
    covered = set()
    for c in index.chunks:
        assert c.text == text[c.start : c.end]
        assert len("doc:" + c.text) + 1 < model.max_length
        covered.update(range(c.start, c.end))
    assert covered == set(range(len(text)))
    assert index.chunks[-1].end == len(text)
    model.max_length = 1
    with pytest.raises(InvalidInputError):
        DocumentIndex.build(model, {"doc": "x"})


@pytest.mark.parametrize(
    "documents,kwargs",
    [
        ({}, {}),
        ({"": "x"}, {}),
        ({"doc": " "}, {}),
        ({"doc": "\ud800"}, {}),
        ({"doc": "x"}, {"overlap_chars": 600}),
        ({"doc": "x"}, {"chunk_chars": True}),
    ],
)
def test_invalid_build(model, documents, kwargs):
    with pytest.raises(InvalidInputError):
        DocumentIndex.build(model, documents, **kwargs)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE chunks SET text='corrupt'",
        "UPDATE chunks SET vector=zeroblob(128)",
        "DELETE FROM chunks",
        "UPDATE metadata SET json='{}'",
        "INSERT INTO metadata VALUES ('{}')",
    ],
)
def test_corrupt_snapshot_is_rejected(tmp_path, model, sql):
    target = tmp_path / "index.sqlite"
    DocumentIndex.build(model, {"doc": "Paris"}).save(target)
    with sqlite3.connect(target) as db:
        db.execute(sql)
    with pytest.raises(ManifestError):
        DocumentIndex.load(target)


def test_read_documents_and_bad_query(tmp_path, model):
    (tmp_path / "b.txt").write_text("Paris")
    (tmp_path / "a.md").write_text("Garden")
    (tmp_path / "skip.py").write_text("x")
    (tmp_path / "link.txt").symlink_to(tmp_path / "b.txt")
    assert list(read_documents(tmp_path)) == ["a.md", "b.txt"]
    index = DocumentIndex.build(model, read_documents(tmp_path))
    for query, k in [("", 1), ("Paris", 0), ("Paris", True)]:
        with pytest.raises(InvalidInputError):
            index.search(model, query, k=k)
    (tmp_path / "b.txt").write_bytes(b"\xff")
    with pytest.raises(InvalidInputError):
        read_documents(tmp_path)


def test_index_payload_bound_precedes_embedding(model, monkeypatch):
    import metal_inference.index as module

    monkeypatch.setattr(module, "MAX_INDEX_BYTES", 100)
    monkeypatch.setattr(
        model, "encode", lambda texts: pytest.fail("oversized index reached inference")
    )
    with pytest.raises(InvalidInputError):
        DocumentIndex.build(model, {"doc": "Paris"})
