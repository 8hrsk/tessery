import numpy as np
import pytest

from metal_inference.errors import InvalidInputError
from metal_inference.index import Chunk, DocumentIndex, _contract
from metal_inference.retrieval import _stable_top_k, cosine_search
from tests.unit.test_index import Model


@pytest.mark.parametrize("count", [0, 10, 127, 128, 129, 1000, 10000])
@pytest.mark.parametrize("k", [1, 5, 31, 100, 10001])
def test_partial_selection_matches_stable_sort_with_boundary_ties(count, k):
    rng = np.random.default_rng(109)
    for scores in [
        rng.integers(-3, 4, count).astype(np.float32),
        rng.standard_normal(count).astype(np.float32),
        np.zeros(count, np.float32),
        np.full(count, -1, np.float64),
    ]:
        # Signed zero compares equal and must still use original document order.
        scores[::7] = -0.0
        np.testing.assert_array_equal(
            _stable_top_k(scores, k), np.argsort(-scores, kind="stable")[:k]
        )


@pytest.mark.parametrize("scale", [1.0, 3e38, 1e-40])
@pytest.mark.parametrize("k", [1, 5, 100])
def test_index_cached_search_is_exact_with_generic_finite_inputs(scale, k):
    rng = np.random.default_rng(47)
    documents = rng.choice([-1, 0, 1], size=(256, 32)).astype(np.float32) * scale
    query = np.array([scale] * 32, np.float32)
    model = Model()
    model.encode = lambda texts: query[None, :]
    chunks = tuple(Chunk(str(i), 0, 1, "x") for i in range(len(documents)))
    with np.errstate(all="raise"):
        index = DocumentIndex({"contract": _contract(model), "query_prefix": ""}, chunks, documents)
        expected = cosine_search(query, documents, k=k)
        actual = index.search(model, "query", k=k)
    assert [(int(h.chunk.source), h.score) for h in actual] == [
        (h.index, h.score) for h in expected
    ]
    # Snapshot ownership: callers cannot invalidate the cache via their input array.
    documents.fill(0)
    assert index.search(model, "query", k=k) == actual
    assert index._document_norms.nbytes == len(documents) * 4
    assert not index._document_norms.flags.writeable


@pytest.mark.parametrize("invalid", [0.0, np.inf, np.nan])
def test_cached_index_retains_invalid_vector_rejection(invalid):
    model = Model()
    documents = np.ones((256, 32), np.float32)
    documents[100] = invalid
    chunks = tuple(Chunk(str(i), 0, 1, "x") for i in range(len(documents)))
    index = DocumentIndex({"contract": _contract(model), "query_prefix": ""}, chunks, documents)
    with pytest.raises(InvalidInputError):
        index.search(model, "query")


@pytest.mark.parametrize("query", [np.zeros(32), np.ones(31), np.full(32, np.nan)])
def test_cached_index_retains_invalid_query_rejection(query):
    model = Model()
    index = DocumentIndex.build(model, {"doc": "Paris"})
    model.encode = lambda texts: query[None, :]
    with pytest.raises(InvalidInputError):
        index.search(model, "query")


def test_cached_index_snapshot_roundtrip_matches_scores_and_ties(tmp_path):
    model = Model()
    documents = np.zeros((256, 32), np.float32)
    documents[:30, 0] = 1
    documents[30:, 1] = 1
    metadata = {
        "format": "tessery-exact-index-v1",
        "contract": _contract(model),
        "query_prefix": "",
        "document_prefix": "",
        "chunk_chars": 1,
        "overlap_chars": 0,
    }
    chunks = tuple(Chunk(str(i), 0, 1, "x") for i in range(len(documents)))
    index = DocumentIndex(metadata, chunks, documents)
    path = tmp_path / "index.sqlite"
    index.save(path)
    loaded = DocumentIndex.load(path)
    for k in [1, 5, 29, 30, 31, 100]:
        assert loaded.search(model, "Paris", k=k) == index.search(model, "Paris", k=k)
