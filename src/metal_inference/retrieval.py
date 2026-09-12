"""Small in-memory cosine lookup for already normalized embeddings; no database."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import InvalidInputError


@dataclass(frozen=True)
class SearchHit:
    index: int
    score: float


def cosine_search(
    query: NDArray[np.float32],
    documents: NDArray[np.float32],
    *,
    k: int = 5,
) -> list[SearchHit]:
    """Rank an existing matrix by cosine; ties use the original document index."""
    if (
        query.ndim != 1
        or documents.ndim != 2
        or documents.shape[1] != query.shape[0]
        or type(k) is not int
        or k < 1
        or not np.isfinite(query).all()
        or not np.isfinite(documents).all()
    ):
        raise InvalidInputError()
    return _search_with_norms(query, documents, None, k=k)


def _search_with_norms(
    query: NDArray[np.float32],
    documents: NDArray[np.float32],
    norms: NDArray[np.float32] | None,
    *,
    k: int,
) -> list[SearchHit]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        qnorm = np.linalg.norm(query)
        if norms is None:
            norms = np.linalg.norm(documents, axis=1)
        scores = (documents @ query) / (norms * qnorm)
    # Keep the usual normalized-embedding path unchanged. Rescale extreme finite
    # inputs before any squaring or dot product, avoiding overflow and underflow.
    floor = np.sqrt(np.finfo(np.float32).tiny)
    if (
        not np.isfinite(qnorm)
        or not np.isfinite(norms).all()
        or qnorm < floor
        or np.any(norms < floor)
        or not np.isfinite(scores).all()
    ):
        q = np.array(query, dtype=np.float64, copy=True)
        docs = np.array(documents, dtype=np.float64, copy=True)
        qscale = np.max(np.abs(q), initial=0)
        scales = np.max(np.abs(docs), axis=1, initial=0)
        if qscale == 0 or np.any(scales == 0):
            raise InvalidInputError()
        q /= qscale
        docs /= scales[:, None]
        q /= np.linalg.norm(q)
        docs /= np.linalg.norm(docs, axis=1)[:, None]
        scores = docs @ q
        if not np.isfinite(scores).all():
            raise InvalidInputError()
        scores = np.clip(scores, -1, 1)
    order = _stable_top_k(scores, k)
    return [SearchHit(int(i), float(scores[i])) for i in order]


def _stable_top_k(scores: NDArray[np.floating], k: int) -> NDArray[np.intp]:
    """Partition large, narrow selections while retaining original-index ties."""
    count = scores.size
    if count < 128 or k * 4 >= count:
        return np.argsort(-scores, kind="stable")[:k]
    cutoff = np.partition(scores, count - k)[count - k]
    higher = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[: k - higher.size]
    selected = np.concatenate((higher, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def _prepare_document_norms(documents: NDArray[np.float32]) -> NDArray[np.float32] | None:
    """Cache only immutable, finite matrices; unusual inputs use public validation."""
    if documents.ndim != 2 or documents.flags.writeable or not np.isfinite(documents).all():
        return None
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(documents, axis=1)
    norms.flags.writeable = False
    return norms


def _cached_cosine_search(
    query: NDArray[np.float32],
    documents: NDArray[np.float32],
    norms: NDArray[np.float32] | None,
    *,
    k: int,
) -> list[SearchHit]:
    """Internal path for the index's private immutable snapshot and its cached norms."""
    if norms is None:
        return cosine_search(query, documents, k=k)
    if (
        query.ndim != 1
        or query.shape[0] != documents.shape[1]
        or type(k) is not int
        or k < 1
        or not np.isfinite(query).all()
    ):
        raise InvalidInputError()
    return _search_with_norms(query, documents, norms, k=k)
