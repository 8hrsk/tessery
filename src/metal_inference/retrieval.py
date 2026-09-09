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
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        qnorm = np.linalg.norm(query)
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
    order = np.argsort(-scores, kind="stable")[:k]
    return [SearchHit(int(i), float(scores[i])) for i in order]
