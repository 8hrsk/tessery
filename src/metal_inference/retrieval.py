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
    qnorm = np.linalg.norm(query)
    norms = np.linalg.norm(documents, axis=1)
    if qnorm == 0 or np.any(norms == 0):
        raise InvalidInputError()
    scores = (documents @ query) / (norms * qnorm)
    order = np.argsort(-scores, kind="stable")[:k]
    return [SearchHit(int(i), float(scores[i])) for i in order]
