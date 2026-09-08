"""Bounded length buckets with stable ordering within equal lengths."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def length_batches(
    lengths: NDArray[np.uint32], max_padded_tokens: int
) -> Iterator[tuple[NDArray[np.intp], int]]:
    """Yield original row indices and each bucket's padded sequence width.

    A bucket never pads its shortest input to more than twice its length and
    never exceeds the backend token budget. Caller validates nonzero lengths
    not exceeding the budget; at most 32 inputs are admitted by EmbeddingModel.
    """
    order = np.argsort(lengths, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        shortest = int(lengths[order[start]])
        width = shortest
        while end < len(order):
            candidate = int(lengths[order[end]])
            if candidate > shortest * 2 or candidate * (end - start + 1) > max_padded_tokens:
                break
            width = candidate
            end += 1
        yield order[start:end], width
        start = end
