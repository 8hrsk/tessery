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


def execution_batches(
    lengths: NDArray[np.uint32], max_padded_tokens: int, max_length: int, architecture: str
) -> Iterator[tuple[NDArray[np.intp], int]]:
    """Select measured matrix/attention alignment without changing token lengths.

    Eight-row projection tiles use batch * width rows. Do not pad an already
    aligned matrix unless the new width also enables tiled attention. Qwen's
    small projection tail becomes negligible at longer lengths, so only align
    those when the next eight-token boundary is also an attention boundary.
    Keep the original plan when any padding, context or token budget is exceeded.
    If eight-token padding is blocked, a short two-input Qwen bucket can still
    align its matrix with four-token padding. Larger buckets retain the old
    policy: their extra padded rows can cost more than the projection tail.
    """
    for rows, width in length_batches(lengths, max_padded_tokens):
        aligned = (width + 7) // 8 * 8
        attention = aligned >= 64 and aligned % 32 == 0
        projection = width * len(rows) % 8 != 0 and (architecture == "bert_f32" or width < 128)
        if (
            architecture in {"qwen3_uint4", "bert_f32"}
            and width >= 5
            and (attention or projection)
            and aligned <= max_length
            and aligned <= 2 * int(lengths[rows].min())
            and aligned * len(rows) <= max_padded_tokens
        ):
            width = aligned
        elif architecture == "qwen3_uint4" and len(rows) == 2 and 5 <= width < 128 and projection:
            aligned = (width + 3) // 4 * 4
            if (
                aligned <= max_length
                and aligned <= 2 * int(lengths[rows].min())
                and aligned * len(rows) <= max_padded_tokens
            ):
                width = aligned
        yield rows, width
