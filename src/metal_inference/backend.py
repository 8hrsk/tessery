"""Model adapter contracts and bounded configuration parsing."""

import math
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .cancellation import CancelCheck
from .errors import UnsupportedProfileError
from .metal import MetalRuntime


class Tokenizer(Protocol):
    vocab_size: int
    pad_id: int

    def batch(
        self, texts: list[str], *, max_length: int, canceled: CancelCheck = None
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]: ...


class Backend(Protocol):
    runtime: MetalRuntime
    max_padded_tokens: int
    vocab_size: int

    def forward(
        self, ids: NDArray[np.uint32], lengths: NDArray[np.uint32], *, dimensions: int
    ) -> NDArray[np.float32]: ...

    def close(self) -> None: ...


def config_int(config: dict[str, Any], key: str, low: int, high: int) -> int:
    value = config.get(key)
    if type(value) is not int or not low <= value <= high:
        raise UnsupportedProfileError()
    return int(value)


def config_float(config: dict[str, Any], key: str, low: float, high: float) -> float:
    value = config.get(key)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not low <= value <= high
        or not math.isfinite(value)
    ):
        raise UnsupportedProfileError()
    return float(value)
