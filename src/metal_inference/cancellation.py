"""Cooperative CPU cancellation; GPU commands finish before releasing buffers."""

from collections.abc import Callable

from .errors import CanceledError

CancelCheck = Callable[[], bool] | None


def checkpoint(canceled: CancelCheck) -> None:
    if canceled is not None and canceled():
        raise CanceledError()
