"""Eager float32 tensors backed by owned Metal memory, without host intermediates."""

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .errors import ClosedError, InferenceError

if TYPE_CHECKING:
    from .metal import Buffer


class Tensor:
    """Create with MetalRuntime.tensor(); operations return independent allocations.

    Commands finish before returning, but data stays on Metal until numpy().
    There is no broadcasting, implicit dtype conversion or automatic differentiation.
    """

    def __init__(self, buffer: "Buffer", shape: tuple[int, ...]) -> None:
        if (
            not isinstance(shape, tuple)
            or any(type(d) is not int or d <= 0 for d in shape)
            or math.prod(shape) * 4 != buffer.size
        ):
            raise InferenceError()
        self._buffer = buffer
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[np.float32]:
        return np.dtype(np.float32)

    @property
    def nbytes(self) -> int:
        return self._buffer.size

    def _check(self, other: "Tensor | None" = None) -> None:
        runtime = self._buffer.runtime
        if not runtime._pointer or not self._buffer.pointer:
            raise ClosedError()
        if runtime._recording:
            raise InferenceError()
        if other is not None:
            if not isinstance(other, Tensor) or other._buffer.runtime is not runtime:
                raise InferenceError()
            other._check()

    def numpy(self) -> NDArray[np.float32]:
        """Copy into an independent, writable host array."""
        runtime = self._buffer.runtime
        with runtime._lock:
            self._check()
            return runtime.read(self._buffer, self.shape)

    def _unary(self, kernel: str, shape: tuple[int, ...], **params: int) -> "Tensor":
        runtime = self._buffer.runtime
        with runtime._lock:
            self._check()
            output = runtime.buffer(self.nbytes)
            try:
                with runtime.command():
                    runtime._dispatch(
                        kernel, [self._buffer, output], threads=self.nbytes // 4, **params
                    )
                return Tensor(output, shape)
            except BaseException:
                output.close()
                raise

    def transpose(self) -> "Tensor":
        """Transpose a matrix on Metal into a new contiguous allocation."""
        if len(self.shape) != 2:
            raise InferenceError()
        rows, cols = self.shape
        return self._unary("transpose_f32", (cols, rows), rows=rows, cols=cols)

    def silu(self) -> "Tensor":
        """Elementwise x * sigmoid(x), preserving the input."""
        return self._unary("silu_f32", self.shape, n=self.nbytes // 4)

    def __add__(self, other: "Tensor") -> "Tensor":
        runtime = self._buffer.runtime
        with runtime._lock:
            if not isinstance(other, Tensor):
                raise InferenceError()
            self._check(other)
            if self.shape != other.shape:
                raise InferenceError()
            output = runtime.buffer(self.nbytes)
            try:
                with runtime.command():
                    runtime._dispatch(
                        "add",
                        [self._buffer, other._buffer, output],
                        threads=self.nbytes // 4,
                        n=self.nbytes // 4,
                    )
                return Tensor(output, self.shape)
            except BaseException:
                output.close()
                raise

    def __matmul__(self, other: "Tensor") -> "Tensor":
        runtime = self._buffer.runtime
        with runtime._lock:
            if not isinstance(other, Tensor):
                raise InferenceError()
            self._check(other)
            if len(self.shape) != 2 or len(other.shape) != 2 or self.shape[1] != other.shape[0]:
                raise InferenceError()
            m, k = self.shape
            n = other.shape[1]
            output = runtime.buffer(m * n * 4)
            transposed = None
            try:
                transposed = runtime.buffer(other.nbytes)
                with runtime.command():
                    runtime._dispatch(
                        "transpose_f32",
                        [other._buffer, transposed],
                        threads=k * n,
                        rows=k,
                        cols=n,
                    )
                    runtime._matmul_f32(
                        [self._buffer, transposed, output],
                        rows=m,
                        cols=n,
                        k=k,
                    )
                return Tensor(output, (m, n))
            except BaseException:
                output.close()
                raise
            finally:
                if transposed is not None:
                    transposed.close()

    def close(self) -> None:
        """Free the allocation; safe to repeat, including after runtime.close()."""
        self._buffer.close()

    def __enter__(self) -> "Tensor":
        with self._buffer.runtime._lock:
            self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
