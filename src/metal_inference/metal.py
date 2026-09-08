"""Direct C ABI to our Objective-C++ runtime and our Metal kernels.

The framework has no dependency on MLX, torch or MPS. Float32 is used for compute;
BF16 is a weight storage format, decoded by our kernels into float32 registers.
"""

import ctypes as ct
import platform
import struct
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from weakref import WeakSet

import numpy as np
from numpy.typing import NDArray

from .errors import ClosedError, InferenceError, MetalUnavailableError, NativeBuildError
from .tensor import Tensor


def _library() -> Any:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise MetalUnavailableError()
    root = Path(__file__).parent
    candidates = list(root.glob("_native*.so")) + list(root.glob("_native.dylib"))
    if len(candidates) != 1:
        raise NativeBuildError()
    try:
        lib = ct.CDLL(str(candidates[0]))
    except OSError:
        raise NativeBuildError() from None
    signatures = {
        "mi_create": ([ct.c_char_p], ct.c_void_p),
        "mi_destroy": ([ct.c_void_p], None),
        "mi_buffer": ([ct.c_void_p, ct.c_void_p, ct.c_uint64], ct.c_void_p),
        "mi_free": ([ct.c_void_p], None),
        "mi_read": ([ct.c_void_p, ct.c_void_p, ct.c_uint64], ct.c_int),
        "mi_begin": ([ct.c_void_p], ct.c_int),
        "mi_finish": ([ct.c_void_p], ct.c_int),
        "mi_abort": ([ct.c_void_p], None),
        "mi_dispatch": (
            [
                ct.c_void_p,
                ct.c_char_p,
                ct.POINTER(ct.c_void_p),
                ct.c_uint32,
                ct.c_void_p,
                ct.c_uint32,
                ct.c_uint64,
                ct.c_uint32,
            ],
            ct.c_int,
        ),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib


class Buffer:
    """Owned GPU buffer. Only the runtime creates buffers with checked sizes."""

    def __init__(self, runtime: "MetalRuntime", size: int, data: NDArray[Any] | None) -> None:
        self.runtime = runtime
        self.size = size
        self.pointer: int | None = runtime._lib.mi_buffer(
            runtime._pointer, None if data is None else data.ctypes.data, size
        )
        if not self.pointer:
            raise InferenceError()
        runtime.active_bytes += size
        runtime.peak_bytes = max(runtime.peak_bytes, runtime.active_bytes)
        runtime._buffers.add(self)

    def close(self) -> None:
        with self.runtime._lock:
            if getattr(self, "pointer", None):
                self.runtime._lib.mi_free(self.pointer)
                self.pointer = None
                self.runtime.active_bytes -= self.size

    def __del__(self) -> None:
        self.close()


class MetalRuntime:
    """Synchronous GPU command owner. Commands are serialized per instance."""

    def __init__(self) -> None:
        self._lib = _library()
        source = (Path(__file__).parent / "native/kernels.metal").read_bytes()
        self._pointer: int | None = self._lib.mi_create(source)
        if not self._pointer:
            raise MetalUnavailableError()
        self._lock = threading.RLock()
        self._recording = False
        self._inflight: list[Buffer] = []
        self._buffers: WeakSet[Buffer] = WeakSet()
        self.active_bytes = 0
        self.peak_bytes = 0

    def tensor(self, data: NDArray[np.float32]) -> Tensor:
        """Upload a nonempty float32 array into an owned, contiguous Metal tensor."""
        with self._lock:
            if not self._pointer:
                raise ClosedError()
            if (
                self._recording
                or not isinstance(data, np.ndarray)
                or data.dtype != np.float32
                or not data.size
            ):
                raise InferenceError()
            return Tensor(self.buffer(data.nbytes, np.ascontiguousarray(data)), data.shape)

    def buffer(self, size: int, data: NDArray[Any] | None = None) -> Buffer:
        with self._lock:
            if not self._pointer:
                raise ClosedError()
            if type(size) is not int or not 0 < size <= 2**31:
                raise InferenceError()
            if data is not None and (not data.flags.c_contiguous or data.nbytes != size):
                raise InferenceError()
            return Buffer(self, size, data)

    @contextmanager
    def command(self) -> Iterator[None]:
        with self._lock:
            if not self._pointer:
                raise ClosedError()
            if self._recording or self._lib.mi_begin(self._pointer):
                raise InferenceError()
            self._recording = True
            try:
                yield
                if self._lib.mi_finish(self._pointer):
                    raise InferenceError()
            finally:
                self._lib.mi_abort(self._pointer)
                self._recording = False
                self._inflight.clear()

    def _dispatch(
        self,
        name: str,
        buffers: Sequence[Buffer],
        *,
        threads: int,
        group_size: int = 256,
        n: int = 0,
        rows: int = 0,
        cols: int = 0,
        k: int = 0,
        heads: int = 0,
        kv_heads: int = 0,
        seq: int = 0,
        batch: int = 0,
        dim: int = 0,
        group: int = 64,
        eps: float = 1e-6,
        theta: float = 1e6,
        scale: float = 1.0,
    ) -> None:
        if not self._recording or any(not b.pointer or b.runtime is not self for b in buffers):
            raise InferenceError()
        parameters = struct.pack(
            "<12I4f",
            n,
            rows,
            cols,
            k,
            heads,
            kv_heads,
            seq,
            batch,
            dim,
            group,
            0,
            0,
            eps,
            theta,
            scale,
            0.0,
        )
        pointers = (ct.c_void_p * len(buffers))(*(b.pointer for b in buffers))
        self._inflight.extend(buffers)
        if self._lib.mi_dispatch(
            self._pointer,
            name.encode("ascii"),
            pointers,
            len(buffers),
            parameters,
            len(parameters),
            threads,
            group_size,
        ):
            raise InferenceError()

    def read(self, buffer: Buffer, shape: tuple[int, ...]) -> NDArray[np.float32]:
        with self._lock:
            if not self._pointer:
                raise ClosedError()
            if self._recording or buffer.runtime is not self or not buffer.pointer:
                raise InferenceError()
            size = int(np.prod(shape)) * 4
            if size != buffer.size:
                raise InferenceError()
            result = np.empty(shape, dtype=np.float32)
            if self._lib.mi_read(buffer.pointer, result.ctypes.data, result.nbytes):
                raise InferenceError()
            return result

    def add(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reusable elementwise GPU addition, independent of any model."""
        if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32 or not a.size:
            raise InferenceError()
        with self._lock:
            buffers: list[Buffer] = []
            try:
                for data in (a, b):
                    buffers.append(self.buffer(data.nbytes, np.ascontiguousarray(data)))
                buffers.append(self.buffer(a.nbytes))
                with self.command():
                    self._dispatch("add", buffers, threads=a.size, n=a.size)
                return self.read(buffers[-1], a.shape)
            finally:
                for buffer in buffers:
                    buffer.close()

    def matmul(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
        """General float32 [M,K] @ [K,N] on this engine's Metal kernel."""
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[0]
            or a.dtype != np.float32
            or b.dtype != np.float32
            or not a.size
            or not b.size
        ):
            raise InferenceError()
        m, k = a.shape
        n = b.shape[1]
        with self._lock:
            buffers: list[Buffer] = []
            try:
                for data in (a, b.T):
                    buffers.append(self.buffer(data.nbytes, np.ascontiguousarray(data)))
                buffers.append(self.buffer(m * n * 4))
                with self.command():
                    self._dispatch(
                        "matmul_f32",
                        buffers,
                        threads=((m + 3) // 4) * n * 32,
                        group_size=32,
                        rows=m,
                        cols=n,
                        k=k,
                    )
                return self.read(buffers[-1], (m, n))
            finally:
                for buffer in buffers:
                    buffer.close()

    def close(self) -> None:
        with self._lock:
            if self._recording:
                raise InferenceError()
            if self._pointer:
                for buffer in list(self._buffers):
                    buffer.close()
                self._lib.mi_destroy(self._pointer)
                self._pointer = None

    def __enter__(self) -> "MetalRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_pointer", None) and hasattr(self, "_lock"):
            self.close()
