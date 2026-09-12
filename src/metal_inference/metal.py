"""Direct C ABI to our Objective-C++ runtime and our Metal kernels.

The framework has no dependency on MLX, torch or MPS. Float32 is used for compute;
BF16 is a weight storage format, decoded by our kernels into float32 registers.
"""

import ctypes as ct
import hashlib
import platform
import struct
import threading
import time
from collections import Counter
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
        native_bytes = candidates[0].read_bytes()
        lib: Any = ct.CDLL(str(candidates[0]))
        if candidates[0].read_bytes() != native_bytes:
            raise NativeBuildError()
        lib._tessery_sha256 = hashlib.sha256(native_bytes).hexdigest()
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
        "mi_gpu_seconds": ([ct.c_void_p], ct.c_double),
        "mi_abort": ([ct.c_void_p], None),
        "mi_profile_enable": ([ct.c_void_p, ct.c_int], ct.c_int),
        "mi_profile_read": ([ct.c_void_p, ct.POINTER(ct.c_double), ct.c_uint32], ct.c_int),
        "mi_plan_create": ([ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_uint32], ct.c_void_p),
        "mi_plan_add": (
            [
                ct.c_void_p,
                ct.c_char_p,
                ct.POINTER(ct.c_uint32),
                ct.c_uint32,
                ct.c_void_p,
                ct.c_uint64,
                ct.c_uint32,
            ],
            ct.c_int,
        ),
        "mi_plan_bytes": ([ct.c_void_p], ct.c_uint64),
        "mi_plan_free": ([ct.c_void_p], None),
        "mi_plan_run": ([ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_uint32], ct.c_int),
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
    try:
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
    except AttributeError:
        # Source checkouts need an explicit rebuild when the native ABI changes.
        raise NativeBuildError() from None
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
        runtime._allocations += 1
        runtime._allocated_bytes += size
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


class WorkspaceLease:
    """Internal forward-scoped allocator; a released lease cannot be reused."""

    def __init__(self, runtime: "MetalRuntime") -> None:
        self.runtime = runtime
        self.buffers: list[tuple[Buffer, bool]] = []
        self.closed = False

    def __call__(self, size: int, data: NDArray[Any] | None = None) -> Buffer:
        if self.closed:
            raise InferenceError()
        rt = self.runtime
        buffer = None
        if data is None:
            for index, cached in enumerate(rt._scratch):
                if cached.size == size:
                    buffer = rt._scratch.pop(index)
                    rt.cache_bytes -= size
                    break
        if buffer is None:
            buffer = rt.buffer(size, data)
        self.buffers.append((buffer, data is None))
        return buffer

    def release(self) -> None:
        self.closed = True
        rt = self.runtime
        for buffer, reusable in self.buffers:
            if (
                reusable
                and buffer.pointer
                and buffer.size <= rt.workspace_limit_bytes - rt._plan_bytes
            ):
                while (
                    rt._scratch
                    and rt.cache_bytes + rt._plan_bytes + buffer.size > rt.workspace_limit_bytes
                ):
                    old = rt._scratch.pop(0)
                    rt.cache_bytes -= old.size
                    old.close()
                rt._scratch.append(buffer)
                rt.cache_bytes += buffer.size
            else:
                buffer.close()
        self.buffers.clear()


class MetalRuntime:
    """Synchronous GPU command owner. Commands are serialized per instance."""

    def __init__(self, *, workspace_limit_bytes: int = 64 * 1024 * 1024) -> None:
        if type(workspace_limit_bytes) is not int or not 0 <= workspace_limit_bytes <= 2**30:
            raise InferenceError()
        self.workspace_limit_bytes = workspace_limit_bytes
        self.cache_bytes = 0
        self._plans_enabled = True
        self._plans: dict[tuple[Any, ...], tuple[int, int, Counter[str]]] = {}
        self._plan_capture: tuple[int, dict[Buffer, int], Counter[str]] | None = None
        self._plan_bytes = 0
        self._plan_hits = 0
        self._plan_builds = 0
        self._scratch: list[Buffer] = []
        self._workspace_active = False
        self._lib = _library()
        source = (Path(__file__).parent / "native/kernels.metal").read_bytes()
        self._shader_sha256 = hashlib.sha256(source).hexdigest()
        self._pointer: int | None = self._lib.mi_create(source)
        if not self._pointer:
            raise MetalUnavailableError()
        self._lock = threading.RLock()
        self._recording = False
        self._inflight: list[Buffer] = []
        self._buffers: WeakSet[Buffer] = WeakSet()
        self.active_bytes = 0
        self.peak_bytes = 0
        self._allocations = 0
        self._allocated_bytes = 0
        self._commands = 0
        self._gpu_samples = 0
        self._gpu_seconds = 0.0
        self._encode_seconds = 0.0
        self._submit_wait_seconds = 0.0
        self._dispatches: Counter[str] = Counter()
        self._profile: list[dict[str, Any]] | None = None
        self._profile_command: list[dict[str, Any]] = []

    def diagnostics(self) -> dict[str, Any]:
        """Cumulative counters; GPU time covers completed commands, not individual kernels.

        Encoding includes Python/driver work and allocations inside command().
        Submit/wait includes GPU execution and must not be added to GPU time.
        Dispatch counts include encoded work discarded by command aborts.
        """
        with self._lock:
            return {
                "shader_sha256": self._shader_sha256,
                "native_library_sha256": self._lib._tessery_sha256,
                "allocations": self._allocations,
                "allocated_bytes": self._allocated_bytes,
                "completed_commands": self._commands,
                "gpu_timed_commands": self._gpu_samples,
                "gpu_seconds": self._gpu_seconds,
                "encode_seconds": self._encode_seconds,
                "submit_wait_seconds": self._submit_wait_seconds,
                "dispatches": dict(self._dispatches),
                "active_bytes": self.active_bytes,
                "cache_bytes": self.cache_bytes,
                "plan_cache_bytes": self._plan_bytes,
                "plan_cache_entries": len(self._plans),
                "plan_hits": self._plan_hits,
                "plan_builds": self._plan_builds,
                "peak_bytes": self.peak_bytes,
            }

    @contextmanager
    def profile_kernels(self) -> Iterator[list[dict[str, Any]]]:
        """Intrusive stage-boundary GPU timings, at most 2048 dispatches per command.

        Hold the runtime lock; call backend.forward directly from this thread, not
        the model's executor API. Unsupported counters fail explicitly. Results
        contain completed commands only and are not normal inference latencies.
        """
        with self._lock:
            if not self._pointer:
                raise ClosedError()
            if self._recording or self._profile is not None:
                raise InferenceError()
            if self._lib.mi_profile_enable(self._pointer, 1):
                raise InferenceError()
            records: list[dict[str, Any]] = []
            self._profile = records
            try:
                yield records
            finally:
                self._lib.mi_profile_enable(self._pointer, 0)
                self._profile = None
                self._profile_command.clear()

    @contextmanager
    def _workspace(self) -> Iterator[WorkspaceLease]:
        # The lock spans encoding, GPU completion and host readback. Buffers are
        # returned only after command() has finished or aborted unsubmitted work.
        with self._lock:
            if not self._pointer:
                raise ClosedError()
            if self._recording or self._workspace_active:
                raise InferenceError()
            self._workspace_active = True
            lease = WorkspaceLease(self)
            try:
                yield lease
            finally:
                lease.release()
                self._workspace_active = False

    def trim_workspace(self) -> None:
        """Release retained scratch buffers; waits for the current forward."""
        with self._lock:
            self._clear_plans()
            for buffer in self._scratch:
                buffer.close()
            self._scratch.clear()
            self.cache_bytes = 0

    def _clear_plans(self) -> None:
        for pointer, _, _ in self._plans.values():
            self._lib.mi_plan_free(pointer)
        self._plans.clear()
        self._plan_bytes = 0

    @contextmanager
    def _execution_plan(self, key: tuple[Any, ...], bindings: Sequence[Buffer]) -> Iterator[bool]:
        """Internal static graph capture; True means replay already encoded the graph.

        Call within command/workspace after all dynamic buffers are allocated.
        The backend supplies a complete, stable ordering of weights and scratch slots.
        Profiling bypasses plans to retain per-dispatch counter records.
        """
        if not self._recording or not self._workspace_active or self._plan_capture is not None:
            raise InferenceError()
        if not self._plans_enabled or self._profile is not None or not self.workspace_limit_bytes:
            yield False
            return
        if any(not b.pointer or b.runtime is not self for b in bindings):
            raise InferenceError()
        pointers = (ct.c_void_p * len(bindings))(*(b.pointer for b in bindings))
        self._inflight.extend(bindings)
        cached = self._plans.get(key)
        if cached is not None:
            if self._lib.mi_plan_run(self._pointer, cached[0], pointers, len(bindings)):
                raise InferenceError()
            self._dispatches.update(cached[2])
            self._plan_hits += 1
            yield True
            return
        pointer = self._lib.mi_plan_create(self._pointer, pointers, len(bindings))
        if not pointer:
            raise InferenceError()
        counts: Counter[str] = Counter()
        self._plan_capture = (pointer, {b: i for i, b in enumerate(bindings)}, counts)
        try:
            yield False
            size = int(self._lib.mi_plan_bytes(pointer))
            if size <= self.workspace_limit_bytes:
                # Bound metadata and scratch together; plans own no model/workspace data.
                while self._plans and (
                    len(self._plans) >= 4 or self._plan_bytes + size > self.workspace_limit_bytes
                ):
                    oldest = next(iter(self._plans))
                    old, used, _ = self._plans.pop(oldest)
                    self._lib.mi_plan_free(old)
                    self._plan_bytes -= used
                while (
                    self._scratch
                    and self.cache_bytes + self._plan_bytes + size > self.workspace_limit_bytes
                ):
                    old_buffer = self._scratch.pop(0)
                    self.cache_bytes -= old_buffer.size
                    old_buffer.close()
                self._plans[key] = (pointer, size, counts)
                self._plan_bytes += size
                self._plan_builds += 1
                pointer = None
        finally:
            self._plan_capture = None
            if pointer:
                self._lib.mi_plan_free(pointer)

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
            if self._recording:
                raise InferenceError()
            if self._lib.mi_begin(self._pointer):
                self._lib.mi_abort(self._pointer)
                raise InferenceError()
            self._recording = True
            self._profile_command.clear()
            started = time.perf_counter()
            try:
                yield
                submitted = time.perf_counter()
                if self._lib.mi_finish(self._pointer):
                    raise InferenceError()
                if self._profile is not None:
                    timings = (ct.c_double * len(self._profile_command))()
                    if self._lib.mi_profile_read(self._pointer, timings, len(timings)):
                        raise InferenceError()
                    self._profile.extend(
                        dict(record, gpu_seconds=float(seconds))
                        for record, seconds in zip(self._profile_command, timings, strict=True)
                    )
                self._encode_seconds += submitted - started
                self._submit_wait_seconds += time.perf_counter() - submitted
                self._commands += 1
                gpu_seconds = self._lib.mi_gpu_seconds(self._pointer)
                if gpu_seconds >= 0:
                    self._gpu_seconds += gpu_seconds
                    self._gpu_samples += 1
            except BaseException:
                self._clear_plans()
                raise
            finally:
                self._lib.mi_abort(self._pointer)
                self._recording = False
                self._inflight.clear()
                self._profile_command.clear()

    def _linear4(self, buffers: Sequence[Buffer], *, rows: int, cols: int, k: int) -> None:
        qwen_shape = (cols, k) in (
            (1024, 1024),
            (2048, 1024),
            (3072, 1024),
            (1024, 2048),
            (1024, 3072),
        )
        if rows == 3 and qwen_shape:
            self._dispatch(
                "linear4_small3",
                buffers,
                threads=cols * 32,
                group_size=32,
                rows=rows,
                cols=cols,
                k=k,
            )
            return
        # Partition verified Qwen shapes into full 16-row tiles, then at most
        # one eight-row tile and one bounded tail. Regions never overlap.
        start = 0
        if rows >= 16 and qwen_shape:
            self._dispatch(
                "linear4_16x32_k64",
                buffers,
                threads=(rows // 16) * (cols // 32) * 256,
                group_size=256,
                rows=rows,
                cols=cols,
                k=k,
            )
            start = (rows // 16) * 16
            if start == rows:
                return
        if rows >= 5 and cols % 32 == 0 and k % 64 == 0:
            complete = (rows - start) // 8
            if complete:
                self._dispatch(
                    "linear4_tiled",
                    buffers,
                    threads=complete * (cols // 32) * 128,
                    group_size=128,
                    n=start,
                    rows=rows,
                    cols=cols,
                    k=k,
                )
            if rows % 8:
                # n is the starting row for the single partial row tile.
                small = rows % 8 <= 4
                self._dispatch(
                    "linear4" if small else "linear4_tail",
                    buffers,
                    threads=cols * 32 if small else (cols // 32) * 128,
                    group_size=32 if small else 128,
                    n=start + complete * 8,
                    rows=rows,
                    cols=cols,
                    k=k,
                )
        else:
            self._dispatch(
                "linear4",
                buffers,
                threads=((rows + 3) // 4) * cols * 32,
                group_size=32,
                rows=rows,
                cols=cols,
                k=k,
            )

    def _gated4(self, buffers: Sequence[Buffer], *, rows: int, cols: int, k: int) -> None:
        # x, gate weight/scale/bias, up weight/scale/bias, gate output, up scratch.
        # Only measured full-model shapes use the fused epilogue. Keep the
        # original complete path for all other heights and projection shapes.
        if rows in (24, 128, 160, 256, 512) and (cols, k) == (3072, 1024):
            self._dispatch(
                "gated4_16x32_k64",
                buffers[:8],
                threads=(rows // 16) * (cols // 32) * 256,
                group_size=256,
                rows=rows,
                cols=cols,
                k=k,
            )
            if rows == 24:
                # One complete 16-row prefix followed by a disjoint 8-row tile.
                self._dispatch(
                    "gated4_8x32",
                    buffers[:8],
                    threads=(cols // 32) * 128,
                    group_size=128,
                    n=16,
                    rows=rows,
                    cols=cols,
                    k=k,
                )
            return
        self._linear4([*buffers[:4], buffers[7]], rows=rows, cols=cols, k=k)
        self._linear4([buffers[0], *buffers[4:7], buffers[8]], rows=rows, cols=cols, k=k)
        self._dispatch("silu_gate", [buffers[7], buffers[8]], threads=rows * cols, n=rows * cols)

    def _attention(
        self,
        buffers: Sequence[Buffer],
        *,
        tokens: int,
        seq: int,
        heads: int,
        kv_heads: int,
        dim: int,
        bidirectional: int = 0,
    ) -> None:
        if 64 <= seq <= 512 and seq % 32 and dim in (32, 128):
            self._dispatch(
                f"attention_tail_{dim}",
                buffers,
                threads=(tokens // seq) * ((seq + 7) // 8) * heads * 128,
                group_size=128,
                seq=seq,
                heads=heads,
                kv_heads=kv_heads,
                dim=dim,
                scale=dim**-0.5,
                bidirectional=bidirectional,
            )
            return
        tiled = seq >= 64 and seq % 32 == 0 and dim in (32, 128)
        self._dispatch(
            "attention_tiled" if tiled else "attention",
            buffers,
            threads=(tokens // 8 if tiled else tokens) * heads * (128 if tiled else 32),
            group_size=128 if tiled else 32,
            seq=seq,
            heads=heads,
            kv_heads=kv_heads,
            dim=dim,
            scale=dim**-0.5,
            bidirectional=bidirectional,
        )

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
        bidirectional: int = 0,
        first_token: int = 0,
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
            int(bidirectional),
            int(first_token),
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
        if self._plan_capture is not None:
            plan, slots, counts = self._plan_capture
            indices = (ct.c_uint32 * len(buffers))(*(slots[b] for b in buffers))
            if self._lib.mi_plan_add(
                plan, name.encode("ascii"), indices, len(buffers), parameters, threads, group_size
            ):
                raise InferenceError()
            counts[name] += 1
        self._dispatches[name] += 1
        if self._profile is not None:
            self._profile_command.append(
                {
                    "kernel": name,
                    "rows": rows,
                    "cols": cols,
                    "k": k,
                    "seq": seq,
                    "heads": heads,
                    "threads": threads,
                }
            )

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

    def _matmul_f32(self, buffers: Sequence[Buffer], *, rows: int, cols: int, k: int) -> None:
        if rows >= 8 and (cols, k) in ((384, 384), (1536, 384), (384, 1536)):
            complete = rows // 8
            self._dispatch(
                "matmul_f32_chunk32",
                buffers,
                threads=complete * (cols // 32) * 128,
                group_size=128,
                rows=rows,
                cols=cols,
                k=k,
            )
            if rows % 8:
                self._dispatch(
                    "matmul_f32",
                    buffers,
                    threads=((rows % 8 + 3) // 4) * cols * 32,
                    group_size=32,
                    n=complete * 8,
                    rows=rows,
                    cols=cols,
                    k=k,
                )
            return
        # Preserve the previous route outside the qualified BGE shapes.
        tiled = rows >= 8 and rows % 8 == 0 and cols % 32 == 0 and k % 8 == 0 and k <= 512
        self._dispatch(
            "matmul_f32_tiled" if tiled else "matmul_f32",
            buffers,
            threads=((rows + 7) // 8) * (cols // 32) * 128
            if tiled
            else ((rows + 3) // 4) * cols * 32,
            group_size=128 if tiled else 32,
            rows=rows,
            cols=cols,
            k=k,
        )

    def _matmul_bias_f32(self, buffers: Sequence[Buffer], *, rows: int, cols: int, k: int) -> None:
        """Affine BGE projection; buffers are input, transposed weight, output, bias."""
        if (cols, k) not in ((384, 384), (1536, 384), (384, 1536)):
            self._matmul_f32(buffers[:3], rows=rows, cols=cols, k=k)
            self._dispatch(
                "add_bias", [buffers[2], buffers[3]], threads=rows * cols, n=rows * cols, cols=cols
            )
            return
        complete = rows // 8
        if complete:
            self._dispatch(
                "matmul_bias_f32_chunk32",
                buffers,
                threads=complete * (cols // 32) * 128,
                group_size=128,
                rows=rows,
                cols=cols,
                k=k,
            )
        if rows % 8:
            self._dispatch(
                "matmul_bias_f32",
                buffers,
                threads=((rows % 8 + 3) // 4) * cols * 32,
                group_size=32,
                n=complete * 8,
                rows=rows,
                cols=cols,
                k=k,
            )

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
                    self._matmul_f32(
                        buffers,
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
            if self._recording or self._workspace_active or self._profile is not None:
                raise InferenceError()
            if self._pointer:
                self.trim_workspace()
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
