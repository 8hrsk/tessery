"""Reusable text-to-vector API with bounded admission and serialized GPU work."""

import asyncio
import hashlib
import json
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .errors import (
    CanceledError,
    ClosedError,
    ConfigurationError,
    EmbeddingError,
    InferenceError,
    InvalidInputError,
    OverloadError,
)
from .qwen3 import Qwen3Backend
from .tokenizer import QwenTokenizer
from .weights import ARTIFACTS, MODEL_ID, REVISION, read_json

ENGINE_ID = "metal-inference-qwen3-f32-v1"


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str = MODEL_ID
    revision: str = REVISION
    tokenizer_revision: str = REVISION
    native_dimensions: int = 1024
    min_dimensions: int = 32
    max_dimensions: int = 1024
    max_length: int = 512
    pooling: str = "last_non_padding_token"
    normalization: str = "unit_l2"
    quantization: str = "affine_uint4_group64_bf16_scales"
    compute_dtype: str = "float32"
    compatibility_id: str = ENGINE_ID
    manifest_sha256: str = hashlib.sha256(
        json.dumps(ARTIFACTS, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class HealthStatus:
    loaded: bool
    ready: bool
    compatibility_id: str


@dataclass(frozen=True)
class MemoryStats:
    active_bytes: int
    peak_bytes: int
    cache_bytes: int = 0


class EmbeddingModel:
    """One loaded model, one tokenizer, one Metal forward at a time.

    Admission is bounded across synchronous and asynchronous callers. Cancellation
    prevents queued work; an already submitted GPU command completes and its
    result is discarded. No prompt cache or implicit download is performed.
    """

    descriptor = ModelDescriptor()

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        dimensions: int = 384,
        max_length: int = 512,
        max_pending: int = 8,
    ) -> "EmbeddingModel":
        if (
            type(dimensions) is not int
            or not 32 <= dimensions <= 1024
            or type(max_length) is not int
            or not 1 <= max_length <= 512
            or type(max_pending) is not int
            or not 1 <= max_pending <= 64
        ):
            raise ConfigurationError()
        tokenizer = QwenTokenizer(read_json(str(model_dir), "tokenizer.json"))
        backend = Qwen3Backend(str(model_dir))
        return cls(backend, tokenizer, dimensions, max_length, max_pending)

    def __init__(
        self,
        backend: Qwen3Backend,
        tokenizer: QwenTokenizer,
        dimensions: int,
        max_length: int,
        max_pending: int,
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self.dimensions = dimensions
        self.max_length = max_length
        self._admission = threading.BoundedSemaphore(max_pending)
        self._metal_lock = threading.Lock()
        self._closed = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metal-inference")

    def _prepare(self, texts: Sequence[str], dimensions: int | None) -> tuple[list[str], int]:
        if self._closed.is_set():
            raise ClosedError()
        dims = self.dimensions if dimensions is None else dimensions
        if type(dims) is not int or not 32 <= dims <= 1024:
            raise InvalidInputError()
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence) or len(texts) > 32:
            raise InvalidInputError()
        snapshot = list(texts)
        size = 0
        for text in snapshot:
            if not isinstance(text, str) or not text.strip() or len(text) > 1024 * 1024:
                raise InvalidInputError()
            try:
                size += len(text.encode("utf-8"))
            except UnicodeError:
                raise InvalidInputError() from None
            if size > 1024 * 1024:
                raise InvalidInputError()
        return snapshot, dims

    def _run(
        self, texts: list[str], dimensions: int, canceled: threading.Event
    ) -> NDArray[np.float32]:
        try:
            with self._metal_lock:
                if self._closed.is_set():
                    raise ClosedError()
                if canceled.is_set():
                    raise CanceledError()
                ids, lengths = self._tokenizer.batch(texts, max_length=self.max_length)
                result = np.empty((len(texts), dimensions), dtype=np.float32)
                # Limit temporary GPU memory independently of caller batch size.
                step = max(1, self._backend.max_padded_tokens // ids.shape[1])
                for start in range(0, len(texts), step):
                    if canceled.is_set():
                        raise CanceledError()
                    end = min(start + step, len(texts))
                    result[start:end] = self._backend.forward(
                        ids[start:end],
                        lengths[start:end],
                        dimensions=dimensions,
                    )
                if canceled.is_set():
                    raise CanceledError()
                return result
        except EmbeddingError:
            raise
        except Exception:
            raise InferenceError() from None
        finally:
            self._admission.release()

    def encode(self, texts: Sequence[str], *, dimensions: int | None = None) -> NDArray[np.float32]:
        snapshot, dims = self._prepare(texts, dimensions)
        if not snapshot:
            return np.empty((0, dims), dtype=np.float32)
        if not self._admission.acquire(blocking=False):
            raise OverloadError()
        return self._run(snapshot, dims, threading.Event())

    async def encode_async(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> NDArray[np.float32]:
        snapshot, dims = self._prepare(texts, dimensions)
        if not snapshot:
            return np.empty((0, dims), dtype=np.float32)
        if not self._admission.acquire(blocking=False):
            raise OverloadError()
        canceled = threading.Event()
        try:
            future = self._executor.submit(self._run, snapshot, dims, canceled)
        except RuntimeError:
            self._admission.release()
            raise ClosedError() from None

        def on_done(completed: Future[NDArray[np.float32]]) -> None:
            # A canceled queued future never calls _run, so release its slot here.
            if completed.cancelled():
                self._admission.release()

        future.add_done_callback(on_done)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            canceled.set()
            raise

    def health(self) -> HealthStatus:
        loaded = not self._closed.is_set()
        return HealthStatus(loaded, loaded, ENGINE_ID)

    def memory_stats(self) -> MemoryStats:
        runtime = self._backend.runtime
        return MemoryStats(runtime.active_bytes, runtime.peak_bytes)

    def warmup(self) -> None:
        self.encode(["warmup"])

    def close(self) -> None:
        self._closed.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._metal_lock:
            self._backend.close()

    def __enter__(self) -> "EmbeddingModel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
