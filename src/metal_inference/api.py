"""Reusable text-to-vector API with bounded admission and serialized GPU work."""

import asyncio
import hashlib
import json
import threading
from collections.abc import Sequence
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .backend import Backend, Tokenizer
from .batching import execution_batches
from .bert import BertBackend
from .errors import (
    CanceledError,
    ClosedError,
    ConfigurationError,
    EmbeddingError,
    InferenceError,
    InvalidInputError,
    OverloadError,
    UnsupportedProfileError,
)
from .profiles import QWEN3_PROFILE, ModelProfile, get_profile
from .qwen3 import Qwen3Backend
from .tokenizer import QwenTokenizer, validate_qwen_profile
from .weights import ARTIFACTS, MODEL_ID, REVISION, read_json
from .wordpiece import WordPieceTokenizer

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
    architecture: str = "qwen3_uint4"
    tokenizer: str = "qwen_bpe"

    @classmethod
    def from_profile(cls, profile: ModelProfile) -> "ModelDescriptor":
        return cls(
            model_id=profile.model_id,
            revision=profile.revision,
            tokenizer_revision=profile.revision,
            native_dimensions=profile.native_dimensions,
            min_dimensions=profile.min_dimensions,
            max_dimensions=profile.native_dimensions,
            max_length=profile.max_length,
            pooling=profile.pooling,
            quantization="none"
            if profile.architecture == "bert_f32"
            else "affine_uint4_group64_bf16_scales",
            compatibility_id=profile.compatibility_id,
            manifest_sha256=cls().manifest_sha256
            if profile.identity_sha256 == QWEN3_PROFILE.identity_sha256
            else profile.identity_sha256,
            architecture=profile.architecture,
            tokenizer=profile.tokenizer,
        )


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
    prevents queued work and cooperatively stops CPU tokenization; a submitted GPU
    command completes and its
    result is discarded. No prompt cache or implicit download is performed.
    """

    descriptor = ModelDescriptor()

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        dimensions: int | None = None,
        max_length: int | None = None,
        max_pending: int = 8,
        workspace_limit_bytes: int = 64 * 1024 * 1024,
        profile: str | ModelProfile = "qwen3-embedding-0.6b-dwq",
    ) -> "EmbeddingModel":
        selected = get_profile(profile)
        dimensions = selected.default_dimensions if dimensions is None else dimensions
        max_length = selected.max_length if max_length is None else max_length
        if (
            type(dimensions) is not int
            or not selected.min_dimensions <= dimensions <= selected.native_dimensions
            or type(max_length) is not int
            or not selected.min_length <= max_length <= selected.max_length
            or type(max_pending) is not int
            or not 1 <= max_pending <= 64
            or type(workspace_limit_bytes) is not int
            or not 0 <= workspace_limit_bytes <= 2**30
        ):
            raise ConfigurationError()
        tokenizer_data = read_json(str(model_dir), "tokenizer.json", profile=selected)
        if selected.tokenizer == "qwen_bpe":
            validate_qwen_profile(tokenizer_data)
        tokenizer: Tokenizer = (
            QwenTokenizer(tokenizer_data)
            if selected.tokenizer == "qwen_bpe"
            else WordPieceTokenizer(tokenizer_data)
        )
        config = read_json(str(model_dir), "config.json", profile=selected)
        if tokenizer.vocab_size != config.get("vocab_size"):
            raise UnsupportedProfileError()
        backend: Backend = (
            Qwen3Backend(str(model_dir), selected, workspace_limit_bytes=workspace_limit_bytes)
            if selected.architecture == "qwen3_uint4"
            else BertBackend(str(model_dir), selected, workspace_limit_bytes=workspace_limit_bytes)
        )
        try:
            return cls(
                backend,
                tokenizer,
                dimensions,
                max_length,
                max_pending,
                ModelDescriptor.from_profile(selected),
            )
        except BaseException:
            backend.close()
            raise

    def __init__(
        self,
        backend: Backend,
        tokenizer: Tokenizer,
        dimensions: int,
        max_length: int,
        max_pending: int,
        descriptor: ModelDescriptor | None = None,
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self.descriptor = ModelDescriptor() if descriptor is None else descriptor
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
        if (
            type(dims) is not int
            or not self.descriptor.min_dimensions <= dims <= self.descriptor.max_dimensions
        ):
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
                ids, lengths = self._tokenizer.batch(
                    texts, max_length=self.max_length, canceled=canceled.is_set
                )
                result = np.empty((len(texts), dimensions), dtype=np.float32)
                # Limit temporary GPU memory independently of caller batch size.
                for rows, width in execution_batches(
                    lengths,
                    self._backend.max_padded_tokens,
                    self.max_length,
                    self.descriptor.architecture,
                ):
                    if canceled.is_set():
                        raise CanceledError()
                    batch_ids = np.ascontiguousarray(ids[rows, :width])
                    if width > ids.shape[1]:
                        padded = np.full((len(rows), width), self._tokenizer.pad_id, np.uint32)
                        padded[:, : ids.shape[1]] = batch_ids
                        batch_ids = padded
                    result[rows] = self._backend.forward(
                        batch_ids,
                        lengths[rows],
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
        try:
            return self._submit(snapshot, dims, threading.Event()).result()
        except FutureCancelledError:
            raise ClosedError() from None

    def _submit(
        self, snapshot: list[str], dims: int, canceled: threading.Event
    ) -> Future[NDArray[np.float32]]:
        # Both APIs submit to the same single worker so synchronous callers do
        # not bypass already queued asynchronous work by racing for a lock.
        if not self._admission.acquire(blocking=False):
            raise OverloadError()
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
        return future

    async def encode_async(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> NDArray[np.float32]:
        snapshot, dims = self._prepare(texts, dimensions)
        if not snapshot:
            return np.empty((0, dims), dtype=np.float32)
        canceled = threading.Event()
        future = self._submit(snapshot, dims, canceled)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            canceled.set()
            raise

    def health(self) -> HealthStatus:
        loaded = not self._closed.is_set()
        return HealthStatus(loaded, loaded, self.descriptor.compatibility_id)

    def memory_stats(self) -> MemoryStats:
        # Keep active/cache counters from the same completed forward or trim.
        with self._metal_lock:
            runtime = self._backend.runtime
            return MemoryStats(
                runtime.active_bytes, runtime.peak_bytes, getattr(runtime, "cache_bytes", 0)
            )

    def trim_memory(self) -> None:
        """Release cached workspace after any running forward finishes."""
        with self._metal_lock:
            if self._closed.is_set():
                raise ClosedError()
            self._backend.runtime.trim_workspace()

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
