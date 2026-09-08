"""Bounded exact retrieval with immutable SQLite snapshots and explicit vector spaces."""

import hashlib
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .api import EmbeddingModel
from .errors import InvalidInputError, ManifestError
from .json_codec import dumps, loads
from .retrieval import cosine_search

MAX_CHUNKS = 10000
MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Chunk:
    source: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float


def _contract(model: EmbeddingModel) -> dict[str, Any]:
    return {
        "descriptor": asdict(model.descriptor),
        "dimensions": model.dimensions,
        "max_length": model.max_length,
    }


def _digest(
    metadata: dict[str, Any], chunks: tuple[Chunk, ...], vectors: NDArray[np.float32]
) -> str:
    digest = hashlib.sha256(dumps(metadata, limit=65536))
    for chunk, vector in zip(chunks, vectors, strict=True):
        encoded = dumps(asdict(chunk), limit=1024 * 1024)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(vector.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def read_documents(directory: str | Path) -> dict[str, str]:
    """Read a bounded directory of UTF-8 .txt/.md files in stable path order."""
    root = Path(directory)
    if not root.is_dir():
        raise InvalidInputError()
    documents: dict[str, str] = {}
    remaining = MAX_TEXT_BYTES
    paths = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file() or path.is_symlink():
            continue
        paths.append(path)
        if len(paths) > 1000:
            raise InvalidInputError()
    for path in sorted(paths):
        with path.open("rb") as stream:
            raw = stream.read(remaining + 1)
        remaining -= len(raw)
        if remaining < 0 or len(documents) >= 1000:
            raise InvalidInputError()
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            raise InvalidInputError() from None
        if text.strip():
            documents[path.relative_to(root).as_posix()] = text
    if not documents:
        raise InvalidInputError()
    return documents


class DocumentIndex:
    """Small corpus exact cosine index. No ANN, generation or implicit model download."""

    def __init__(
        self, metadata: dict[str, Any], chunks: tuple[Chunk, ...], vectors: NDArray[np.float32]
    ) -> None:
        self._metadata = metadata
        self.chunks = chunks
        self._vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
        self._vectors.flags.writeable = False

    @classmethod
    def build(
        cls,
        model: EmbeddingModel,
        documents: Mapping[str, str],
        *,
        chunk_chars: int = 600,
        overlap_chars: int = 80,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> "DocumentIndex":
        if (
            not isinstance(documents, Mapping)
            or not 1 <= len(documents) <= 1000
            or type(chunk_chars) is not int
            or not 1 <= chunk_chars <= 8192
            or type(overlap_chars) is not int
            or not 0 <= overlap_chars < chunk_chars
            or not isinstance(document_prefix, str)
            or not isinstance(query_prefix, str)
            or len(document_prefix) > 1024
            or len(query_prefix) > 1024
        ):
            raise InvalidInputError()
        chunks: list[Chunk] = []
        total = 0
        for source, text in documents.items():
            if (
                not isinstance(source, str)
                or not source
                or len(source) > 4096
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise InvalidInputError()
            try:
                total += len(text.encode("utf-8"))
                source.encode("utf-8")
            except UnicodeError:
                raise InvalidInputError() from None
            if total > MAX_TEXT_BYTES:
                raise InvalidInputError()
            start = 0
            while start < len(text):
                end = min(start + chunk_chars, len(text))
                # A saturated tokenizer length might conceal truncation. Shrink
                # conservatively until strictly below the cap, including prefix.
                while text[start:end].strip():
                    _, lengths = model._tokenizer.batch(
                        [document_prefix + text[start:end]], max_length=model.max_length
                    )
                    if int(lengths[0]) < model.max_length:
                        break
                    if end - start == 1:
                        raise InvalidInputError()
                    end = start + (end - start) // 2
                if text[start:end].strip():
                    chunks.append(Chunk(source, start, end, text[start:end]))
                if len(chunks) > MAX_CHUNKS:
                    raise InvalidInputError()
                if end == len(text):
                    break
                start = max(start + 1, end - min(overlap_chars, (end - start) // 2))
        if not chunks:
            raise InvalidInputError()
        # Include duplicated passages, source labels and vectors in the bound,
        # so even high-dimensional custom profiles produce reloadable snapshots.
        text_bytes = sum(len(c.text.encode("utf-8")) for c in chunks)
        payload_bytes = text_bytes + sum(len(c.source.encode("utf-8")) + 64 for c in chunks)
        payload_bytes += len(chunks) * model.dimensions * 4
        if text_bytes > 2 * MAX_TEXT_BYTES or payload_bytes > MAX_INDEX_BYTES // 2:
            raise InvalidInputError()
        vectors = np.concatenate(
            [
                model.encode([document_prefix + c.text for c in chunks[offset : offset + 32]])
                for offset in range(0, len(chunks), 32)
            ]
        )
        metadata = {
            "format": "tessery-exact-index-v1",
            "contract": _contract(model),
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "document_prefix": document_prefix,
            "query_prefix": query_prefix,
        }
        return cls(metadata, tuple(chunks), vectors)

    def search(self, model: EmbeddingModel, query: str, *, k: int = 5) -> list[RetrievalHit]:
        if _contract(model) != self._metadata["contract"]:
            raise ManifestError()
        if (
            not isinstance(query, str)
            or not query.strip()
            or type(k) is not int
            or not 1 <= k <= 100
        ):
            raise InvalidInputError()
        vector = model.encode([self._metadata["query_prefix"] + query])[0]
        return [
            RetrievalHit(self.chunks[hit.index], hit.score)
            for hit in cosine_search(vector, self._vectors, k=k)
        ]

    def save(self, path: str | Path) -> None:
        """Publish a new snapshot atomically. Never overwrite an existing index."""
        target = Path(path)
        fd, temporary = tempfile.mkstemp(prefix=".tessery-index-", dir=target.parent)
        os.close(fd)
        try:
            connection = sqlite3.connect(temporary)
            try:
                connection.execute("CREATE TABLE metadata (json TEXT NOT NULL)")
                connection.execute(
                    "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source TEXT, start INTEGER, "
                    "end INTEGER, text TEXT, vector BLOB)"
                )
                envelope = {
                    "metadata": self._metadata,
                    "sha256": _digest(self._metadata, self.chunks, self._vectors),
                }
                connection.execute(
                    "INSERT INTO metadata VALUES (?)", (dumps(envelope, limit=65536).decode(),)
                )
                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (i, c.source, c.start, c.end, c.text, v.astype("<f4", copy=False).tobytes())
                        for i, (c, v) in enumerate(zip(self.chunks, self._vectors, strict=True))
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, target)  # Atomic create-only publication, same filesystem.
        finally:
            os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> "DocumentIndex":
        target = Path(path).absolute()
        if target.stat().st_size > MAX_INDEX_BYTES:
            raise ManifestError()
        try:
            connection = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error:
            raise ManifestError() from None
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024 * 1024)
            envelope_rows = connection.execute("SELECT json FROM metadata LIMIT 2").fetchall()
            if len(envelope_rows) != 1:
                raise ManifestError()
            envelope = loads(envelope_rows[0][0].encode(), limit=65536)
            meta = envelope["metadata"]
            if (
                set(meta)
                != {
                    "format",
                    "contract",
                    "chunk_chars",
                    "overlap_chars",
                    "document_prefix",
                    "query_prefix",
                }
                or meta["format"] != "tessery-exact-index-v1"
                or not isinstance(meta["query_prefix"], str)
            ):
                raise ManifestError()
            dimensions = meta["contract"]["dimensions"]
            if type(dimensions) is not int or not 1 <= dimensions <= 4096:
                raise ManifestError()
            chunks = []
            vectors = []
            total = 0
            rows = connection.execute(
                "SELECT id, source, start, end, text, vector FROM chunks ORDER BY id LIMIT ?",
                (MAX_CHUNKS + 1,),
            )
            for expected_id, (index, source, start, end, text, blob) in enumerate(rows):
                if (
                    index != expected_id
                    or expected_id >= MAX_CHUNKS
                    or not isinstance(source, str)
                    or not source
                    or len(source) > 4096
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end
                    or not isinstance(text, str)
                    or not text.strip()
                    or end - start != len(text)
                    or not isinstance(blob, bytes)
                    or len(blob) != dimensions * 4
                ):
                    raise ManifestError()
                total += len(text.encode("utf-8"))
                if total > 2 * MAX_TEXT_BYTES:
                    raise ManifestError()
                chunks.append(Chunk(source, start, end, text))
                vectors.append(np.frombuffer(blob, dtype="<f4"))
            matrix = np.asarray(vectors, dtype=np.float32)
            if (
                not chunks
                or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4, rtol=0)
                or _digest(meta, tuple(chunks), matrix) != envelope["sha256"]
            ):
                raise ManifestError()
            return cls(meta, tuple(chunks), matrix)
        except (sqlite3.Error, KeyError, TypeError, ValueError, AttributeError, InvalidInputError):
            raise ManifestError() from None
        finally:
            connection.close()
