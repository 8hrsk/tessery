"""Verify an exact flat model pack against an externally pinned manifest digest.

Uses descriptor-relative, no-follow opens for every path component and streams
hashes without loading weights into memory. This is validation, not a signature
authority: the supervisor must supply a trusted manifest SHA-256. A future loader
must consume verified bytes/handles, never reopen unverified paths after this call.
"""

import hashlib
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import strict_json
from .descriptors import MODEL_ID, REVISION
from .errors import InvalidInputError, ManifestError

REQUIRED_FILES = frozenset(
    {
        "model.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
MANIFEST_LIMIT = 64 * 1024
HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True)
class ModelManifest:
    artifacts: tuple[Artifact, ...]
    sha256: str
    model_revision: str = REVISION
    tokenizer_revision: str = REVISION


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def parse_manifest(data: bytes, *, expected_sha256: str) -> ModelManifest:
    if not _valid_hash(expected_sha256) or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ManifestError()
    try:
        obj = strict_json.loads(data, limit=MANIFEST_LIMIT)
    except InvalidInputError:
        raise ManifestError() from None
    if not isinstance(obj, dict) or set(obj) != {
        "schema_version",
        "model",
        "model_revision",
        "tokenizer_revision",
        "license",
        "files",
    }:
        raise ManifestError()
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != 1
        or obj["model"] != MODEL_ID
        or obj["model_revision"] != REVISION
        or obj["tokenizer_revision"] != REVISION
        or obj["license"] != "Apache-2.0"
        or not isinstance(obj["files"], list)
        or len(obj["files"]) != len(REQUIRED_FILES)
    ):
        raise ManifestError()
    files: dict[str, Artifact] = {}
    for row in obj["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size", "mode"}:
            raise ManifestError()
        name = row["path"]
        if (
            not isinstance(name, str)
            or name not in REQUIRED_FILES
            or name in files
            or not _valid_hash(row["sha256"])
            or type(row["size"]) is not int
            or not 0 < row["size"] <= 4 * 1024**3
            or type(row["mode"]) is not int
            or row["mode"] not in {0o400, 0o444, 0o600, 0o644}
        ):
            raise ManifestError()
        files[name] = Artifact(name, row["sha256"], row["size"], row["mode"])
    return ModelManifest(tuple(files[name] for name in sorted(files)), expected_sha256)


@contextmanager
def _directory(path: str) -> Iterator[int]:
    # Reject lexical traversal before pathlib normalizes the spelling.
    if not path.startswith("/") or any(part in {".", ".."} for part in path.split("/")):
        raise ManifestError()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(path).parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def verify_model(model_dir: str, manifest: ModelManifest) -> None:
    """Raise a safe ManifestError on any missing, extra, special or corrupt file."""
    # Dataclass construction is public. Revalidate names before descriptor-relative
    # open so a caller cannot bypass the parser with a hand-constructed manifest.
    if (
        len(manifest.artifacts) != len(REQUIRED_FILES)
        or {a.path for a in manifest.artifacts} != REQUIRED_FILES
        or manifest.model_revision != REVISION
        or manifest.tokenizer_revision != REVISION
    ):
        raise ManifestError()
    try:
        with _directory(model_dir) as directory:
            if set(os.listdir(directory)) != REQUIRED_FILES:
                raise ManifestError()
            for artifact in manifest.artifacts:
                # NONBLOCK prevents a replaced FIFO from blocking before fstat.
                fd = os.open(
                    artifact.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
                )
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or before.st_nlink != 1
                        or before.st_size != artifact.size
                        or stat.S_IMODE(before.st_mode) != artifact.mode
                    ):
                        raise ManifestError()
                    digest = hashlib.sha256()
                    count = 0
                    while chunk := stream.read(min(1024 * 1024, artifact.size - count + 1)):
                        count += len(chunk)
                        if count > artifact.size:
                            raise ManifestError()
                        digest.update(chunk)
                    after = os.fstat(stream.fileno())
                    if (
                        count != artifact.size
                        or digest.hexdigest() != artifact.sha256
                        or (before.st_mtime_ns, before.st_ctime_ns, before.st_size)
                        != (after.st_mtime_ns, after.st_ctime_ns, after.st_size)
                    ):
                        raise ManifestError()
            if set(os.listdir(directory)) != REQUIRED_FILES:
                raise ManifestError()
    except (OSError, ValueError):
        raise ManifestError() from None
