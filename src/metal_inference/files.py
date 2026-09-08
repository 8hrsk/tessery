"""Descriptor-relative local directory traversal without following symlinks."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import ManifestError


@contextmanager
def directory(path: str) -> Iterator[int]:
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
