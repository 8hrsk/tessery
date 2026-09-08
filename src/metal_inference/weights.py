"""Read verified local bytes; parse SafeTensors without executable deserialization."""

import hashlib
import math
import os
import stat
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import json_codec as strict_json
from .errors import InvalidInputError, ManifestError
from .files import directory

MODEL_ID = "Qwen3-Embedding-0.6B-4bit-DWQ"
REVISION = "6c3ae70858513f1a78e9cdca3cae330d9075cd2a"
ARTIFACTS = {
    "config.json": (937, "e7dfa5b73fb2a03cbc8fb40c394e95b99f03348e237f7f28e7a1daf56a2169bb"),
    "tokenizer.json": (
        11423705,
        "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    ),
    "model.safetensors": (
        335296756,
        "3d773d5ee582eda445daeee23f7a2b76124011796df244ddb45e22638fdb7cde",
    ),
}


def read_artifact(model_dir: str, name: str) -> bytes:
    """No-follow opens, no hardlinks, bounded read and hash of bytes actually used.

    Additional files in a standalone download directory are ignored, never read
    or imported. The optional legacy strict manifest tool still enforces exact sets.
    """
    if name not in ARTIFACTS:
        raise ManifestError()
    size, digest = ARTIFACTS[name]
    try:
        with directory(model_dir) as directory_fd:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != size:
                    raise ManifestError()
                data = stream.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ManifestError()
        return data
    except (OSError, ValueError):
        raise ManifestError() from None


def read_json(model_dir: str, name: str) -> dict[str, Any]:
    try:
        value = strict_json.loads(read_artifact(model_dir, name), limit=16 * 1024 * 1024)
    except InvalidInputError:
        raise ManifestError() from None
    if not isinstance(value, dict):
        raise ManifestError()
    return value


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


class SafeTensors:
    """A byte snapshot; all GPU uploads consume views of this verified snapshot."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        if len(data) < 8:
            raise ManifestError()
        length = struct.unpack_from("<Q", data)[0]
        if not 0 < length <= 4 * 1024 * 1024 or length + 8 > len(data):
            raise ManifestError()
        try:
            header = strict_json.loads(data[8 : 8 + length], limit=4 * 1024 * 1024)
        except InvalidInputError:
            raise ManifestError() from None
        if not isinstance(header, dict):
            raise ManifestError()
        self.base = length + 8
        self.tensors: dict[str, TensorInfo] = {}
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise ManifestError()
            dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
            if (
                not isinstance(dtype, str)
                or dtype not in {"BF16", "U32"}
                or not isinstance(shape, list)
                or not 1 <= len(shape) <= 2
                or any(type(d) is not int or not 0 < d <= 200000 for d in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(d) is not int for d in offsets)
            ):
                raise ManifestError()
            start, end = offsets
            width = 2 if dtype == "BF16" else 4
            if (
                not 0 <= start < end <= len(data) - self.base
                or end - start != math.prod(shape) * width
            ):
                raise ManifestError()
            self.tensors[name] = TensorInfo(dtype, tuple(shape), start, end)
        cursor = 0
        for item in sorted(self.tensors.values(), key=lambda x: x.start):
            if item.start != cursor:
                raise ManifestError()
            cursor = item.end
        if cursor != len(data) - self.base:
            raise ManifestError()

    def view(self, name: str, *, shape: tuple[int, ...], dtype: str) -> NDArray[np.uint8]:
        tensor = self.tensors.get(name)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            raise ManifestError()
        return np.frombuffer(
            self.data,
            dtype=np.uint8,
            count=tensor.end - tensor.start,
            offset=self.base + tensor.start,
        )
