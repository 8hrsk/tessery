"""Immutable, data-only model profiles; supported architecture code lives in the engine."""

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import InvalidInputError, ManifestError, UnsupportedProfileError
from .json_codec import loads


@dataclass(frozen=True)
class Artifact:
    name: str
    filename: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name not in {"config.json", "tokenizer.json", "model.safetensors"}
            or not isinstance(self.filename, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", self.filename) is None
            or type(self.size) is not int
            or not 0 < self.size <= 2**31
            or not isinstance(self.sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", self.sha256) is None
        ):
            raise ManifestError()
        if self.name != "model.safetensors" and self.size > 16 * 1024 * 1024:
            raise ManifestError()


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    revision: str
    architecture: str
    tokenizer: str
    pooling: str
    native_dimensions: int
    min_dimensions: int
    default_dimensions: int
    max_length: int
    artifacts: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) or not 1 <= len(x) <= 256 for x in (self.model_id, self.revision)
        ):
            raise ManifestError()
        if (self.architecture, self.tokenizer, self.pooling) not in (
            ("qwen3_uint4", "qwen_bpe", "last_non_padding_token"),
            ("bert_f32", "bert_wordpiece", "cls"),
        ):
            raise UnsupportedProfileError()
        if (
            any(
                type(x) is not int
                for x in (
                    self.native_dimensions,
                    self.min_dimensions,
                    self.default_dimensions,
                    self.max_length,
                )
            )
            or not 32
            <= self.min_dimensions
            <= self.default_dimensions
            <= self.native_dimensions
            <= 4096
            or not self.min_length <= self.max_length <= 512
            or (self.architecture == "bert_f32" and self.min_dimensions != self.native_dimensions)
            or not isinstance(self.artifacts, tuple)
            or len(self.artifacts) != 3
            or any(not isinstance(x, Artifact) for x in self.artifacts)
            or {x.name for x in self.artifacts}
            != {"config.json", "tokenizer.json", "model.safetensors"}
            or len({x.filename for x in self.artifacts}) != 3
            or sum(x.size for x in self.artifacts) > 2**31
        ):
            raise ManifestError()

    @property
    def min_length(self) -> int:
        return 2 if self.tokenizer == "bert_wordpiece" else 1

    def artifact(self, name: str) -> Artifact:
        for item in self.artifacts:
            if item.name == name:
                return item
        raise ManifestError()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            **asdict(self),
            "artifacts": [asdict(a) for a in self.artifacts],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ModelProfile":
        if not isinstance(data, dict):
            raise ManifestError()
        fields = set(cls.__dataclass_fields__)
        if (
            set(data) != fields | {"schema_version"}
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
        ):
            raise ManifestError()
        try:
            values = {k: data[k] for k in fields}
            if not isinstance(values["artifacts"], list):
                raise ManifestError()
            values["artifacts"] = tuple(Artifact(**row) for row in values["artifacts"])
            return cls(**values)
        except (TypeError, ValueError):
            raise ManifestError() from None

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelProfile":
        """Read a caller-selected trusted manifest. Its hashes are an integrity anchor,
        not proof of publisher identity or model quality. No imports or network hooks.
        """
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ManifestError()
                data = loads(stream.read(65537), limit=65536)
            return cls.from_dict(data)
        except (OSError, InvalidInputError):
            raise ManifestError() from None

    @property
    def identity_sha256(self) -> str:
        # Locations and manifest ordering do not change the embedding space.
        data = self.to_dict()
        data["artifacts"] = [
            {"name": x.name, "size": x.size, "sha256": x.sha256}
            for x in sorted(self.artifacts, key=lambda x: x.name)
        ]
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def compatibility_id(self) -> str:
        if self.identity_sha256 == QWEN3_PROFILE.identity_sha256:
            return "metal-inference-qwen3-f32-v1"
        return f"tessery-{self.architecture}-v1:{self.identity_sha256}"


QWEN3_PROFILE = ModelProfile(
    "Qwen3-Embedding-0.6B-4bit-DWQ",
    "6c3ae70858513f1a78e9cdca3cae330d9075cd2a",
    "qwen3_uint4",
    "qwen_bpe",
    "last_non_padding_token",
    1024,
    32,
    384,
    512,
    (
        Artifact(
            "config.json",
            "config.json",
            937,
            "e7dfa5b73fb2a03cbc8fb40c394e95b99f03348e237f7f28e7a1daf56a2169bb",
        ),
        Artifact(
            "tokenizer.json",
            "tokenizer.json",
            11423705,
            "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        ),
        Artifact(
            "model.safetensors",
            "model.safetensors",
            335296756,
            "3d773d5ee582eda445daeee23f7a2b76124011796df244ddb45e22638fdb7cde",
        ),
    ),
)

BGE_SMALL_PROFILE = ModelProfile(
    "BAAI/bge-small-en-v1.5",
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "bert_f32",
    "bert_wordpiece",
    "cls",
    384,
    384,
    384,
    512,
    (
        Artifact(
            "config.json",
            "config.json",
            743,
            "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
        ),
        Artifact(
            "tokenizer.json",
            "tokenizer.json",
            711396,
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        ),
        Artifact(
            "model.safetensors",
            "model.safetensors",
            133466304,
            "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        ),
    ),
)


def list_profiles() -> tuple[str, ...]:
    return ("qwen3-embedding-0.6b-dwq", "bge-small-en-v1.5")


def get_profile(profile: str | ModelProfile) -> ModelProfile:
    if isinstance(profile, ModelProfile):
        return profile
    if profile == "qwen3-embedding-0.6b-dwq":
        return QWEN3_PROFILE
    if profile == "bge-small-en-v1.5":
        return BGE_SMALL_PROFILE
    raise UnsupportedProfileError()
