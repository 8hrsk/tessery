"""Immutable, static profile registry; no plugins or model-directory imports."""

from dataclasses import dataclass
from types import MappingProxyType

MODEL_ID = "Qwen3-Embedding-0.6B-4bit-DWQ"
REVISION = "6c3ae70858513f1a78e9cdca3cae330d9075cd2a"
PROTOCOL = "yuri-embedding-protocol-v1"
# Deliberately distinct from the legacy engine and pack identifiers.
ENGINE_ID = "yuri-independent-qwen3-mrl384-v0-unverified"
PACK_ID = "yuri-independent-qwen3-0.6b-4bit-mrl384-v0-unverified"


@dataclass(frozen=True)
class ModelProfile:
    model_id: str = MODEL_ID
    revision: str = REVISION
    tokenizer_revision: str = REVISION
    native_dimensions: int = 1024
    output_dimensions: tuple[int, ...] = (384,)
    max_length: int = 512
    pooling: str = "last_token"
    normalization: str = "unit_l2"
    quantization: str = "4bit-DWQ"
    compatibility_id: str = ENGINE_ID


YURI_V1 = ModelProfile()
PROFILES = MappingProxyType({MODEL_ID: YURI_V1})


@dataclass(frozen=True)
class ModelDescriptor:
    profile: ModelProfile
    model_manifest_sha256: str
    tokenizer_manifest_sha256: str


@dataclass(frozen=True)
class HealthStatus:
    loaded: bool
    ready: bool
    compatibility_id: str


@dataclass(frozen=True)
class MemoryStats:
    active_bytes: int | None
    peak_bytes: int | None
    cache_bytes: int | None
