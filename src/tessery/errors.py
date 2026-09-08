"""Stable Tessery exceptions, shared with the compatibility namespace."""

from metal_inference.errors import (
    CanceledError,
    ClosedError,
    ConfigurationError,
    DeadlineExceededError,
    EmbeddingError,
    InferenceError,
    InvalidInputError,
    ManifestError,
    MetalUnavailableError,
    NativeBuildError,
    OverloadError,
    PrerequisiteError,
    UnsupportedProfileError,
)

__all__ = [
    "EmbeddingError",
    "ConfigurationError",
    "InvalidInputError",
    "ManifestError",
    "UnsupportedProfileError",
    "OverloadError",
    "CanceledError",
    "DeadlineExceededError",
    "InferenceError",
    "ClosedError",
    "PrerequisiteError",
    "MetalUnavailableError",
    "NativeBuildError",
]
