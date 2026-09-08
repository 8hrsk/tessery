"""Stable errors: callers cannot accidentally include sensitive payloads."""


class EmbeddingError(Exception):
    code = "embedding_error"
    http_status = 500

    def __init__(self) -> None:
        super().__init__(self.code)


class ConfigurationError(EmbeddingError):
    code = "configuration_error"


class InvalidInputError(EmbeddingError):
    code = "invalid_input"
    http_status = 400


class ManifestError(EmbeddingError):
    code = "invalid_manifest"


class UnsupportedProfileError(EmbeddingError):
    code = "unsupported_profile"


class OverloadError(EmbeddingError):
    code = "overloaded"
    http_status = 429


class CanceledError(EmbeddingError):
    code = "canceled"


class DeadlineExceededError(EmbeddingError):
    code = "deadline_exceeded"


class InferenceError(EmbeddingError):
    code = "inference_failed"


class ClosedError(EmbeddingError):
    code = "closed"


class PrerequisiteError(EmbeddingError):
    code = "baseline_required"


class MetalUnavailableError(EmbeddingError):
    code = "metal_unavailable"


class NativeBuildError(EmbeddingError):
    code = "native_engine_not_built"
