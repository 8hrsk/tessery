"""Public Tessery API. Imports do not load a model or initialize Metal."""

from metal_inference import (
    Artifact,
    Chunk,
    DocumentIndex,
    EmbeddingModel,
    HealthStatus,
    MemoryStats,
    MetalRuntime,
    ModelDescriptor,
    ModelProfile,
    RetrievalHit,
    SearchHit,
    Tensor,
    cosine_search,
    get_profile,
    list_profiles,
    read_documents,
)
from metal_inference import (
    __version__ as __version__,
)

__all__ = [
    "EmbeddingModel",
    "DocumentIndex",
    "Chunk",
    "RetrievalHit",
    "read_documents",
    "Artifact",
    "ModelProfile",
    "get_profile",
    "list_profiles",
    "HealthStatus",
    "MemoryStats",
    "ModelDescriptor",
    "MetalRuntime",
    "SearchHit",
    "Tensor",
    "cosine_search",
]
