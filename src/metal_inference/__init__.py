"""Independent Metal inference, with no tensor framework dependency."""

from .api import EmbeddingModel, HealthStatus, MemoryStats, ModelDescriptor
from .index import Chunk, DocumentIndex, RetrievalHit, read_documents
from .metal import MetalRuntime
from .profiles import Artifact, ModelProfile, get_profile, list_profiles
from .retrieval import SearchHit, cosine_search
from .tensor import Tensor

__version__ = "0.5.1a1"
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
