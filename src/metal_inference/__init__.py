"""Independent Metal inference, with no tensor framework dependency."""

from .api import EmbeddingModel, HealthStatus, MemoryStats, ModelDescriptor
from .metal import MetalRuntime
from .retrieval import SearchHit, cosine_search

__version__ = "0.2.0a1"
__all__ = [
    "EmbeddingModel",
    "HealthStatus",
    "MemoryStats",
    "ModelDescriptor",
    "MetalRuntime",
    "SearchHit",
    "cosine_search",
]
