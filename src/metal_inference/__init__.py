"""Independent Metal inference, with no tensor framework dependency."""

from .api import EmbeddingModel, HealthStatus, MemoryStats, ModelDescriptor
from .metal import MetalRuntime
from .retrieval import SearchHit, cosine_search
from .tensor import Tensor

__version__ = "0.2.0a1"
__all__ = [
    "EmbeddingModel",
    "HealthStatus",
    "MemoryStats",
    "ModelDescriptor",
    "MetalRuntime",
    "SearchHit",
    "Tensor",
    "cosine_search",
]
