"""Yuri embedding contracts. Model inference awaits the baseline gate."""

from .config import LoadOptions
from .descriptors import HealthStatus, MemoryStats, ModelDescriptor, ModelProfile

__version__ = "0.1.0a1"
__all__ = ["HealthStatus", "LoadOptions", "MemoryStats", "ModelDescriptor", "ModelProfile"]
