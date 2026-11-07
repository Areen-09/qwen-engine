"""
qwen_engine: Cross-platform Python package for zero-dependency CUDA inference on Qwen2.5-1.5B.
"""
from .engine import Engine, SamplingParams
from .downloader import download, get_default_cache_dir
from .exceptions import (
    QwenError,
    QwenEngineError,
    ConcurrencyError,
    ContextWindowExceededError,
    CudaOutOfMemoryError,
    ModelNotFoundError,
)

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "SamplingParams",
    "download",
    "get_default_cache_dir",
    "QwenError",
    "QwenEngineError",
    "ConcurrencyError",
    "ContextWindowExceededError",
    "CudaOutOfMemoryError",
    "ModelNotFoundError",
    "__version__",
]

