"""
Exception hierarchy for qwen_engine.
"""
from typing import Optional

try:
    from . import _core
    QwenError = _core.QwenError
    ConcurrencyError = _core.ConcurrencyError
    ContextWindowExceededError = _core.ContextWindowExceededError
    CudaOutOfMemoryError = _core.CudaOutOfMemoryError
    ModelNotFoundError = _core.ModelNotFoundError
except ImportError:
    # Fallback classes if native _core is not yet compiled
    class QwenError(RuntimeError):
        """Base exception for all qwen_engine errors."""
        pass

    class ConcurrencyError(QwenError):
        """Raised when concurrent inference is attempted on an active engine session."""
        pass

    class ContextWindowExceededError(QwenError):
        """Raised when sequence position exceeds the model context window (MAX_SEQ_LEN)."""
        pass

    class CudaOutOfMemoryError(QwenError):
        """Raised when CUDA device memory allocation fails."""
        pass

    class ModelNotFoundError(FileNotFoundError, QwenError):
        """Raised when a model weight or tokenizer binary cannot be found."""
        pass

# Ergonomic aliases
QwenEngineError = QwenError

__all__ = [
    "QwenError",
    "QwenEngineError",
    "ConcurrencyError",
    "ContextWindowExceededError",
    "CudaOutOfMemoryError",
    "ModelNotFoundError",
]

