# src/embeddy/__init__.py
from __future__ import annotations

"""Public package interface for Embeddy.

This module defines the public API surface that consumers are expected to
import from. It re-exports the core configuration models, the main
:class:`Embedder` type, result models, and the custom exception hierarchy.

Example
-------
>>> from embeddy import Embedder, EmbedderConfig
>>> config = EmbedderConfig(model_path="/models/test-model")
>>> embedder = Embedder(config=config)
"""

from embeddy.config import EmbedderConfig, RuntimeConfig, load_config_file
from embeddy.embedder import Embedder
from embeddy.exceptions import (
    EmbeddyError,
    EncodingError,
    ModelLoadError,
    SearchError,
    ValidationError,
)
from embeddy.models import (
    Embedding,
    EmbeddingResult,
    SearchResult,
    SearchResults,
    SimilarityScore,
)

__version__ = "0.2.1"

__all__ = [
    "Embedder",
    "EmbedderConfig",
    "RuntimeConfig",
    "load_config_file",
    "Embedding",
    "EmbeddingResult",
    "SearchResult",
    "SearchResults",
    "SimilarityScore",
    "EmbeddyError",
    "ModelLoadError",
    "EncodingError",
    "ValidationError",
    "SearchError",
    "__version__",
]
