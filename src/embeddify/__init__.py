# src/embeddify/__init__.py
from __future__ import annotations

"""Public package interface for Embeddify.

This module defines the public API surface that consumers are expected to
import from. It re-exports the core configuration models, the main
:class:`Embedder` type, result models, and the custom exception hierarchy.

Example
-------
>>> from embeddify import Embedder, EmbedderConfig
>>> config = EmbedderConfig(model_path="/models/test-model")
>>> embedder = Embedder(config=config)
"""

from embeddify.config import EmbedderConfig, RuntimeConfig, load_config_file
from embeddify.embedder import Embedder
from embeddify.exceptions import (
    EmbeddifyError,
    EncodingError,
    ModelLoadError,
    SearchError,
    ValidationError,
)
from embeddify.models import (
    Embedding,
    EmbeddingResult,
    SearchResult,
    SearchResults,
    SimilarityScore,
)

__version__ = "0.1.0"

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
    "EmbeddifyError",
    "ModelLoadError",
    "EncodingError",
    "ValidationError",
    "SearchError",
    "__version__",
]
