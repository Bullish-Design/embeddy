# src/embeddy/__init__.py
"""Public package interface for embeddy.

Re-exports the core types, configuration, and exceptions that consumers
are expected to import from the top-level package.
"""

from __future__ import annotations

from embeddy.config import (
    ChunkConfig,
    EmbedderConfig,
    EmbeddyConfig,
    PipelineConfig,
    ServerConfig,
    StoreConfig,
    load_config_file,
)
from embeddy.exceptions import (
    ChunkingError,
    EmbeddyError,
    EncodingError,
    IngestError,
    ModelLoadError,
    SearchError,
    ServerError,
    StoreError,
    ValidationError,
)
from embeddy.models import (
    Chunk,
    Collection,
    CollectionStats,
    ContentType,
    DistanceMetric,
    EmbedInput,
    Embedding,
    FusionStrategy,
    IngestResult,
    IngestStats,
    SearchFilters,
    SearchMode,
    SearchResult,
    SearchResults,
    SimilarityScore,
    SourceMetadata,
)

__version__ = "0.3.0"

__all__ = [
    # Version
    "__version__",
    # Config
    "EmbedderConfig",
    "StoreConfig",
    "ChunkConfig",
    "PipelineConfig",
    "ServerConfig",
    "EmbeddyConfig",
    "load_config_file",
    # Models - enums
    "ContentType",
    "SearchMode",
    "FusionStrategy",
    "DistanceMetric",
    # Models - embedding
    "EmbedInput",
    "Embedding",
    "SimilarityScore",
    # Models - ingestion
    "SourceMetadata",
    "IngestResult",
    # Models - chunks
    "Chunk",
    # Models - collections
    "Collection",
    "CollectionStats",
    # Models - search
    "SearchFilters",
    "SearchResult",
    "SearchResults",
    # Models - pipeline
    "IngestStats",
    # Exceptions
    "EmbeddyError",
    "ModelLoadError",
    "EncodingError",
    "ValidationError",
    "SearchError",
    "IngestError",
    "StoreError",
    "ChunkingError",
    "ServerError",
]
