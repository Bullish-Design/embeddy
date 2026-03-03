# tests/test_package_init.py
"""Tests for the embeddy package's public import surface (v0.3.0)."""

from __future__ import annotations

import embeddy
from embeddy import (
    # Config
    ChunkConfig,
    EmbedderConfig,
    EmbeddyConfig,
    PipelineConfig,
    ServerConfig,
    StoreConfig,
    load_config_file,
    # Embedding layer
    Embedder,
    EmbedderBackend,
    LocalBackend,
    RemoteBackend,
    # Models - enums
    ContentType,
    DistanceMetric,
    FusionStrategy,
    SearchMode,
    # Models - embedding
    EmbedInput,
    Embedding,
    SimilarityScore,
    # Models - ingestion
    IngestResult,
    SourceMetadata,
    # Models - chunks
    Chunk,
    # Models - collections
    Collection,
    CollectionStats,
    # Models - search
    SearchFilters,
    SearchResult,
    SearchResults,
    # Models - pipeline
    IngestStats,
    # Exceptions
    ChunkingError,
    EmbeddyError,
    EncodingError,
    IngestError,
    ModelLoadError,
    SearchError,
    ServerError,
    StoreError,
    ValidationError,
    # Version
    __version__,
)
from embeddy.config import EmbedderConfig as DirectEmbedderConfig
from embeddy.config import EmbeddyConfig as DirectEmbeddyConfig
from embeddy.config import load_config_file as direct_load_config_file
from embeddy.exceptions import EmbeddyError as DirectEmbeddyError
from embeddy.models import Embedding as DirectEmbedding
from embeddy.models import SearchResult as DirectSearchResult


class TestPublicAPI:
    """Tests for the embeddy package's public import surface."""

    def test_top_level_reexports_config_types(self) -> None:
        assert EmbedderConfig is DirectEmbedderConfig
        assert EmbeddyConfig is DirectEmbeddyConfig
        assert load_config_file is direct_load_config_file

    def test_top_level_reexports_model_types(self) -> None:
        assert Embedding is DirectEmbedding
        assert SearchResult is DirectSearchResult

    def test_top_level_reexports_exceptions(self) -> None:
        assert EmbeddyError is DirectEmbeddyError

    def test_version_attribute_is_defined(self) -> None:
        assert isinstance(__version__, str)
        assert __version__ == "0.3.1"

    def test_all_contains_expected_names(self) -> None:
        public_names = set(getattr(embeddy, "__all__", []))
        expected_names = {
            "__version__",
            # Config
            "EmbedderConfig",
            "StoreConfig",
            "ChunkConfig",
            "PipelineConfig",
            "ServerConfig",
            "EmbeddyConfig",
            "load_config_file",
            # Embedding layer
            "Embedder",
            "EmbedderBackend",
            "LocalBackend",
            "RemoteBackend",
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
        }
        missing = expected_names - public_names
        assert not missing, f"Missing names from embeddy.__all__: {sorted(missing)}"

    def test_no_unexpected_exports(self) -> None:
        """__all__ should only contain documented names."""
        public_names = set(getattr(embeddy, "__all__", []))
        expected_names = {
            "__version__",
            "EmbedderConfig",
            "StoreConfig",
            "ChunkConfig",
            "PipelineConfig",
            "ServerConfig",
            "EmbeddyConfig",
            "load_config_file",
            "Embedder",
            "EmbedderBackend",
            "LocalBackend",
            "RemoteBackend",
            "ContentType",
            "SearchMode",
            "FusionStrategy",
            "DistanceMetric",
            "EmbedInput",
            "Embedding",
            "SimilarityScore",
            "SourceMetadata",
            "IngestResult",
            "Chunk",
            "Collection",
            "CollectionStats",
            "SearchFilters",
            "SearchResult",
            "SearchResults",
            "IngestStats",
            "EmbeddyError",
            "ModelLoadError",
            "EncodingError",
            "ValidationError",
            "SearchError",
            "IngestError",
            "StoreError",
            "ChunkingError",
            "ServerError",
        }
        extra = public_names - expected_names
        assert not extra, f"Unexpected names in embeddy.__all__: {sorted(extra)}"
