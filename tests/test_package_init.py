# tests/test_package_init.py
"""Tests for the embeddy package's public import surface (v0.3.11)."""

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
    # Chunking layer
    BaseChunker,
    PythonChunker,
    MarkdownChunker,
    ParagraphChunker,
    TokenWindowChunker,
    DoclingChunker,
    get_chunker,
    # Store layer
    VectorStore,
    # Ingest layer
    Ingestor,
    compute_content_hash,
    detect_content_type,
    is_docling_path,
    # Pipeline layer
    Pipeline,
    # Search layer
    SearchService,
    # Server layer
    create_app,
    # Client layer
    EmbeddyClient,
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
from embeddy.chunking import BaseChunker as DirectBaseChunker
from embeddy.chunking import get_chunker as direct_get_chunker
from embeddy.store import VectorStore as DirectVectorStore
from embeddy.ingest import Ingestor as DirectIngestor
from embeddy.ingest import compute_content_hash as direct_compute_content_hash
from embeddy.ingest import detect_content_type as direct_detect_content_type
from embeddy.ingest import is_docling_path as direct_is_docling_path
from embeddy.pipeline import Pipeline as DirectPipeline
from embeddy.search import SearchService as DirectSearchService
from embeddy.server import create_app as direct_create_app
from embeddy.client import EmbeddyClient as DirectEmbeddyClient


class TestPublicAPI:
    """Tests for the embeddy package's public import surface."""

    def test_top_level_reexports_config_types(self) -> None:
        assert EmbedderConfig is DirectEmbedderConfig
        assert EmbeddyConfig is DirectEmbeddyConfig
        assert load_config_file is direct_load_config_file

    def test_top_level_reexports_model_types(self) -> None:
        assert Embedding is DirectEmbedding
        assert SearchResult is DirectSearchResult

    def test_top_level_reexports_store(self) -> None:
        assert VectorStore is DirectVectorStore

    def test_top_level_reexports_ingest(self) -> None:
        assert Ingestor is DirectIngestor
        assert compute_content_hash is direct_compute_content_hash
        assert detect_content_type is direct_detect_content_type
        assert is_docling_path is direct_is_docling_path

    def test_top_level_reexports_pipeline(self) -> None:
        assert Pipeline is DirectPipeline

    def test_top_level_reexports_search_service(self) -> None:
        assert SearchService is DirectSearchService

    def test_top_level_reexports_server(self) -> None:
        assert create_app is direct_create_app

    def test_top_level_reexports_client(self) -> None:
        assert EmbeddyClient is DirectEmbeddyClient

    def test_top_level_reexports_chunking(self) -> None:
        assert BaseChunker is DirectBaseChunker
        assert get_chunker is direct_get_chunker

    def test_top_level_reexports_exceptions(self) -> None:
        assert EmbeddyError is DirectEmbeddyError

    def test_version_attribute_is_defined(self) -> None:
        assert isinstance(__version__, str)
        assert __version__ == "0.3.11"

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
            # Chunking layer
            "BaseChunker",
            "PythonChunker",
            "MarkdownChunker",
            "ParagraphChunker",
            "TokenWindowChunker",
            "DoclingChunker",
            "get_chunker",
            # Store layer
            "VectorStore",
            # Ingest layer
            "Ingestor",
            "compute_content_hash",
            "detect_content_type",
            "is_docling_path",
            # Pipeline layer
            "Pipeline",
            # Search layer
            "SearchService",
            # Server layer
            "create_app",
            # Client layer
            "EmbeddyClient",
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
            "BaseChunker",
            "PythonChunker",
            "MarkdownChunker",
            "ParagraphChunker",
            "TokenWindowChunker",
            "DoclingChunker",
            "get_chunker",
            "VectorStore",
            "Ingestor",
            "compute_content_hash",
            "detect_content_type",
            "is_docling_path",
            "Pipeline",
            "SearchService",
            "create_app",
            "EmbeddyClient",
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
