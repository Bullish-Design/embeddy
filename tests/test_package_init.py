# tests/test_package_init.py
from __future__ import annotations

from typing import Any

import embeddy
from embeddy import (
    Embedding as PublicEmbedding,
    EmbeddingResult as PublicEmbeddingResult,
    Embedder as PublicEmbedder,
    EmbedderConfig as PublicEmbedderConfig,
    EmbeddyError as PublicEmbeddyError,
    EncodingError as PublicEncodingError,
    ModelLoadError as PublicModelLoadError,
    RuntimeConfig as PublicRuntimeConfig,
    SearchError as PublicSearchError,
    SearchResult as PublicSearchResult,
    SearchResults as PublicSearchResults,
    SimilarityScore as PublicSimilarityScore,
    ValidationError as PublicValidationError,
    load_config_file as public_load_config_file,
    __version__ as public_version,
)
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


class TestPublicAPI:
    """Tests for the embeddy package's public import surface."""

    def test_top_level_reexports_core_types(self) -> None:
        """The root package should re-export the main public API types."""
        assert PublicEmbedder is Embedder
        assert PublicEmbedderConfig is EmbedderConfig
        assert PublicRuntimeConfig is RuntimeConfig
        assert PublicEmbedding is Embedding
        assert PublicEmbeddingResult is EmbeddingResult
        assert PublicSearchResult is SearchResult
        assert PublicSearchResults is SearchResults
        assert PublicSimilarityScore is SimilarityScore

    def test_top_level_reexports_exceptions_and_helpers(self) -> None:
        """Exceptions and helper functions should also be available at package root."""
        assert PublicEmbeddyError is EmbeddyError
        assert PublicModelLoadError is ModelLoadError
        assert PublicEncodingError is EncodingError
        assert PublicValidationError is ValidationError
        assert PublicSearchError is SearchError
        assert public_load_config_file is load_config_file

    def test_version_attribute_is_defined(self) -> None:
        """The package must expose a semantic version string via __version__."""
        assert isinstance(public_version, str)
        # Keep this in sync with SPEC.md version.
        assert public_version == "0.1.0"

    def test_all_contains_public_names(self) -> None:
        """The __all__ attribute should list the documented public API symbols."""
        public_names = set(getattr(embeddy, "__all__", []))
        expected_names = {
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
        }
        # All expected names must be present, but __all__ may contain extras if
        # we decide to expand the public API in the future.
        missing: set[str] = expected_names - public_names
        assert not missing, f"Missing names from embeddy.__all__: {sorted(missing)}"

