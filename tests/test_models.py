# tests/test_models.py
"""Tests for all embeddy data models (Phase 1).

Covers: ContentType, SearchMode, FusionStrategy, DistanceMetric, EmbedInput,
Embedding, SimilarityScore, SourceMetadata, IngestResult, Chunk, Collection,
CollectionStats, SearchFilters, SearchResult, SearchResults, IngestError (model),
IngestStats.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from embeddy.models import (
    Chunk,
    Collection,
    CollectionStats,
    ContentType,
    DistanceMetric,
    EmbedInput,
    Embedding,
    FusionStrategy,
    IngestError,
    IngestResult,
    IngestStats,
    SearchFilters,
    SearchMode,
    SearchResult,
    SearchResults,
    SimilarityScore,
    SourceMetadata,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestContentType:
    def test_values(self) -> None:
        assert ContentType.PYTHON == "python"
        assert ContentType.MARKDOWN == "markdown"
        assert ContentType.DOCLING == "docling"
        assert ContentType.GENERIC == "generic"

    def test_all_members_are_strings(self) -> None:
        for member in ContentType:
            assert isinstance(member.value, str)


class TestSearchMode:
    def test_values(self) -> None:
        assert SearchMode.VECTOR == "vector"
        assert SearchMode.FULLTEXT == "fulltext"
        assert SearchMode.HYBRID == "hybrid"


class TestFusionStrategy:
    def test_values(self) -> None:
        assert FusionStrategy.RRF == "rrf"
        assert FusionStrategy.WEIGHTED == "weighted"


class TestDistanceMetric:
    def test_values(self) -> None:
        assert DistanceMetric.COSINE == "cosine"
        assert DistanceMetric.DOT == "dot"


# ---------------------------------------------------------------------------
# EmbedInput
# ---------------------------------------------------------------------------


class TestEmbedInput:
    def test_text_only(self) -> None:
        inp = EmbedInput(text="hello world")
        assert inp.text == "hello world"
        assert inp.image is None
        assert inp.video is None

    def test_image_only(self) -> None:
        inp = EmbedInput(image="/path/to/image.png")
        assert inp.image == "/path/to/image.png"

    def test_video_only(self) -> None:
        inp = EmbedInput(video="/path/to/video.mp4")
        assert inp.video == "/path/to/video.mp4"

    def test_multimodal(self) -> None:
        inp = EmbedInput(text="caption", image="/img.png")
        assert inp.text == "caption"
        assert inp.image == "/img.png"

    def test_with_instruction(self) -> None:
        inp = EmbedInput(text="query", instruction="Find documents about X")
        assert inp.instruction == "Find documents about X"

    def test_no_content_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError, match="At least one"):
            EmbedInput()

    def test_all_none_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError, match="At least one"):
            EmbedInput(text=None, image=None, video=None)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_embedding_with_list_vector(self) -> None:
        emb = Embedding(vector=[0.1, 0.2, 0.3], model_name="test-model", normalized=True)
        assert emb.model_name == "test-model"
        assert emb.normalized is True
        assert emb.dimension == 3

    def test_embedding_with_numpy_vector(self) -> None:
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        emb = Embedding(vector=vec, model_name="test-model", normalized=False)
        assert emb.dimension == 4
        assert isinstance(emb.vector, np.ndarray)

    def test_empty_vector_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            Embedding(vector=[], model_name="test-model", normalized=True)

    def test_empty_numpy_vector_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            Embedding(vector=np.array([], dtype=float), model_name="test-model")

    def test_to_list_from_numpy(self) -> None:
        vec = np.array([1.0, 2.0, 3.0])
        emb = Embedding(vector=vec, model_name="test")
        result = emb.to_list()
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_to_list_from_list(self) -> None:
        emb = Embedding(vector=[1.0, 2.0], model_name="test")
        assert emb.to_list() == [1.0, 2.0]

    def test_default_input_type(self) -> None:
        emb = Embedding(vector=[1.0], model_name="test")
        assert emb.input_type == "text"

    def test_custom_input_type(self) -> None:
        emb = Embedding(vector=[1.0], model_name="test", input_type="image")
        assert emb.input_type == "image"


# ---------------------------------------------------------------------------
# SimilarityScore
# ---------------------------------------------------------------------------


class TestSimilarityScore:
    def test_default_metric(self) -> None:
        score = SimilarityScore(score=0.5)
        assert math.isclose(score.score, 0.5)
        assert score.metric == "cosine"

    def test_comparisons(self) -> None:
        low = SimilarityScore(score=0.1)
        high = SimilarityScore(score=0.9)
        assert low < high
        assert high > low
        assert low <= high
        assert high >= low
        assert high == SimilarityScore(score=0.9)

    def test_comparison_with_float(self) -> None:
        score = SimilarityScore(score=0.5)
        assert score < 0.6
        assert score > 0.4
        assert score == 0.5

    def test_invalid_metric_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            SimilarityScore(score=0.1, metric="euclidean")

    def test_dot_metric(self) -> None:
        score = SimilarityScore(score=42.0, metric="dot")
        assert score.metric == "dot"


# ---------------------------------------------------------------------------
# SourceMetadata
# ---------------------------------------------------------------------------


class TestSourceMetadata:
    def test_defaults_all_none(self) -> None:
        meta = SourceMetadata()
        assert meta.file_path is None
        assert meta.url is None
        assert meta.size_bytes is None
        assert meta.modified_at is None
        assert meta.content_hash is None

    def test_with_values(self) -> None:
        now = datetime.now()
        meta = SourceMetadata(
            file_path="/foo/bar.py",
            size_bytes=1024,
            modified_at=now,
            content_hash="abc123",
        )
        assert meta.file_path == "/foo/bar.py"
        assert meta.size_bytes == 1024
        assert meta.modified_at == now
        assert meta.content_hash == "abc123"


# ---------------------------------------------------------------------------
# IngestResult
# ---------------------------------------------------------------------------


class TestIngestResult:
    def test_minimal(self) -> None:
        result = IngestResult(text="hello", content_type=ContentType.GENERIC)
        assert result.text == "hello"
        assert result.content_type == ContentType.GENERIC
        assert result.source is not None
        assert result.docling_document is None

    def test_with_source(self) -> None:
        source = SourceMetadata(file_path="/test.py")
        result = IngestResult(text="code", content_type=ContentType.PYTHON, source=source)
        assert result.source.file_path == "/test.py"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_minimal(self) -> None:
        chunk = Chunk(content="hello world", content_type=ContentType.GENERIC)
        assert chunk.content == "hello world"
        assert chunk.content_type == ContentType.GENERIC
        assert chunk.chunk_type == "generic"
        assert chunk.id  # auto-generated UUID

    def test_with_metadata(self) -> None:
        chunk = Chunk(
            content="def foo(): pass",
            content_type=ContentType.PYTHON,
            chunk_type="function",
            name="foo",
            start_line=1,
            end_line=1,
            metadata={"docstring": "none"},
        )
        assert chunk.name == "foo"
        assert chunk.start_line == 1
        assert chunk.metadata["docstring"] == "none"

    def test_empty_content_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError, match="empty"):
            Chunk(content="", content_type=ContentType.GENERIC)

    def test_whitespace_only_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError, match="empty"):
            Chunk(content="   \n  ", content_type=ContentType.GENERIC)

    def test_parent_field(self) -> None:
        chunk = Chunk(content="method body", content_type=ContentType.PYTHON, parent="MyClass")
        assert chunk.parent == "MyClass"

    def test_unique_ids(self) -> None:
        c1 = Chunk(content="a", content_type=ContentType.GENERIC)
        c2 = Chunk(content="b", content_type=ContentType.GENERIC)
        assert c1.id != c2.id


# ---------------------------------------------------------------------------
# Collection & CollectionStats
# ---------------------------------------------------------------------------


class TestCollection:
    def test_minimal(self) -> None:
        col = Collection(name="test", dimension=2048, model_name="test-model")
        assert col.name == "test"
        assert col.dimension == 2048
        assert col.distance_metric == DistanceMetric.COSINE
        assert col.id  # auto-generated

    def test_custom_distance_metric(self) -> None:
        col = Collection(name="test", dimension=512, model_name="m", distance_metric=DistanceMetric.DOT)
        assert col.distance_metric == DistanceMetric.DOT

    def test_empty_name_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            Collection(name="", dimension=2048, model_name="m")

    def test_zero_dimension_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            Collection(name="test", dimension=0, model_name="m")

    def test_negative_dimension_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            Collection(name="test", dimension=-1, model_name="m")


class TestCollectionStats:
    def test_defaults(self) -> None:
        stats = CollectionStats(name="test")
        assert stats.name == "test"
        assert stats.chunk_count == 0
        assert stats.source_count == 0
        assert stats.storage_bytes is None


# ---------------------------------------------------------------------------
# SearchFilters
# ---------------------------------------------------------------------------


class TestSearchFilters:
    def test_defaults_all_none(self) -> None:
        filters = SearchFilters()
        assert filters.content_types is None
        assert filters.source_path_prefix is None
        assert filters.chunk_types is None
        assert filters.metadata_match is None

    def test_with_content_types(self) -> None:
        filters = SearchFilters(content_types=[ContentType.PYTHON, ContentType.MARKDOWN])
        assert filters.content_types == [ContentType.PYTHON, ContentType.MARKDOWN]

    def test_with_metadata_match(self) -> None:
        filters = SearchFilters(metadata_match={"language": "python"})
        assert filters.metadata_match == {"language": "python"}


# ---------------------------------------------------------------------------
# SearchResult & SearchResults
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_valid(self) -> None:
        result = SearchResult(chunk_id="abc", content="hello", score=0.9)
        assert result.chunk_id == "abc"
        assert result.content == "hello"
        assert math.isclose(result.score, 0.9)

    def test_with_metadata(self) -> None:
        result = SearchResult(
            chunk_id="abc",
            content="hello",
            score=0.5,
            source_path="/foo.py",
            content_type="python",
            chunk_type="function",
            name="foo",
            start_line=10,
            end_line=20,
        )
        assert result.source_path == "/foo.py"
        assert result.name == "foo"

    def test_non_finite_score_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            SearchResult(chunk_id="abc", content="hello", score=float("nan"))

    def test_infinite_score_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            SearchResult(chunk_id="abc", content="hello", score=float("inf"))


class TestSearchResults:
    def test_valid_sorted(self) -> None:
        hits = [
            SearchResult(chunk_id="a", content="a", score=0.9),
            SearchResult(chunk_id="b", content="b", score=0.7),
            SearchResult(chunk_id="c", content="c", score=0.5),
        ]
        results = SearchResults(results=hits, query="test query", collection="default", total_results=3)
        assert len(results.results) == 3
        assert results.query == "test query"
        assert results.total_results == 3

    def test_unsorted_results_raise_validation_error(self) -> None:
        hits = [
            SearchResult(chunk_id="a", content="a", score=0.5),
            SearchResult(chunk_id="b", content="b", score=0.8),
        ]
        with pytest.raises(PydanticValidationError, match="sorted"):
            SearchResults(results=hits, query="test")

    def test_empty_results(self) -> None:
        results = SearchResults()
        assert results.results == []
        assert results.query == ""
        assert results.mode == SearchMode.HYBRID

    def test_default_mode_is_hybrid(self) -> None:
        results = SearchResults()
        assert results.mode == SearchMode.HYBRID


# ---------------------------------------------------------------------------
# IngestError (model) & IngestStats
# ---------------------------------------------------------------------------


class TestIngestErrorModel:
    def test_minimal(self) -> None:
        err = IngestError(error="something broke")
        assert err.error == "something broke"
        assert err.file_path is None
        assert err.error_type == ""

    def test_with_file_path(self) -> None:
        err = IngestError(file_path="/bad/file.pdf", error="parse error", error_type="ParseError")
        assert err.file_path == "/bad/file.pdf"
        assert err.error_type == "ParseError"


class TestIngestStats:
    def test_defaults(self) -> None:
        stats = IngestStats()
        assert stats.files_processed == 0
        assert stats.chunks_created == 0
        assert stats.chunks_embedded == 0
        assert stats.chunks_stored == 0
        assert stats.chunks_skipped == 0
        assert stats.errors == []
        assert stats.elapsed_seconds == 0.0

    def test_with_errors(self) -> None:
        errs = [IngestError(error="fail1"), IngestError(error="fail2")]
        stats = IngestStats(files_processed=5, chunks_created=10, errors=errs)
        assert stats.files_processed == 5
        assert len(stats.errors) == 2
