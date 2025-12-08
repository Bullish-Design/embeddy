from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from embeddify.models import (
    Embedding,
    EmbeddingResult,
    SearchResult,
    SearchResults,
    SimilarityScore,
)


class TestEmbedding:
    def test_embedding_with_list_vector(self) -> None:
        emb = Embedding(vector=[0.1, 0.2, 0.3], model_name="test-model", normalized=True)

        assert emb.model_name == "test-model"
        assert emb.normalized is True
        assert emb.dimensions == 3

    def test_embedding_with_numpy_vector(self) -> None:
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        emb = Embedding(vector=vec, model_name="test-model", normalized=False)

        assert emb.dimensions == 4
        assert isinstance(emb.vector, np.ndarray)

    def test_empty_vector_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            Embedding(vector=[], model_name="test-model", normalized=True)


class TestSimilarityScore:
    def test_similarity_score_default_metric(self) -> None:
        score = SimilarityScore(score=0.5)

        assert math.isclose(score.score, 0.5)
        assert score.metric == "cosine"

    def test_similarity_score_comparisons(self) -> None:
        low = SimilarityScore(score=0.1)
        high = SimilarityScore(score=0.9)

        assert low < high
        assert high > low
        assert low <= high
        assert high >= low
        assert high == SimilarityScore(score=0.9)

    def test_invalid_metric_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            SimilarityScore(score=0.1, metric="euclidean")


class TestEmbeddingResult:
    def test_valid_embedding_result_and_count(self) -> None:
        emb1 = Embedding(vector=[1.0, 2.0], model_name="test", normalized=True)
        emb2 = Embedding(vector=[3.0, 4.0], model_name="test", normalized=True)

        result = EmbeddingResult(embeddings=[emb1, emb2], model_name="test", dimensions=2)

        assert result.count == 2
        assert result.dimensions == 2

    def test_inconsistent_dimensions_raise_validation_error(self) -> None:
        emb1 = Embedding(vector=[1.0, 2.0], model_name="test", normalized=True)
        emb2 = Embedding(vector=[3.0, 4.0, 5.0], model_name="test", normalized=True)

        with pytest.raises(PydanticValidationError):
            EmbeddingResult(embeddings=[emb1, emb2], model_name="test", dimensions=2)


class TestSearchResult:
    def test_valid_search_result(self) -> None:
        result = SearchResult(corpus_id=0, score=0.9, text="hello")

        assert result.corpus_id == 0
        assert math.isclose(result.score, 0.9)
        assert result.text == "hello"

    def test_negative_corpus_id_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            SearchResult(corpus_id=-1, score=0.5, text="invalid")

    def test_non_finite_score_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            SearchResult(corpus_id=0, score=float("nan"), text="invalid")


class TestSearchResults:
    def test_valid_sorted_search_results_and_num_queries(self) -> None:
        hits = [
            SearchResult(corpus_id=0, score=0.9, text="a"),
            SearchResult(corpus_id=1, score=0.7, text="b"),
        ]
        search_results = SearchResults(results=[hits], query_texts=["query"])

        assert search_results.num_queries == 1
        assert len(search_results.results[0]) == 2

    def test_unsorted_results_raise_validation_error(self) -> None:
        hits = [
            SearchResult(corpus_id=0, score=0.5, text="a"),
            SearchResult(corpus_id=1, score=0.8, text="b"),
        ]

        with pytest.raises(PydanticValidationError):
            SearchResults(results=[hits], query_texts=["query"])

