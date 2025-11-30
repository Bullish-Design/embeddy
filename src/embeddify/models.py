# src/embeddify/models.py
from __future__ import annotations

from typing import Any

import math
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Embedding(BaseModel):
    """Represents a single embedding vector with associated metadata.

    Attributes:
        vector: The numeric embedding vector as a list of floats or numpy array.
        model_name: Identifier of the model that produced the embedding.
        normalized: Whether the vector has been L2-normalized.
        text: Optional original source text that was encoded.
    """

    model_config = {"arbitrary_types_allowed": True}

    vector: list[float] | np.ndarray = Field(description="Embedding vector representation")
    model_name: str = Field(description="Name of the model used to generate this embedding")
    normalized: bool = Field(description="Whether the vector is L2-normalized")
    text: str | None = Field(default=None, description="Optional original text that produced this embedding")

    @field_validator("vector")
    @classmethod
    def validate_vector_not_empty(cls, value: list[float] | np.ndarray) -> list[float] | np.ndarray:
        """Ensure the embedding vector is not empty."""
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("Embedding vector cannot be empty")
        else:
            if len(value) == 0:
                raise ValueError("Embedding vector cannot be empty")
        return value

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vector."""
        if isinstance(self.vector, np.ndarray):
            if self.vector.ndim == 0:
                return 0
            return int(self.vector.shape[-1])
        return len(self.vector)


class SimilarityScore(BaseModel):
    """Represents the similarity between two embeddings.

    The score is typically in the range [-1, 1] for cosine similarity, but this
    is not enforced so that other metrics can be supported in the future.
    """

    score: float = Field(description="Numeric similarity value")
    metric: str = Field(default="cosine", description="Similarity metric identifier (e.g. 'cosine', 'dot')")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        """Validate that the metric name is supported."""
        allowed = {"cosine", "dot"}
        if value not in allowed:
            raise ValueError(f"Invalid similarity metric '{value}'. Must be one of {sorted(allowed)}")
        return value

    def _other_score(self, other: Any) -> float:
        if isinstance(other, SimilarityScore):
            return other.score
        if isinstance(other, (int, float)):
            return float(other)
        return NotImplemented  # type: ignore[return-value]

    def __lt__(self, other: Any) -> bool:
        other_score = self._other_score(other)
        if other_score is NotImplemented:  # type: ignore[comparison-overlap]
            return NotImplemented  # type: ignore[return-value]
        return self.score < other_score

    def __le__(self, other: Any) -> bool:
        other_score = self._other_score(other)
        if other_score is NotImplemented:  # type: ignore[comparison-overlap]
            return NotImplemented  # type: ignore[return-value]
        return self.score <= other_score

    def __gt__(self, other: Any) -> bool:
        other_score = self._other_score(other)
        if other_score is NotImplemented:  # type: ignore[comparison-overlap]
            return NotImplemented  # type: ignore[return-value]
        return self.score > other_score

    def __ge__(self, other: Any) -> bool:
        other_score = self._other_score(other)
        if other_score is NotImplemented:  # type: ignore[comparison-overlap]
            return NotImplemented  # type: ignore[return-value]
        return self.score >= other_score

    def __eq__(self, other: Any) -> bool:  # type: ignore[override]
        other_score = self._other_score(other)
        if other_score is NotImplemented:  # type: ignore[comparison-overlap]
            return NotImplemented  # type: ignore[return-value]
        return self.score == other_score


class EmbeddingResult(BaseModel):
    """Batch of embeddings produced in a single encode operation.

    Attributes:
        embeddings: The list of individual embedding objects.
        model_name: Name of the model used for encoding.
        dimensions: Expected dimensionality for all embeddings.
    """

    model_config = {"arbitrary_types_allowed": True}

    embeddings: list[Embedding] = Field(default_factory=list, description="Embeddings in this batch")
    model_name: str = Field(description="Model used to produce all embeddings")
    dimensions: int = Field(description="Dimensionality of every embedding vector")

    @model_validator(mode="after")
    def validate_dimensions(self) -> "EmbeddingResult":
        """Ensure all embeddings share the same dimensionality as `dimensions`."""
        if not self.embeddings:
            return self
        if self.dimensions <= 0:
            raise ValueError("EmbeddingResult.dimensions must be a positive integer")

        for index, embedding in enumerate(self.embeddings):
            emb_dim = embedding.dimensions
            if emb_dim != self.dimensions:
                raise ValueError(
                    f"Inconsistent embedding dimensions at index {index}: "
                    f"expected {self.dimensions}, got {emb_dim}"
                )
        return self

    @property
    def count(self) -> int:
        """Return the number of embeddings in this result."""
        return len(self.embeddings)


class SearchResult(BaseModel):
    """Represents a single semantic search hit."""

    corpus_id: int = Field(description="Index of the matching item in the original corpus")
    score: float = Field(description="Similarity score for this match")
    text: str | None = Field(default=None, description="Optional corpus text corresponding to the hit")

    @field_validator("corpus_id")
    @classmethod
    def validate_corpus_id(cls, value: int) -> int:
        """Ensure the corpus index is non-negative."""
        if value < 0:
            raise ValueError("SearchResult.corpus_id must be non-negative")
        return value

    @field_validator("score")
    @classmethod
    def validate_score_finite(cls, value: float) -> float:
        """Ensure the similarity score is a finite float."""
        if not math.isfinite(value):
            raise ValueError("SearchResult.score must be a finite float")
        return value


class SearchResults(BaseModel):
    """Container for semantic search results for one or more queries."""

    results: list[list[SearchResult]] = Field(
        default_factory=list,
        description="Outer list is per-query, inner list contains hits for that query",
    )
    query_texts: list[str] | None = Field(
        default=None,
        description="Optional list of original query texts, aligned with `results`",
    )

    @model_validator(mode="after")
    def validate_sorted_results(self) -> "SearchResults":
        """Ensure that each query's results are sorted by score descending."""
        for query_index, query_results in enumerate(self.results):
            if len(query_results) < 2:
                continue
            scores = [hit.score for hit in query_results]
            if scores != sorted(scores, reverse=True):
                raise ValueError(
                    f"Search results for query index {query_index} must be sorted "
                    f"by score in descending order"
                )
        return self

    @property
    def num_queries(self) -> int:
        """Return the number of queries represented in this result set."""
        return len(self.results)
