"""Protocol definitions: types + EmbeddingProvider (keystone, M1 draft) +
RerankerProvider (M3 draft)."""

from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.protocol.rerank import RerankerProvider, RerankHit
from embeddy.protocol.types import (
    CollectionStats,
    EmbedInput,
    ImageInput,
    Metric,
    ScoredDocument,
    SourceId,
    SourceMetadata,
    StoredChunk,
    Vector,
    assert_unit_vector,
    cosine_similarity,
    normalize_l2,
)

__all__ = [
    "CollectionStats",
    "EmbedInput",
    "EmbeddingProvider",
    "ImageInput",
    "Metric",
    "RerankHit",
    "RerankerProvider",
    "ScoredDocument",
    "SourceId",
    "SourceMetadata",
    "StoredChunk",
    "Vector",
    "assert_unit_vector",
    "cosine_similarity",
    "normalize_l2",
]
