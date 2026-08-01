"""embeddy — typed, metric-honest embedding, storage, search, and serving.

Consumes chonkai (one-way dependency). Importable with zero extras
(IMPLEMENTATION_PLAN §2) — heavy providers (local/http) import lazily.
"""

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
from embeddy.registry import (
    DEFAULT_MODELS,
    ModelSpec,
    RegistryError,
    get_model,
    resolve_dimension,
    resolve_instruction,
    truncate_and_renormalize,
)
from embeddy.search import RRF_DEFAULT_K, fuse_rrf, fuse_weighted

__version__ = "0.1.0"

__all__ = [
    "CollectionStats",
    "DEFAULT_MODELS",
    "EmbedInput",
    "ImageInput",
    "Metric",
    "ModelSpec",
    "RRF_DEFAULT_K",
    "RegistryError",
    "ScoredDocument",
    "SourceId",
    "SourceMetadata",
    "StoredChunk",
    "Vector",
    "assert_unit_vector",
    "cosine_similarity",
    "fuse_rrf",
    "fuse_weighted",
    "get_model",
    "normalize_l2",
    "resolve_dimension",
    "resolve_instruction",
    "truncate_and_renormalize",
]
