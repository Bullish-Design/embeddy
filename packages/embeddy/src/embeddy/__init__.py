"""embeddy — typed, metric-honest embedding, storage, search, and serving.

Consumes chonkai (one-way dependency). Importable with zero extras
(IMPLEMENTATION_PLAN §2) — heavy providers (local/http) import lazily.
"""

from embeddy.budget import DEFAULT_HEADROOM_TOKENS, chunk_budget
from embeddy.config import (
    DEFAULT_PIPELINE_CONCURRENCY,
    EmbedderSettings,
    PipelineSettings,
    load_pipeline_settings,
    load_settings,
)
from embeddy.errors import (
    EmbeddyError,
    HTTPProviderError,
    ModelNotLoadedError,
    ProviderError,
    ProviderInputError,
    RerankError,
    WrongDimensionError,
)
from embeddy.pipeline import (
    FileEvent,
    FileStatus,
    IngestPipeline,
    IngestStats,
    PipelineError,
    SourceError,
    SourcePhase,
    generate_source_id,
)
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
from embeddy.providers import (
    CrossEncoderReranker,
    FakeProvider,
    HTTPProvider,
    HTTPReranker,
    LocalProvider,
    build_embeddings_request,
    build_provider,
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
from embeddy.search import RRF_DEFAULT_K, fuse_rrf, fuse_weighted, search_hybrid

__version__ = "0.1.0"

__all__ = [
    "CollectionStats",
    "CrossEncoderReranker",
    "DEFAULT_HEADROOM_TOKENS",
    "DEFAULT_MODELS",
    "DEFAULT_PIPELINE_CONCURRENCY",
    "EmbedInput",
    "EmbedderSettings",
    "FileEvent",
    "FileStatus",
    "IngestPipeline",
    "IngestStats",
    "PipelineError",
    "PipelineSettings",
    "SourceError",
    "SourcePhase",
    "generate_source_id",
    "load_pipeline_settings",
    "EmbeddyError",
    "FakeProvider",
    "HTTPProvider",
    "HTTPProviderError",
    "HTTPReranker",
    "ImageInput",
    "LocalProvider",
    "Metric",
    "ModelNotLoadedError",
    "ModelSpec",
    "ProviderError",
    "ProviderInputError",
    "RRF_DEFAULT_K",
    "RegistryError",
    "RerankError",
    "RerankHit",
    "RerankerProvider",
    "ScoredDocument",
    "SearchResult",
    "SourceId",
    "SourceMetadata",
    "StoredChunk",
    "Vector",
    "WrongDimensionError",
    "assert_unit_vector",
    "build_embeddings_request",
    "build_provider",
    "chunk_budget",
    "cosine_similarity",
    "fuse_rrf",
    "fuse_weighted",
    "get_model",
    "load_settings",
    "normalize_l2",
    "resolve_dimension",
    "resolve_instruction",
    "search_hybrid",
    "truncate_and_renormalize",
]
