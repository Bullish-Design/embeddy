"""PROTOCOLS — draft-for-M1 design of embeddy/protocol/embedding.py,
embeddy/protocol/search.py and embeddy/registry.py (NOT wired in).

Freeze policy (IMPLEMENTATION_PLAN §12): these protocols are DRAFTED at M1 and
FROZEN at M4. Source ops, filter compilation, rerank and the multimodal path
may still reshape them during Phases 2-4; after the M4 freeze, changes need a
docs/decisions/ record.

Two rules encoded here (CONCEPT §5.1):
  * Role -> instruction resolution happens in the CALLER (registry look-up);
    providers only ever receive a RESOLVED instruction string.
  * resolve_dimension is the single MRL policy point (CONCEPT §5.2):
    None -> native; non-MRL + wrong dim -> ValidationError; truncation is
    ALWAYS followed by L2 re-normalization (a sliced vector is not unit-norm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core_types import EmbedInput, Metric, ScoredDocument, SourceId, SourceMetadata, Vector


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model fact entry. Facts come from the registry, never from config
    (CONCEPT §5.1). `mrl_range` is a half-open range over valid MRL dims."""

    id: str
    native_dimension: int
    mrl_range: range | None       # None = not MRL-capable
    context_length: int
    instructions: dict[str, str]  # role -> resolved prompt string
    license: str = ""

    def __post_init__(self) -> None:
        if self.native_dimension < 1:
            raise ValueError("native_dimension must be >= 1")
        if self.mrl_range is not None:
            if not (
                self.mrl_range.step == 1
                and self.mrl_range.start >= 1
                and self.mrl_range.stop > self.mrl_range.start
            ):
                raise ValueError(
                    "mrl_range must be an ascending half-open range (step 1)"
                )
            if self.native_dimension not in self.mrl_range:
                raise ValueError(
                    "native_dimension must lie inside mrl_range"
                )


class RegistryError(ValueError):
    """Unknown model id / invalid dimension request."""


class _Registry:
    """Tiny registry; the real one (Phase 3) is a module dict + load hook."""

    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        self._models[spec.id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError:
            raise RegistryError(f"unknown model: {model_id!r}") from None

    def resolve_instruction(self, model_id: str, role: str) -> str:
        """The ONLY place role -> instruction is resolved (CONCEPT §5.1).

        The pipeline/search/server call this and hand the RESULT string to
        the provider. Providers never see a role and can never mix document
        vs query instructions (the H2 bug class).
        """
        spec = self.get(model_id)
        try:
            return spec.instructions[role]
        except KeyError:
            raise RegistryError(
                f"model {model_id!r} has no instruction for role {role!r}"
            ) from None


def resolve_dimension(spec: ModelSpec, requested: int | None) -> int:
    """CONCEPT §5.2 — exact logic. `requested is None` -> native dimension."""
    if requested is None:
        return spec.native_dimension
    if spec.mrl_range is None:
        if requested != spec.native_dimension:
            raise ValueError(
                f"{spec.id} is not MRL-capable; embedding_dimension must be "
                f"{spec.native_dimension}"
            )
        return requested
    if requested not in spec.mrl_range:
        raise ValueError(
            f"{spec.id} supports MRL {spec.mrl_range.start}-"
            f"{spec.mrl_range.stop - 1}; got {requested}"
        )
    return requested


def truncate_and_renormalize(vector: Vector, dim: int) -> Vector:
    """MRL truncation followed by L2 re-normalization (CONCEPT §5.2).

    A sliced MRL vector is not unit-norm; cosine scoring assumes unit
    vectors, so the slice must be re-normalized. This is the ONE truncation
    point — the provider post-process calls it keyed on model facts.
    """
    import numpy as np

    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D vector, got shape {arr.shape}")
    if dim > arr.shape[0]:
        raise ValueError(
            f"cannot truncate {arr.shape[0]}-dim vector to {dim} dims"
        )
    if dim == arr.shape[0]:
        return arr
    return normalize(arr[:dim])


def normalize(vec) -> Vector:
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"cannot normalize zero/NaN vector (norm={norm})")
    return (arr / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# EmbeddingProvider — the keystone protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A model adapter. Caller resolves role -> instruction and passes the
    resolved string; the provider is deliberately dumb about roles."""

    dimension: int            # RESOLVED dimension (native or MRL-truncated)
    context_length: int       # drives the chunk budget (chonkai ChunkBudget)
    model_name: str

    async def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[Vector]:
        """Embed `inputs`, returning unit-norm float32 vectors of
        `self.dimension` dims (assert_unit_vector/assert_dim guards live in
        the pipeline wrapper, not in each adapter)."""
        ...


# ---------------------------------------------------------------------------
# RerankerProvider
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerankHit:
    """One reranked hit: index into the input document list + score."""

    index: int
    score: float
    metric: Metric = Metric.RERANK


@runtime_checkable
class RerankerProvider(Protocol):
    """Optional post-fusion stage (CONCEPT §5.3). Uses the TEI/Jina rerank
    shape on the wire; there is no OpenAI /v1/rerank standard (CONCEPT §7)."""

    model_name: str

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        instruction: str | None = None,  # already resolved by the caller
    ) -> list[RerankHit]:
        ...


# ---------------------------------------------------------------------------
# Searchable — storage protocol INCLUDING source operations (CONCEPT §5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Compiled-to-SQL pre-filters (never post-filter over-fetch — fixes M3).

    `metadata_match` is a list of (field, value) pairs; backends compile it
    into the WHERE clause of the pre-filter join. Typed values (str) — no
    dict[str, Any]. Expected to grow (range filters) before the M4 freeze.
    """

    content_types: tuple[str, ...] = ()
    source_path_prefix: str | None = None
    chunk_types: tuple[str, ...] = ()
    metadata_match: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, **pairs: str) -> "SearchFilters":
        return cls(metadata_match=tuple(sorted(pairs.items())))

    def is_empty(self) -> bool:
        return not (
            self.content_types or self.source_path_prefix
            or self.chunk_types or self.metadata_match
        )


@runtime_checkable
class Searchable(Protocol):
    """Every storage backend implements THIS contract — sqlite-vec+FTS5 is
    the default, Qdrant the scale path. Source operations are part of the
    protocol so the Qdrant adapter has a defined contract (not a sqlite-only
    side layer). All methods are async; backends own their connections."""

    # --- chunks ----------------------------------------------------------
    async def add(
        self,
        collection: str,
        chunks: list[StoredChunk],
        vectors: list[Vector],
    ) -> None: ...
    async def delete(self, collection: str, chunk_ids: list[str]) -> None: ...

    # --- search ----------------------------------------------------------
    async def search_vector(
        self,
        collection: str,
        query_vector: Vector,
        filters: SearchFilters,
        top_k: int,
    ) -> list[ScoredDocument]: ...
    async def search_fts(
        self,
        collection: str,
        query: str,
        filters: SearchFilters,
        top_k: int,
    ) -> list[ScoredDocument]: ...

    # --- collections -----------------------------------------------------
    async def stats(self, collection: str) -> object: ...  # CollectionStats

    # --- sources (first-class — CONCEPT §3.3) ----------------------------
    async def upsert_source(
        self, collection: str, source: SourceMetadata
    ) -> SourceId: ...
    async def get_source(
        self, collection: str, path: str
    ) -> SourceMetadata | None: ...
    async def reindex_source(
        self,
        collection: str,
        source: SourceMetadata,
        chunks: list[StoredChunk],
        vectors: list[Vector],
    ) -> None:
        """Atomic swap of a source's chunk set in ONE transaction (fixes H7):
        on failure the old chunks must remain intact and queryable."""
        ...
    async def delete_source(
        self, collection: str, source_id: SourceId
    ) -> None:
        """Cascade-deletes the source's chunks (chunks.source_id FK
        ON DELETE CASCADE in the sqlite schema)."""
        ...
    async def list_sources(self, collection: str) -> list[SourceMetadata]: ...
