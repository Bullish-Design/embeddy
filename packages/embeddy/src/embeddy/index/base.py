"""Searchable — the storage protocol, INCLUDING source operations.

Ported from the Phase-0 spike (`spikes/protocols.py`). Drafted at M1, FROZEN
at M4 (IMPLEMENTATION_PLAN §12). Every backend implements THIS contract —
sqlite-vec+FTS5 is the default, Qdrant the scale path. Source operations are
part of the protocol so the Qdrant adapter has a defined contract (not a
sqlite-only side layer — CONCEPT §3.3, §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embeddy.protocol.types import (
    CollectionStats,
    ScoredDocument,
    SourceId,
    SourceMetadata,
    StoredChunk,
    Vector,
)


class StoreError(RuntimeError):
    """A store-level operation failed (unknown collection, dimension
    mismatch, missing source, ...). Shared by EVERY Searchable backend
    (sqlite and Qdrant raise the same class so the server's 404/400
    mapping is backend-agnostic — moved to the protocol module at Phase 8
    so the Qdrant adapter does not depend on the sqlite module)."""


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """One collection row's metadata (the server's GET /api/v1/collections).

    A store EXTRA record (like the create_collection/count_fts methods) —
    NOT part of the frozen `Searchable` protocol; the server owns
    collection lifecycle and reads this through the store's
    `list_collections` extra. Defined on the protocol module (Phase 8) so
    every backend's `list_collections` returns the same record.
    """

    collection_id: str
    vector_dimension: int  # the RESOLVED dimension the collection stores


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
    def from_mapping(cls, **pairs: str) -> SearchFilters:
        return cls(metadata_match=tuple(sorted(pairs.items())))

    def is_empty(self) -> bool:
        return not (
            self.content_types or self.source_path_prefix or self.chunk_types or self.metadata_match
        )


@runtime_checkable
class Searchable(Protocol):
    """Storage contract. All methods async; backends own their connections.

    M4-converged signatures (frozen at M4, IMPLEMENTATION_PLAN §12):
      * `min_score` on both search methods — a score threshold compared
        against the metric's TRUE semantics (fixes C3): for cosine the
        threshold applies to scores in [0, 1] (higher = better), for BM25 it
        applies to the FTS5 rank (<= 0, higher/less-negative = better). A
        `min_score` is NEVER comparable across metrics (CONCEPT §3.4).
      * `raw` on `search_fts` — the plan §14 opt-in: raw=True passes the
        query string to FTS5 verbatim (caller owns metacharacter safety);
        the default sanitizer quote-wraps tokens and ANDs them.
    """

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
        *,
        min_score: float | None = None,
    ) -> list[ScoredDocument]: ...

    async def search_fts(
        self,
        collection: str,
        query: str,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
        raw: bool = False,
    ) -> list[ScoredDocument]: ...

    # --- collections -----------------------------------------------------
    async def stats(self, collection: str) -> CollectionStats: ...

    # --- sources (first-class — CONCEPT §3.3) ----------------------------
    async def upsert_source(self, collection: str, source: SourceMetadata) -> SourceId: ...
    async def get_source(self, collection: str, path: str) -> SourceMetadata | None: ...
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

    async def delete_source(self, collection: str, source_id: SourceId) -> None:
        """Cascade-deletes the source's chunks (chunks.source_id FK
        ON DELETE CASCADE in the sqlite schema)."""
        ...

    async def list_sources(self, collection: str) -> list[SourceMetadata]: ...
