"""SqliteStore — sqlite-vec + FTS5 backend (the default Searchable).

Phase-4 core (plan §6), built on the Phase-1 slice and the Phase-0 spike
(`spikes/schema.sql` + `spikes/store_schema.py`, 7/7 checks proven).
CONCEPT §3.3/§3.4/§5.4/§5.5.

aiosqlite from day one: the connection lives on a worker thread, so
`db.isolation_level` must NEVER be set from the caller thread (sqlite3
threading check — verified in the spike). Atomicity is expressed with the
implicit transaction + `commit()`/`rollback()` only.

Schema facts locked here (schema.sql, bumped to user_version=2 at M4):
  * sources: UNIQUE(collection_id, path) — dedup is source-level; two
    identical files at different paths are two sources (fixes H-dedup).
    `content_type TEXT` added at M4 so SearchFilters.content_types compiles
    to a real pre-filter.
  * chunks.source_id FK ON DELETE CASCADE — delete_source cascades.
  * per-collection vec0 with distance_metric=cosine — metric-honest
    (fixes C3). score = 1.0 - distance ONLY under cosine. Auxiliary
    columns (chunk_type/parent/granularity/content_type/source_path) carry
    the filterable fields so SearchFilters compile to IN-SCAN constraints
    (sqlite-vec 0.1.9 rejects LIKE but honors EQ/IN/comparisons during the
    KNN scan — probed 2026-08-01; the M3 JOIN-then-filter shape returned
    43/50 rows, the recall hole this phase fixes).
  * FTS5 external content over chunks, porter + unicode61. FTS rows are
    deleted BEFORE content rows (stale-entry probe).
  * PRAGMA user_version stamp — guards dev-time schema breakage; there is
    NO migration (re-ingest from source). A v1 database raises SchemaError.

Phase-4 surface: sources (upsert/get/list/reindex/delete), delete, full
pre-filter compilation (index/filters.py), FTS sanitization + raw mode,
min_score per metric, search_vector/search_fts/count_fts, stats. Hybrid
search + the rerank stage live in search.py (search_hybrid over Searchable).
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import aiosqlite
import numpy as np
import sqlite_vec

from embeddy.index.base import CollectionInfo, SearchFilters, StoreError
from embeddy.index.filters import compile_filters, sanitize_fts_query
from embeddy.index.sources import (
    delete_chunk_index_rows,
    insert_chunks_with_vectors,
    serialize_vector,
)
from embeddy.protocol.types import (
    CollectionStats,
    Metric,
    ScoredDocument,
    SourceId,
    SourceMetadata,
    StoredChunk,
    Vector,
    assert_unit_vector,
)

SCHEMA_VERSION = 2

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id               TEXT PRIMARY KEY,
    vector_dimension INTEGER NOT NULL,          -- the RESOLVED dimension (MRL-aware)
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,              -- SourceId (stable string id)
    collection_id TEXT NOT NULL
                 REFERENCES collections(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,                 -- canonical path within collection
    content_hash TEXT NOT NULL,                 -- sha256 hex from chonkai
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    mtime        TEXT,                          -- ISO-8601, optional
    content_type TEXT,                          -- source document type (M4): "markdown"/"pdf"/...
    UNIQUE (collection_id, path)                -- source-level dedup
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,             -- f"<source_id>:<seq>"
    collection_id TEXT NOT NULL
                 REFERENCES collections(id) ON DELETE CASCADE,
    source_id     TEXT NOT NULL
                 REFERENCES sources(id) ON DELETE CASCADE,   -- cascade delete
    content       TEXT NOT NULL,
    chunk_type    TEXT NOT NULL,                -- paragraph/heading/function/...
    start_line    INTEGER NOT NULL,             -- 1-based, inclusive
    end_line      INTEGER NOT NULL,
    parent        TEXT,                         -- enclosing heading/definition
    granularity   TEXT,                         -- module/class/function
    token_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_collection
    ON chunks(collection_id);

-- FTS5: external content over chunks (single source of truth for content).
-- tokenize = porter + unicode61. The backend keeps this in sync on add/
-- reindex/delete (FTS rows removed BEFORE content rows).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, chunk_type, parent,
    content='chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);

PRAGMA user_version = 2;
"""

# Controlled collection ids only: they are interpolated into virtual table
# names below, so anything exotic is rejected up front.
_COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")

_SELECT_CHUNK_COLS = (
    "c.id, c.collection_id, c.source_id, s.path AS source_path, c.content, "
    "c.chunk_type, c.start_line, c.end_line, c.parent, c.granularity, "
    "c.token_count"
)

# vec0 aux columns: declared at collection creation (each needs a type);
# insert helper in index/sources.py uses the same field order.
_VEC_AUX_COLUMNS = (
    "chunk_type TEXT, parent TEXT, granularity TEXT, content_type TEXT, source_path TEXT"
)


# StoreError + CollectionInfo moved to embeddy.index.base at Phase 8 so
# the Qdrant adapter shares the SAME error + record classes (the server's
# 404/400/501 mapping is backend-agnostic). They remain importable from
# embeddy.index.sqlite for backward compatibility (module-level re-export).


class SchemaError(RuntimeError):
    """The on-disk schema is newer/older than this code; re-ingest from
    source (there is no in-place migration)."""


async def _fetchall(
    db: aiosqlite.Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[sqlite3.Row]:
    """execute_fetchall's stubs type the result as Iterable[Row]; the runtime
    value is a list and every call site indexes it, so normalize here."""
    return list(await db.execute_fetchall(sql, params))


class SqliteStore:
    """sqlite-vec + FTS5 backend. Open with `SqliteStore.open(path)` (async)
    or `async with SqliteStore.open(...) as store`."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._db: aiosqlite.Connection | None = None

    @classmethod
    async def open(cls, path: str | Path = ":memory:") -> SqliteStore:
        store = cls(path)
        await store._connect()
        return store

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> SqliteStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # connection / schema
    # ------------------------------------------------------------------ #

    async def _connect(self) -> None:
        db = await aiosqlite.connect(self._path)
        db.row_factory = sqlite3.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        self._db = db
        try:
            await self._init_schema()
        except BaseException:
            # never leak the connection (and its non-daemon worker thread)
            # on an init failure — stale-schema DBs must not hang the caller.
            await db.close()
            self._db = None
            raise

    async def _init_schema(self) -> None:
        db = self._require_db()
        version = int((await _fetchall(db, "PRAGMA user_version"))[0][0])
        if version == 0:
            await db.executescript(_BASE_SCHEMA)
            await db.commit()
        elif version < SCHEMA_VERSION:
            raise SchemaError(
                f"database schema v{version} predates this code (v{SCHEMA_VERSION}); "
                "there is no in-place migration — delete the database and re-ingest "
                "from source"
            )
        elif version > SCHEMA_VERSION:
            raise SchemaError(
                f"database schema v{version} is newer than this code "
                f"(v{SCHEMA_VERSION}); re-ingest from source"
            )
        # version == SCHEMA_VERSION: assume present. The stamp exists to guard
        # self-inflicted dev-time breakage, not to migrate.

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise StoreError("store is not open; call SqliteStore.open() first")
        return self._db

    @staticmethod
    def _validate_collection_id(collection: str) -> None:
        if not _COLLECTION_ID_RE.match(collection):
            raise StoreError(
                f"invalid collection id {collection!r}; must match {_COLLECTION_ID_RE.pattern}"
            )

    async def _collection_dimension(self, collection: str) -> int:
        db = self._require_db()
        rows = await _fetchall(
            db, "SELECT vector_dimension FROM collections WHERE id = ?", (collection,)
        )
        if not rows:
            raise StoreError(f"unknown collection: {collection!r}")
        return int(rows[0][0])

    async def _missing_sources(self, collection: str, source_ids: set[SourceId]) -> set[SourceId]:
        if not source_ids:
            return set()
        db = self._require_db()
        placeholders = ",".join("?" for _ in source_ids)
        rows = await _fetchall(
            db,
            f"SELECT id FROM sources WHERE collection_id = ? AND id IN ({placeholders})",
            (collection, *sorted(source_ids)),
        )
        present = {row["id"] for row in rows}
        return source_ids - present

    async def _source_lookup(
        self, collection: str, source_ids: set[SourceId]
    ) -> dict[str, tuple[str | None, str]]:
        """source_id -> (content_type, path) for the vec0 aux columns."""
        if not source_ids:
            return {}
        db = self._require_db()
        placeholders = ",".join("?" for _ in source_ids)
        rows = await _fetchall(
            db,
            f"SELECT id, content_type, path FROM sources "
            f"WHERE collection_id = ? AND id IN ({placeholders})",
            (collection, *sorted(source_ids)),
        )
        return {row["id"]: (row["content_type"], row["path"]) for row in rows}

    # ------------------------------------------------------------------ #
    # collections / sources
    # ------------------------------------------------------------------ #

    async def create_collection(self, collection: str, dimension: int) -> None:
        """Register a collection at its RESOLVED vector dimension.

        Re-creating an existing collection with the same dimension is a
        no-op; a dimension mismatch raises (the C2 bug class is impossible:
        the collection records the resolved dimension and the vec table can
        never disagree with it).
        """
        self._validate_collection_id(collection)
        if dimension < 1:
            raise StoreError(f"dimension must be >= 1, got {dimension}")
        db = self._require_db()
        await db.execute(
            "INSERT OR IGNORE INTO collections(id, vector_dimension) VALUES (?, ?)",
            (collection, dimension),
        )
        rows = await _fetchall(
            db, "SELECT vector_dimension FROM collections WHERE id = ?", (collection,)
        )
        existing = int(rows[0][0])
        if existing != dimension:
            raise StoreError(
                f"collection {collection!r} already exists with dimension "
                f"{existing}, got {dimension}"
            )
        await db.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "v_{collection}" USING vec0('
            "id TEXT PRIMARY KEY, "
            f"embedding float[{dimension}] distance_metric=cosine, "
            f"{_VEC_AUX_COLUMNS})"
        )
        await db.commit()

    async def upsert_source(self, collection: str, source: SourceMetadata) -> SourceId:
        """Insert or refresh one source row (UNIQUE(collection_id, path))."""
        self._validate_collection_id(collection)
        await self._collection_dimension(collection)  # collection must exist
        db = self._require_db()
        mtime = source.mtime.isoformat() if source.mtime else None
        await db.execute(
            "INSERT INTO sources(id, collection_id, path, content_hash, "
            "size_bytes, mtime, content_type) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(collection_id, path) DO UPDATE SET "
            "id = excluded.id, content_hash = excluded.content_hash, "
            "size_bytes = excluded.size_bytes, mtime = excluded.mtime, "
            "content_type = excluded.content_type",
            (
                source.id,
                collection,
                source.path,
                source.content_hash,
                source.size_bytes,
                mtime,
                source.content_type,
            ),
        )
        await db.commit()
        return source.id

    async def get_source(self, collection: str, path: str) -> SourceMetadata | None:
        self._validate_collection_id(collection)
        db = self._require_db()
        rows = await _fetchall(
            db,
            "SELECT id, collection_id, path, content_hash, size_bytes, mtime, "
            "content_type FROM sources WHERE collection_id = ? AND path = ?",
            (collection, path),
        )
        if not rows:
            return None
        row = rows[0]
        return SourceMetadata(
            id=row["id"],
            collection_id=row["collection_id"],
            path=row["path"],
            content_hash=row["content_hash"],
            size_bytes=int(row["size_bytes"]),
            mtime=_parse_mtime(row["mtime"]),
            content_type=row["content_type"],
        )

    async def list_sources(self, collection: str) -> list[SourceMetadata]:
        """All sources in a collection, ordered by path (stable diff order
        for the Phase-5 incremental sync)."""
        self._validate_collection_id(collection)
        await self._collection_dimension(collection)  # must exist
        db = self._require_db()
        rows = await _fetchall(
            db,
            "SELECT id, collection_id, path, content_hash, size_bytes, mtime, "
            "content_type FROM sources WHERE collection_id = ? ORDER BY path",
            (collection,),
        )
        return [
            SourceMetadata(
                id=row["id"],
                collection_id=row["collection_id"],
                path=row["path"],
                content_hash=row["content_hash"],
                size_bytes=int(row["size_bytes"]),
                mtime=_parse_mtime(row["mtime"]),
                content_type=row["content_type"],
            )
            for row in rows
        ]

    async def delete_source(self, collection: str, source_id: SourceId) -> None:
        """Cascade-delete a source: vec0 + FTS5 rows explicitly, chunks via
        the sources FK ON DELETE CASCADE."""
        self._validate_collection_id(collection)
        db = self._require_db()
        chunk_ids = [
            row["id"]
            for row in await _fetchall(
                db, "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
            )
        ]
        try:
            await delete_chunk_index_rows(db, collection=collection, chunk_ids=chunk_ids)
            await db.execute(
                "DELETE FROM sources WHERE id = ? AND collection_id = ?",
                (source_id, collection),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    # ------------------------------------------------------------------ #
    # chunk listing (server /api/v1/chunks + /api/v1/similar support) ---- #
    # ------------------------------------------------------------------ #

    async def get_chunk(self, collection: str, chunk_id: str) -> StoredChunk | None:
        """Fetch ONE stored chunk by id (the server's /api/v1/similar).

        Store EXTRA beyond the Searchable protocol (like create_collection /
        count_fts) — the protocol has no get-by-id, and the Phase-6 server
        needs it to re-embed an existing chunk. Returns None when the chunk
        does not exist in `collection`.
        """
        self._validate_collection_id(collection)
        db = self._require_db()
        rows = await _fetchall(
            db,
            "SELECT id, collection_id, source_id, content, chunk_type, "
            "start_line, end_line, parent, granularity, token_count "
            "FROM chunks WHERE collection_id = ? AND id = ?",
            (collection, chunk_id),
        )
        if not rows:
            return None
        row = rows[0]
        return StoredChunk(
            id=row["id"],
            collection_id=row["collection_id"],
            source_id=row["source_id"],
            content=row["content"],
            chunk_type=row["chunk_type"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            parent=row["parent"],
            granularity=row["granularity"],
            token_count=int(row["token_count"]),
        )

    async def list_chunks(
        self,
        collection: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredChunk]:
        """List stored chunks in id order (the server's GET /api/v1/chunks).

        Store EXTRA beyond the Searchable protocol (the server's chunks
        listing surface). `limit`/`offset` are page bounds (both validated
        >= the documented minimums); the server additionally enforces its
        own limit cap (max_top_k, OOM prevention).
        """
        self._validate_collection_id(collection)
        if limit < 1:
            raise StoreError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise StoreError(f"offset must be >= 0, got {offset}")
        await self._collection_dimension(collection)  # must exist
        db = self._require_db()
        rows = await _fetchall(
            db,
            "SELECT id, collection_id, source_id, content, chunk_type, "
            "start_line, end_line, parent, granularity, token_count "
            "FROM chunks WHERE collection_id = ? ORDER BY id LIMIT ? OFFSET ?",
            (collection, limit, offset),
        )
        return [
            StoredChunk(
                id=row["id"],
                collection_id=row["collection_id"],
                source_id=row["source_id"],
                content=row["content"],
                chunk_type=row["chunk_type"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                parent=row["parent"],
                granularity=row["granularity"],
                token_count=int(row["token_count"]),
            )
            for row in rows
        ]

    async def list_collections(self) -> list[CollectionInfo]:
        """List all collections in id order (the server's GET
        /api/v1/collections). Store EXTRA beyond the Searchable protocol —
        the server owns collection lifecycle.
        """
        db = self._require_db()
        rows = await _fetchall(db, "SELECT id, vector_dimension FROM collections ORDER BY id")
        return [
            CollectionInfo(collection_id=row["id"], vector_dimension=int(row["vector_dimension"]))
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # add / delete / reindex
    # ------------------------------------------------------------------ #

    async def add(
        self,
        collection: str,
        chunks: list[StoredChunk],
        vectors: list[Vector],
    ) -> None:
        """Store chunks + vectors + FTS rows in ONE transaction.

        All chunks must reference sources that already exist (call
        upsert_source first — the chunks.source_id FK is ON). Vectors must be
        unit-norm float32 of the collection's resolved dimension.
        """
        self._validate_collection_id(collection)
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")
        if not chunks:
            return
        dim = await self._collection_dimension(collection)  # must exist
        self._validate_vectors(vectors, dim)

        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise StoreError("duplicate chunk id in add() payload")
        missing = await self._missing_sources(collection, {c.source_id for c in chunks})
        if missing:
            raise StoreError(
                f"chunks reference unknown source_id(s) {sorted(missing)}; call upsert_source first"
            )
        source_lookup = await self._source_lookup(collection, {c.source_id for c in chunks})

        db = self._require_db()
        try:
            await insert_chunks_with_vectors(
                db,
                collection=collection,
                chunks=chunks,
                vectors=vectors,
                source_lookup=source_lookup,
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        """Delete specific chunks (vec0 + FTS5 + chunks rows in one txn)."""
        self._validate_collection_id(collection)
        if not chunk_ids:
            return
        await self._collection_dimension(collection)  # must exist
        db = self._require_db()
        try:
            await delete_chunk_index_rows(db, collection=collection, chunk_ids=chunk_ids)
            placeholders = ",".join("?" for _ in chunk_ids)
            await db.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders}) AND collection_id = ?",
                [*chunk_ids, collection],
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    async def reindex_source(
        self,
        collection: str,
        source: SourceMetadata,
        chunks: list[StoredChunk],
        vectors: list[Vector],
    ) -> None:
        """ATOMIC SWAP of a source's chunk set in ONE implicit transaction
        (fixes H7, spike-proven both ways). On ANY failure the old chunks,
        vectors and source metadata remain intact and queryable — the caller
        never sees a half-reindexed source. Never sets db.isolation_level
        (aiosqlite threading constraint)."""
        self._validate_collection_id(collection)
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")
        dim = await self._collection_dimension(collection)  # must exist
        self._validate_vectors(vectors, dim)
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise StoreError("duplicate chunk id in reindex_source payload")
        foreign = {c.source_id for c in chunks} - {source.id}
        if foreign:
            raise StoreError(
                f"reindex_source chunks reference source_id(s) {sorted(foreign)}; "
                f"expected only {source.id!r} (one source at a time)"
            )

        db = self._require_db()
        old_ids = [
            row["id"]
            for row in await _fetchall(
                db, "SELECT id FROM chunks WHERE source_id = ?", (source.id,)
            )
        ]
        mtime = source.mtime.isoformat() if source.mtime else None
        try:
            # swap-in: index rows first (FTS delete reads the chunks rows),
            # then content rows, then the new set, then source metadata.
            await delete_chunk_index_rows(db, collection=collection, chunk_ids=old_ids)
            await db.execute("DELETE FROM chunks WHERE source_id = ?", (source.id,))
            await insert_chunks_with_vectors(
                db,
                collection=collection,
                chunks=chunks,
                vectors=vectors,
                source_lookup={source.id: (source.content_type, source.path)},
            )
            await db.execute(
                "UPDATE sources SET content_hash = ?, size_bytes = ?, "
                "mtime = ?, content_type = ? WHERE id = ? AND collection_id = ?",
                (
                    source.content_hash,
                    source.size_bytes,
                    mtime,
                    source.content_type,
                    source.id,
                    collection,
                ),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

    @staticmethod
    def _validate_vectors(vectors: list[Vector], dim: int) -> None:
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != dim:
                raise StoreError(
                    f"vector shape {arr.shape} does not match collection dimension {dim}"
                )
            assert_unit_vector(arr)

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    async def search_vector(
        self,
        collection: str,
        query_vector: Vector,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[ScoredDocument]:
        """Cosine KNN with IN-SCAN pre-filters (plan §6: filters apply before
        the k-bound scan, so a restrictive filter still returns full top_k —
        the M3 recall hole is closed).

        `min_score` is a COSINE threshold: scores live in [0, 1] and higher
        is better, so `score >= min_score` is pushed down as
        `distance <= 1 - min_score` (probed: distance constraints are honored
        during the scan). It is never comparable to a BM25 threshold.
        """
        self._validate_collection_id(collection)
        if top_k < 1:
            raise StoreError(f"top_k must be >= 1, got {top_k}")
        dim = await self._collection_dimension(collection)  # must exist
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != dim:
            raise StoreError(
                f"query vector shape {q.shape} does not match collection dimension {dim}"
            )
        assert_unit_vector(q)
        where, params = compile_filters(filters, target="vec")
        if min_score is not None:
            _validate_min_score(min_score)
            where += " AND v.distance <= ?"
            params.append(1.0 - min_score)
        sql = (
            f"SELECT {_SELECT_CHUNK_COLS}, v.distance FROM "
            f'"v_{collection}" v '
            "JOIN chunks c ON c.id = v.id "
            "JOIN sources s ON s.id = c.source_id "
            "WHERE v.embedding MATCH ? AND k = ?"
            f"{where} ORDER BY v.distance LIMIT ?"
        )
        db = self._require_db()
        rows = await _fetchall(db, sql, [serialize_vector(q), top_k, *params, top_k])
        return [
            _row_to_scored(row, score=1.0 - float(row["distance"]), metric=Metric.COSINE)
            for row in rows
        ]

    async def search_fts(
        self,
        collection: str,
        query: str,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
        raw: bool = False,
    ) -> list[ScoredDocument]:
        """BM25 over FTS5 (porter + unicode61). Filters apply to the FULL
        match set before ORDER BY/LIMIT (full recall; the plan's pre-filter
        requirement relative to truncation).

        `min_score` is a BM25/rank threshold: FTS5 ranks are <= 0 and higher
        (less negative) is better, so `rank >= min_score` keeps the
        "good enough" tail. Never comparable to a cosine threshold.
        `raw=True` passes the query to FTS5 verbatim (plan §14 opt-in); a
        malformed raw query raises StoreError.
        """
        self._validate_collection_id(collection)
        if top_k < 1:
            raise StoreError(f"top_k must be >= 1, got {top_k}")
        await self._collection_dimension(collection)  # must exist
        match = sanitize_fts_query(query, raw=raw)
        if not match:
            return []
        where, params = compile_filters(filters, target="join")
        if min_score is not None:
            _validate_min_score(min_score)
            where += " AND f.rank >= ?"
            params.append(min_score)
        sql = (
            f"SELECT {_SELECT_CHUNK_COLS}, f.rank AS score FROM chunks_fts f "
            "JOIN chunks c ON c.rowid = f.rowid "
            "JOIN sources s ON s.id = c.source_id "
            "WHERE chunks_fts MATCH ?"
            f"{where} ORDER BY f.rank LIMIT ?"
        )
        db = self._require_db()
        try:
            rows = await _fetchall(db, sql, [match, *params, top_k])
        except sqlite3.OperationalError as exc:
            raise StoreError(f"invalid FTS5 query {query!r} (raw={raw}): {exc}") from exc
        return [_row_to_scored(row, score=float(row["score"]), metric=Metric.BM25) for row in rows]

    async def count_fts(
        self,
        collection: str,
        query: str,
        filters: SearchFilters,
        *,
        raw: bool = False,
    ) -> int:
        """Exact PRE-TRUNCATION match count — plan §14's cheap COUNT(*), not
        a materialization. Available for FTS-only UIs; search_hybrid reports
        its own candidate-union total (see SearchResult)."""
        self._validate_collection_id(collection)
        await self._collection_dimension(collection)  # must exist
        match = sanitize_fts_query(query, raw=raw)
        if not match:
            return 0
        where, params = compile_filters(filters, target="join")
        sql = (
            "SELECT COUNT(*) FROM chunks_fts f "
            "JOIN chunks c ON c.rowid = f.rowid "
            "JOIN sources s ON s.id = c.source_id "
            "WHERE chunks_fts MATCH ?"
            f"{where}"
        )
        db = self._require_db()
        rows = await _fetchall(db, sql, [match, *params])
        return int(rows[0][0])

    # ------------------------------------------------------------------ #
    # stats
    # ------------------------------------------------------------------ #

    async def stats(self, collection: str) -> CollectionStats:
        self._validate_collection_id(collection)
        db = self._require_db()
        dim = await self._collection_dimension(collection)  # must exist
        chunk_count = int(
            (
                await _fetchall(
                    db, "SELECT COUNT(*) FROM chunks WHERE collection_id = ?", (collection,)
                )
            )[0][0]
        )
        source_count = int(
            (
                await _fetchall(
                    db, "SELECT COUNT(*) FROM sources WHERE collection_id = ?", (collection,)
                )
            )[0][0]
        )
        size_bytes = int(
            (
                await _fetchall(
                    db,
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM sources WHERE collection_id = ?",
                    (collection,),
                )
            )[0][0]
        )
        return CollectionStats(
            collection_id=collection,
            chunk_count=chunk_count,
            source_count=source_count,
            vector_dimension=dim,
            size_bytes=size_bytes,
            last_updated=None,  # no updated_at column in the schema
        )


def _validate_min_score(min_score: float) -> None:
    if not math.isfinite(min_score):
        raise StoreError(f"min_score must be finite, got {min_score!r}")


def _parse_mtime(value: object) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def _row_to_scored(row: sqlite3.Row, *, score: float, metric: Metric) -> ScoredDocument:
    return ScoredDocument(
        chunk_id=row["id"],
        collection_id=row["collection_id"],
        source_id=row["source_id"],
        source_path=row["source_path"],
        content=row["content"],
        score=score,
        metric=metric,
    )
