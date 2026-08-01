"""SqliteStore — sqlite-vec + FTS5 backend (the default Searchable).

Phase-1 keystone slice, ported from the Phase-0 spike (`spikes/schema.sql` +
`spikes/store_schema.py`, 7/7 checks proven). CONCEPT §3.3/§5.4, plan §3.

aiosqlite from day one: the connection lives on a worker thread, so
`db.isolation_level` must NEVER be set from the caller thread (sqlite3
threading check — verified in the spike). Atomicity is expressed with the
implicit transaction + `commit()`/`rollback()` only.

Schema facts locked here (schema.sql):
  * sources: UNIQUE(collection_id, path) — dedup is source-level.
  * chunks.source_id FK ON DELETE CASCADE — delete_source cascades (Phase 4).
  * PRAGMA user_version stamp — guards dev-time schema breakage; there is
    NO v0.3.x migration (re-ingest from source).
  * per-collection vec0 with distance_metric=cosine — metric-honest (fixes
    C3). score = 1.0 - distance ONLY under cosine.
  * FTS5 external content over chunks, porter + unicode61.
  * vec0 tables with a TEXT PRIMARY KEY take no rowid in INSERT.

Phase-1 surface: create_collection / upsert_source / get_source / add /
search_vector / search_fts / stats. delete / reindex_source / delete_source /
list_sources and full SQL pre-filter compilation (content_types,
source_path_prefix, metadata_match) land in Phase 4.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
import sqlite_vec

from embeddy.index.base import SearchFilters
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

SCHEMA_VERSION = 1

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
-- tokenize = porter + unicode61. The backend keeps this in sync on add.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, chunk_type, parent,
    content='chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);

PRAGMA user_version = 1;
"""

# Controlled collection ids only: they are interpolated into virtual table
# names below, so anything exotic is rejected up front.
_COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")

_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

_SELECT_CHUNK_COLS = (
    "c.id, c.collection_id, c.source_id, s.path AS source_path, c.content, "
    "c.chunk_type, c.start_line, c.end_line, c.parent, c.granularity, "
    "c.token_count"
)


class StoreError(RuntimeError):
    """A store-level operation failed (unknown collection, dimension
    mismatch, missing source, ...)."""


class SchemaError(RuntimeError):
    """The on-disk schema is newer than this code; re-ingest from source."""


def _serialize_vector(vec: Vector) -> str:
    """sqlite-vec takes the embedding as a float-list string."""
    arr = np.asarray(vec, dtype=np.float32)
    return "[" + ",".join(f"{float(x):.9g}" for x in arr) + "]"


async def _fetchall(
    db: aiosqlite.Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[sqlite3.Row]:
    """execute_fetchall's stubs type the result as Iterable[Row]; the runtime
    value is a list and every call site indexes it, so normalize here."""
    return list(await db.execute_fetchall(sql, params))


def _sanitize_fts_query(query: str) -> str:
    """Safe FTS5 default (plan §14): double-quote-wrap each token, AND them.

    Phrase-wrapping avoids hand-escaping FTS5 metacharacters; porter
    stemming still applies inside phrases. Phase 4 adds the documented raw
    opt-in mode.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens)


def _compile_filters(filters: SearchFilters) -> tuple[str, list[Any]]:
    """Phase-1 filter subset. chunk_types pre-filters on the chunks join;
    the remaining fields are compiled to true SQL pre-filters in Phase 4
    (plan §4) — raising beats the H1 silent-no-op class of bug."""
    clauses: list[str] = []
    params: list[Any] = []
    if filters.content_types:
        raise NotImplementedError("content_types pre-filter compiles in Phase 4")
    if filters.source_path_prefix is not None:
        raise NotImplementedError("source_path_prefix pre-filter compiles in Phase 4")
    if filters.chunk_types:
        placeholders = ",".join("?" for _ in filters.chunk_types)
        clauses.append(f"c.chunk_type IN ({placeholders})")
        params.extend(filters.chunk_types)
    if filters.metadata_match:
        raise NotImplementedError("metadata_match pre-filter compiles in Phase 4")
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


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
        await self._init_schema()

    async def _init_schema(self) -> None:
        db = self._require_db()
        version = int((await _fetchall(db, "PRAGMA user_version"))[0][0])
        if version == 0:
            await db.executescript(_BASE_SCHEMA)
            await db.commit()
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
            f"id TEXT PRIMARY KEY, embedding float[{dimension}] "
            "distance_metric=cosine)"
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
            "size_bytes, mtime) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(collection_id, path) DO UPDATE SET "
            "id = excluded.id, content_hash = excluded.content_hash, "
            "size_bytes = excluded.size_bytes, mtime = excluded.mtime",
            (
                source.id,
                collection,
                source.path,
                source.content_hash,
                source.size_bytes,
                mtime,
            ),
        )
        await db.commit()
        return source.id

    async def get_source(self, collection: str, path: str) -> SourceMetadata | None:
        self._validate_collection_id(collection)
        db = self._require_db()
        rows = await _fetchall(
            db,
            "SELECT id, collection_id, path, content_hash, size_bytes, mtime "
            "FROM sources WHERE collection_id = ? AND path = ?",
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
        )

    # ------------------------------------------------------------------ #
    # add
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
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != dim:
                raise StoreError(
                    f"vector shape {arr.shape} does not match collection dimension {dim}"
                )
            assert_unit_vector(arr)

        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise StoreError("duplicate chunk id in add() payload")
        missing = await self._missing_sources(collection, {c.source_id for c in chunks})
        if missing:
            raise StoreError(
                f"chunks reference unknown source_id(s) {sorted(missing)}; call upsert_source first"
            )

        db = self._require_db()
        try:
            await db.executemany(
                "INSERT INTO chunks(id, collection_id, source_id, content, "
                "chunk_type, start_line, end_line, parent, granularity, "
                "token_count) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        c.id,
                        collection,
                        c.source_id,
                        c.content,
                        c.chunk_type,
                        c.start_line,
                        c.end_line,
                        c.parent,
                        c.granularity,
                        c.token_count,
                    )
                    for c in chunks
                ],
            )
            rowids: dict[str, int] = {}
            for cid in ids:
                row = (await _fetchall(db, "SELECT rowid FROM chunks WHERE id = ?", (cid,)))[0]
                rowids[cid] = int(row[0])
            await db.executemany(
                f'INSERT INTO "v_{collection}"(id, embedding) VALUES (?, ?)',
                [(c.id, _serialize_vector(vec)) for c, vec in zip(chunks, vectors, strict=True)],
            )
            await db.executemany(
                "INSERT INTO chunks_fts(rowid, content, chunk_type, parent) VALUES (?,?,?,?)",
                [(rowids[c.id], c.content, c.chunk_type, c.parent) for c in chunks],
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise

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

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    async def search_vector(
        self,
        collection: str,
        query_vector: Vector,
        filters: SearchFilters,
        top_k: int,
    ) -> list[ScoredDocument]:
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
        where, params = _compile_filters(filters)
        sql = (
            f"SELECT {_SELECT_CHUNK_COLS}, v.distance FROM "
            f'"v_{collection}" v '
            "JOIN chunks c ON c.id = v.id "
            "JOIN sources s ON s.id = c.source_id "
            "WHERE v.embedding MATCH ? AND k = ?"
            f"{where} ORDER BY v.distance LIMIT ?"
        )
        db = self._require_db()
        rows = await _fetchall(db, sql, [_serialize_vector(q), top_k, *params, top_k])
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
    ) -> list[ScoredDocument]:
        self._validate_collection_id(collection)
        if top_k < 1:
            raise StoreError(f"top_k must be >= 1, got {top_k}")
        await self._collection_dimension(collection)  # must exist
        match = _sanitize_fts_query(query)
        if not match:
            return []
        where, params = _compile_filters(filters)
        sql = (
            f"SELECT {_SELECT_CHUNK_COLS}, f.rank AS score FROM chunks_fts f "
            "JOIN chunks c ON c.rowid = f.rowid "
            "JOIN sources s ON s.id = c.source_id "
            "WHERE chunks_fts MATCH ?"
            f"{where} ORDER BY f.rank LIMIT ?"
        )
        db = self._require_db()
        rows = await _fetchall(db, sql, [match, *params, top_k])
        return [_row_to_scored(row, score=float(row["score"]), metric=Metric.BM25) for row in rows]

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
            last_updated=None,  # no updated_at column in the Phase-1 schema
        )


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
