-- =============================================================================
-- embeddy sqlite store schema — Phase-0 spike (draft for embeddy/index/sqlite.py)
-- CONCEPT §3.3/§5.4, IMPLEMENTATION_PLAN Phase 4.
--
-- Loaded via spikes/store_schema.py, which substitutes {collection} and {dim}.
-- All statements are idempotent (IF NOT EXISTS) so init is re-runnable.
--
-- Layout decisions locked here:
--   * sources: UNIQUE(collection_id, path) — dedup is source-level; two
--     identical files at different paths are two sources (fixes H-dedup).
--   * chunks.source_id FK ON DELETE CASCADE — delete_source cascades.
--   * PRAGMA user_version stamp — guards self-inflicted dev-time schema
--     breakage; there is NO v0.3.x migration (re-ingest from source).
--   * vec0 per collection, distance_metric=cosine — metric-honest storage
--     (fixes C3). score = 1.0 - distance ONLY under cosine.
--   * FTS5 external-content over chunks with porter + unicode61.
-- =============================================================================

PRAGMA user_version = 1;

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

-- Vector index: one vec0 virtual table per collection. {collection} is a
-- controlled identifier (the spike uses a fixed set); the real backend must
-- sanitize/validate collection ids before interpolating them here.
CREATE VIRTUAL TABLE IF NOT EXISTS "v_{collection}" USING vec0(
    id        TEXT PRIMARY KEY,
    embedding float[{dim}] distance_metric=cosine
);

-- FTS5: external content over chunks (keeps a single source of truth for
-- content). tokenize = porter + unicode61. Insert path must be kept in sync
-- (the backend owns that; the spike proves the sync + BM25 query).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, chunk_type, parent,
    content='chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);
