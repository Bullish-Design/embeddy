# 0004 — The Qdrant scale path: adapter decisions (sparse, atomicity, FTS)

- **Date**: 2026-08-01
- **Status**: Accepted (Phase 8 / M7 gate; plan §10, CONCEPT §5.4)
- **Scope**: the decisions the Qdrant adapter (`embeddy/index/qdrant.py`)
  locks that the frozen protocol and wire surface do NOT pin. API facts
  were verified empirically against qdrant-client 1.18.0 with tiny probe
  scripts before this record was written.

## Context

Phase 8 ships a second `Searchable` backend (Qdrant) behind the frozen M4
protocol, selected by one config line (`store.url`). Three protocol-adjacent
questions have no frozen answer: sparse vectors (the protocol has no sparse
channel), reindex atomicity (Qdrant has no transactions), and full-text
search (Qdrant has no BM25 engine). This record documents the decisions and
the verified API facts behind them.

## Verified API facts (probes, 2026-08-01, qdrant-client 1.18.0)

- `QdrantClient(":memory:")` / `path=":memory:"` are hermetic and offline
  (no docker, no network) — the default test suite runs the real adapter
  against them.
- Point ids MUST be UUIDs or integers (`s1:0` raises "not a valid UUID");
  chunk ids are hashed to stable uuid5 ids with the original restored from
  the payload.
- Upserting a point with only ONE named vector REPLACES the others — sparse
  vectors must arrive together with dense.
- `FieldCondition` has no string-prefix condition in 1.18; the honest
  prefix translation stores all ancestor `path_prefixes` per point and
  filters with `MatchAny` (exact prefix semantics).
- Delete-by-filter with `must`/`must_not` works (the stale-cleanup
  primitive); filtered scroll pagination, exact count-with-filter, and
  `score_threshold` on `query_points` all work. Cosine scores are
  similarities in [-1, 1], higher = better — the same min_score direction
  as SqliteStore's `1 - cosine_distance`.
- Quantization config is accepted at collection creation in local mode
  (searches work) but NOT read back via `get_collection` — a local-mode
  artifact; on a server it applies. The adapter passes it through and never
  reads it back.
- `client.search()` is removed in 1.18 — `query_points` is the only path.

## Decision 1 — Sparse: plumbing ships, the encoder is deferred (option b)

The `EmbeddingProvider` protocol has no sparse channel, and learned-sparse
for bge-m3 needs the FlagEmbedding package (sentence-transformers cannot
emit it). **Ship the QdrantStore's sparse plumbing now and defer the real
encoder** (the plan's option (b)):

- every collection is created with the `sparse` named vector config;
- `add()` / `reindex_source()` accept an optional beyond-protocol
  `sparse_vectors` kwarg (aligned (indices, values) pairs, provided
  TOGETHER with dense — the probe fact above);
- `search_sparse(...)` is a beyond-protocol extension returning
  `ScoredDocument`s with a new `Metric.SPARSE_DOT` member (additive;
  sparse-dot similarity, higher = better);
- tests exercise the plumbing with synthetic sparse vectors; a real
  learned-sparse encoder (bge-m3 + FlagEmbedding) is explicitly OUT OF
  SCOPE (not planned — the probe confirmed sentence-transformers cannot
  emit it, and adding FlagEmbedding as a dependency is not wanted). The
  FakeProvider and the eval gate are untouched.

## Decision 2 — reindex_source atomicity: upsert-first, documented guarantee

Qdrant has no transactions; the protocol's "atomic swap" cannot be
one-transaction like SqliteStore. The adapter's order is:

1. upsert the new chunk set (new ids never collide with the old),
2. delete the STALE old ids by filter (`source_id == X AND NOT chunk_id IN
   new`),
3. refresh the source metadata point.

**Guarantee:** on failure during (1) the old chunks, vectors and source
metadata remain intact and queryable — the H7 data-loss failure mode cannot
happen (nothing was deleted yet). A failure during (2)/(3) may briefly
expose BOTH old and new chunks for the source; the next successful reindex
of the same source self-heals. This is deliberately WEAKER than SqliteStore
and documented as such (the test suite proves the (1)-failure path leaves
the old index queryable).

## Decision 3 — search_fts: pure-Python BM25 over payloads

Qdrant has no BM25/FTS engine. `search_fts` runs a classic BM25
(k1=1.2, b=0.75) scan over the stored payloads:

- filters are TRUE pre-filters (the scan only reads matching points — the
  M3 recall contract holds within the filtered set);
- scores mirror FTS5's negative-rank convention (`<= 0`, higher /
  less-negative = better) so `min_score` means the same thing as on
  SqliteStore;
- `raw` is accepted for protocol compatibility and is a NO-OP (there is no
  FTS5 query syntax to bypass);
- tokenization splits on non-alphanumerics with NO Porter stemming — a
  documented recall difference from the FTS5 backend (only dense search has
  a parity test vs SqliteStore).

This is an O(n) scan per FTS query — honest at the personal-project scale
Qdrant serves. A real lexical index is OUT OF SCOPE (not planned).

## Decision 4 — misc

- **Quantization** arrives via the DSN (`store.url=...?quantization=int8|binary|none`)
  or the `create_collection` kwarg; collection-level scalar/binary config.
- **Sync client, async protocol**: the qdrant-client is synchronous and the
  store methods call it inline (a remote server blocks the event loop
  during a call) — an accepted single-user tradeoff; the in-memory (tested)
  mode is instant. An async client is OUT OF SCOPE (not planned).
- **Extras**: `create_collection` / `get_chunk` / `list_chunks` /
  `list_collections` are implemented (the server's 501 path is unaffected);
  `count_fts` is deliberately absent (sqlite-only; the server never calls
  it).
- **Honest health**: `build_store` verifies qdrant reachability at open
  time; an unreachable qdrant makes `/health/ready` report not-ready with
  the reason.

## Consequences

- The adapter implements the frozen `Searchable` EXACTLY (no protocol or
  wire-shape change). The one additive type change is the `SPARSE_DOT`
  member on the `Metric` enum (types.py), needed to score sparse-search
  results honestly.
- Integration adapters (Haystack / LlamaIndex) are OUT OF SCOPE (plan §10
  lists them as optional post-v1; not built, not planned).
- `store.url` is the one config line; the sqlite default (bare server) is
  unchanged (the field reads `None` -> the server's `store_path`).
