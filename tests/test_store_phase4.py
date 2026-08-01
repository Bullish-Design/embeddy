"""Phase-4 integration tests — real sqlite-vec + FTS5 (plan §6 test plan).

Covers, on a live store:
  * pre-filter recall: restrictive filters return FULL top_k (the M3
    JOIN-then-filter recall hole is closed — 43/50 -> 50/50)
  * metadata_match filters for real (H1 silent-no-op regression)
  * atomic reindex_source: mid-swap failure leaves the OLD index fully
    intact and queryable; success commits cleanly (H7)
  * delete_source cascade + chunk-level delete
  * min_score semantics per metric (cosine [0,1] / BM25 rank <= 0) — C3
  * FTS5 sanitization: metacharacter behavior + raw opt-in (plan §14)
  * total_results / count_fts semantics (plan §14)
  * search_hybrid over the real store (RRF / weighted / rerank stage)
  * schema user_version stamping (stale v1 DB -> SchemaError, re-ingest)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import aiosqlite
import numpy as np
import pytest

from embeddy.index.base import SearchFilters
from embeddy.index.filters import FilterCompileError
from embeddy.index.sqlite import SchemaError, SqliteStore, StoreError
from embeddy.protocol.rerank import RerankHit
from embeddy.protocol.types import Metric, SourceMetadata, StoredChunk
from embeddy.search import search_hybrid

pytestmark = pytest.mark.integration

DIM = 8


def _unit(i: int) -> np.ndarray:
    """Unit basis vector e_{i % DIM} — deterministic, hand-computable cosine."""
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def _chunk(
    chunk_id: str,
    source_id: str,
    content: str,
    *,
    chunk_type: str = "paragraph",
    parent: str | None = None,
    granularity: str | None = None,
) -> StoredChunk:
    return StoredChunk(
        id=chunk_id,
        collection_id="p4",
        source_id=source_id,
        content=content,
        chunk_type=chunk_type,
        start_line=1,
        end_line=1,
        parent=parent,
        granularity=granularity,
        token_count=len(content.split()),
    )


async def _seed(
    *,
    collection: str = "p4",
    dimension: int = DIM,
    sources: list[SourceMetadata] | None = None,
    chunks: list[StoredChunk] | None = None,
    vectors: list[np.ndarray] | None = None,
) -> tuple[SqliteStore, list[SourceMetadata], list[StoredChunk]]:
    """Open an in-memory store seeded with sources + chunks/vectors."""
    store = await SqliteStore.open(":memory:")
    await store.create_collection(collection, dimension)
    srcs = sources if sources is not None else []
    for src in srcs:
        await store.upsert_source(collection, src)
    if chunks and vectors:
        await store.add(collection, chunks, vectors)
    return store, srcs, chunks or []


async def _recall_store() -> tuple[SqliteStore, list[StoredChunk]]:
    """100 chunks (90 paragraph / 10 heading) across two sources with
    different content_types and path prefixes, for the recall tests."""
    srcs = [
        SourceMetadata(
            id="sA",
            collection_id="p4",
            path="docs/a/guide.md",
            content_hash="hA",
            size_bytes=100,
            content_type="markdown",
        ),
        SourceMetadata(
            id="sB",
            collection_id="p4",
            path="docs/b/manual.txt",
            content_hash="hB",
            size_bytes=200,
            content_type="text",
        ),
    ]
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    for src in srcs:
        await store.upsert_source("p4", src)
    chunks: list[StoredChunk] = []
    vectors: list[np.ndarray] = []
    for i in range(100):
        src = srcs[0] if i % 2 == 0 else srcs[1]
        chunk = _chunk(
            f"c{i:03d}",
            src.id,
            f"document chunk number {i}",
            chunk_type="heading" if i % 10 == 0 else "paragraph",
            parent="intro" if i % 10 == 1 else None,
        )
        chunks.append(chunk)
        vectors.append(_unit(i))
    await store.add("p4", chunks, vectors)
    return store, chunks


# --------------------------------------------------------------------------- #
# pre-filter recall (M3 regression) + filter correctness
# --------------------------------------------------------------------------- #


async def test_prefilter_recall_returns_full_topk() -> None:
    """The plan §6 headline test: the M3 JOIN-then-filter shape returned
    43/50 for a 10%-selective filter; in-scan aux-column pre-filters must
    return the full 50."""
    store, chunks = await _recall_store()
    try:
        q = _unit(0)
        hits = await store.search_vector(
            "p4", q, SearchFilters(chunk_types=("paragraph",)), top_k=50
        )
        assert len(hits) == 50, f"expected full top_k, got {len(hits)}"
        expected = {c.id for c in chunks if c.chunk_type == "paragraph"}
        assert {h.chunk_id for h in hits} <= expected  # top-50 of the 90 paragraphs
    finally:
        await store.close()


async def test_prefilter_recall_all_filter_fields() -> None:
    store, chunks = await _recall_store()
    try:
        q = _unit(0)

        # chunk_types
        hits = await store.search_vector("p4", q, SearchFilters(chunk_types=("heading",)), top_k=10)
        assert len(hits) == 10  # only 10 headings exist, but the scan must find ALL of them
        assert len(hits) == sum(1 for c in chunks if c.chunk_type == "heading")

        # content_types (source-level, denormalized onto vec rows)
        hits = await store.search_vector(
            "p4", q, SearchFilters(content_types=("markdown",)), top_k=60
        )
        assert len(hits) == 50  # 50 markdown chunks; asking for 60 returns all 50

        # source_path_prefix (byte-range in-scan constraint)
        hits = await store.search_vector(
            "p4", q, SearchFilters(source_path_prefix="docs/a/"), top_k=60
        )
        assert len(hits) == 50
        assert all(h.source_path.startswith("docs/a/") for h in hits)

        # metadata_match — the H1 regression: filters for real
        hits = await store.search_vector(
            "p4", q, SearchFilters(metadata_match=(("parent", "intro"),)), top_k=60
        )
        assert len(hits) == 10  # chunks with parent="intro" (i % 10 == 1)
        assert all(h.chunk_id in {c.id for c in chunks if c.parent == "intro"} for h in hits)

        # combined filters AND together
        hits = await store.search_vector(
            "p4",
            q,
            SearchFilters(
                chunk_types=("paragraph",),
                content_types=("markdown",),
                source_path_prefix="docs/a/",
            ),
            top_k=60,
        )
        expected = {
            c.id
            for i, c in enumerate(chunks)
            if i % 2 == 0 and i % 10 != 0  # markdown (even) + paragraph (not %10==0)
        }
        assert {h.chunk_id for h in hits} == expected
    finally:
        await store.close()


async def test_prefilter_filters_only_return_matching_rows() -> None:
    store, chunks = await _recall_store()
    try:
        hits = await store.search_vector(
            "p4",
            _unit(0),
            SearchFilters(chunk_types=("paragraph",), content_types=("markdown",)),
            top_k=50,
        )
        by_id = {h.chunk_id for h in hits}
        expected = {f"c{i:03d}" for i in range(100) if i % 2 == 0 and i % 10 != 0}
        assert by_id == expected
        assert len(by_id) == 40  # 50 markdown chunks minus 10 heading chunks
    finally:
        await store.close()


async def test_fts_filters_apply_before_truncation() -> None:
    store, chunks = await _recall_store()
    try:
        # every chunk content contains 'document' and 'chunk'; filters narrow
        hits = await store.search_fts(
            "p4",
            "document",
            SearchFilters(content_types=("markdown",), chunk_types=("paragraph",)),
            top_k=100,
        )
        assert len(hits) == 40  # 40 markdown paragraphs
        assert all(
            h.chunk_id in {f"c{i:03d}" for i in range(100) if i % 2 == 0 and i % 10 != 0}
            for h in hits
        )
        hits = await store.search_fts(
            "p4", "document", SearchFilters(source_path_prefix="docs/b/"), top_k=100
        )
        assert len(hits) == 50
        assert all(h.source_path.startswith("docs/b/") for h in hits)
    finally:
        await store.close()


async def test_metadata_match_unknown_field_raises() -> None:
    """H1: an unfilterable metadata field is an explicit error on BOTH search
    paths — never a silent no-op."""
    store, chunks = await _recall_store()
    try:
        with pytest.raises(FilterCompileError, match="unknown metadata_match field"):
            await store.search_vector(
                "p4", _unit(0), SearchFilters(metadata_match=(("symbols_defined", "x"),)), top_k=5
            )
        with pytest.raises(FilterCompileError, match="unknown metadata_match field"):
            await store.search_fts(
                "p4", "document", SearchFilters(metadata_match=(("nope", "x"),)), top_k=5
            )
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# score correctness + min_score semantics (C3)
# --------------------------------------------------------------------------- #


async def test_cosine_scores_hand_computed() -> None:
    c1 = _unit(0)
    c2 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    c2 = c2 / np.linalg.norm(c2)
    c3 = _unit(1)
    store, _, _ = await _seed(
        chunks=[_chunk("a", "s", "alpha"), _chunk("b", "s", "beta"), _chunk("c", "s", "gamma")],
        vectors=[c1, c2, c3],
        sources=[
            SourceMetadata(id="s", collection_id="p4", path="x.txt", content_hash="h", size_bytes=1)
        ],
    )
    try:
        hits = await store.search_vector("p4", c1, SearchFilters(), top_k=3)
        by_id = {h.chunk_id: h.score for h in hits}
        assert by_id["a"] == pytest.approx(1.0, abs=1e-5)
        assert by_id["b"] == pytest.approx(float(np.dot(c1, c2)), abs=1e-5)
        assert by_id["c"] == pytest.approx(0.0, abs=1e-5)
        assert [h.chunk_id for h in hits] == ["a", "b", "c"]
        assert all(h.metric == Metric.COSINE for h in hits)
    finally:
        await store.close()


async def test_min_score_cosine_semantics() -> None:
    """Cosine scores live in [0, 1], higher = better: min_score keeps
    score >= min_score (pushed down as distance <= 1 - min_score)."""
    c1 = _unit(0)
    c2 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    c2 = c2 / np.linalg.norm(c2)
    c3 = _unit(1)
    store, _, _ = await _seed(
        chunks=[_chunk("a", "s", "alpha"), _chunk("b", "s", "beta"), _chunk("c", "s", "gamma")],
        vectors=[c1, c2, c3],
        sources=[
            SourceMetadata(id="s", collection_id="p4", path="x.txt", content_hash="h", size_bytes=1)
        ],
    )
    try:
        # scores: a=1.0, b~=0.707, c=0.0
        hits = await store.search_vector("p4", c1, SearchFilters(), top_k=3, min_score=0.5)
        assert {h.chunk_id for h in hits} == {"a", "b"}
        hits = await store.search_vector("p4", c1, SearchFilters(), top_k=3, min_score=0.9)
        assert {h.chunk_id for h in hits} == {"a"}
        hits = await store.search_vector("p4", c1, SearchFilters(), top_k=3, min_score=1.0)
        assert {h.chunk_id for h in hits} == {"a"}  # distance <= 0 keeps the exact match
        hits = await store.search_vector("p4", c1, SearchFilters(), top_k=3, min_score=1.1)
        assert hits == []  # impossible threshold -> no results, not an error
        hits = await store.search_vector("p4", c1, SearchFilters(), top_k=3, min_score=-1.0)
        assert len(hits) == 3  # no-op threshold
    finally:
        await store.close()


async def test_min_score_bm25_semantics() -> None:
    """BM25 ranks are <= 0 and higher (less negative) is better: min_score
    keeps rank >= min_score, filtering the worst matches only."""
    srcs = [
        SourceMetadata(id="s", collection_id="p4", path="x.txt", content_hash="h", size_bytes=1)
    ]
    chunks = [_chunk(f"d{i}", "s", "the quick brown fox") for i in range(5)]
    chunks += [_chunk(f"e{i}", "s", "the lazy dog") for i in range(5)]
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    await store.upsert_source("p4", srcs[0])
    await store.add("p4", chunks, [_unit(i) for i in range(len(chunks))])
    try:
        all_hits = await store.search_fts("p4", "fox", SearchFilters(), top_k=100)
        assert all(h.score <= 0.0 for h in all_hits), "BM25 ranks are <= 0"
        assert {h.chunk_id for h in all_hits} == {f"d{i}" for i in range(5)}

        # exact pre-truncation count semantics (plan §14 cheap COUNT(*))
        assert await store.count_fts("p4", "fox", SearchFilters()) == 5
        assert await store.count_fts("p4", "dog", SearchFilters()) == 5
        assert await store.count_fts("p4", "fox", SearchFilters(chunk_types=("heading",))) == 0

        worst = min(h.score for h in all_hits)
        lenient = await store.search_fts(
            "p4", "fox", SearchFilters(), top_k=100, min_score=worst - 1.0
        )
        assert len(lenient) == len(all_hits)  # threshold below the worst keeps everything
        strict = await store.search_fts(
            "p4", "fox", SearchFilters(), top_k=100, min_score=worst + 0.1
        )
        assert len(strict) < len(all_hits)  # threshold above the worst drops the tail
        assert all(h.score >= worst + 0.1 for h in strict)
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# atomic reindex_source (H7) + source ops
# --------------------------------------------------------------------------- #


async def _reindex_fixture() -> tuple[
    SqliteStore, SourceMetadata, list[StoredChunk], list[np.ndarray]
]:
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    src = SourceMetadata(
        id="s1",
        collection_id="p4",
        path="a.md",
        content_hash="hash1",
        size_bytes=100,
        mtime=datetime(2026, 1, 1),
        content_type="markdown",
    )
    await store.upsert_source("p4", src)
    old_chunks = [
        _chunk("s1:0", "s1", "alpha beta", chunk_type="heading", parent="intro"),
        _chunk("s1:1", "s1", "gamma delta"),
    ]
    await store.add("p4", old_chunks, [_unit(0), _unit(1)])
    return store, src, old_chunks, [_unit(0), _unit(1)]


async def test_reindex_failure_keeps_old_index_intact() -> None:
    """H7: a mid-swap failure must leave old chunks, vectors, FTS rows and
    source metadata fully intact and queryable. Failure injection: a new
    chunk id collides with an EXISTING chunk of another source — the INSERT
    violates the chunks PK mid-transaction, after the old rows are gone."""
    store, src, old_chunks, _ = await _reindex_fixture()
    try:
        # a second source owns chunk id "x1" -> reindexing s1 with a new
        # chunk reusing "x1" fails at INSERT time, mid-swap.
        await store.upsert_source(
            "p4",
            SourceMetadata(
                id="s2", collection_id="p4", path="b.md", content_hash="h2", size_bytes=1
            ),
        )
        await store.add("p4", [_chunk("x1", "s2", "other source")], [_unit(2)])

        new_src = SourceMetadata(
            id="s1",
            collection_id="p4",
            path="a.md",
            content_hash="hash2",
            size_bytes=999,
            mtime=datetime(2026, 2, 2),
            content_type="markdown",
        )
        bad_chunks = [_chunk("x1", "s1", "clobber"), _chunk("s1:5", "s1", "new stuff")]
        with pytest.raises(sqlite3.IntegrityError):  # chunk PK collision mid-swap
            await store.reindex_source("p4", new_src, bad_chunks, [_unit(3), _unit(4)])

        # old chunks intact and queryable — vector, FTS, metadata, stats
        hits = await store.search_vector("p4", _unit(0), SearchFilters(), top_k=10)
        assert {h.chunk_id for h in hits} == {"s1:0", "s1:1", "x1"}
        fts = await store.search_fts("p4", "alpha", SearchFilters(), top_k=10)
        assert any("alpha" in h.content for h in fts)
        fts2 = await store.search_fts("p4", "new stuff", SearchFilters(), top_k=10)
        assert fts2 == []
        stats = await store.stats("p4")
        assert stats.chunk_count == 3
        got = await store.get_source("p4", "a.md")
        assert got is not None and got.content_hash == "hash1" and got.size_bytes == 100
        assert got.content_type == "markdown"
    finally:
        await store.close()


async def test_reindex_success_swaps_atomically() -> None:
    store, src, old_chunks, _ = await _reindex_fixture()
    try:
        new_src = SourceMetadata(
            id="s1",
            collection_id="p4",
            path="a.md",
            content_hash="hash2",
            size_bytes=500,
            mtime=datetime(2026, 2, 2),
            content_type="markdown",
        )
        new_chunks = [_chunk("s1:0", "s1", "alpha beta v2"), _chunk("s1:1", "s1", "brand new term")]
        await store.reindex_source("p4", new_src, new_chunks, [_unit(0), _unit(4)])

        stats = await store.stats("p4")
        assert stats.chunk_count == 2
        # old FTS terms gone, new terms present (no stale FTS rows)
        assert await store.search_fts("p4", "gamma", SearchFilters(), top_k=10) == []
        assert {
            h.chunk_id for h in await store.search_fts("p4", "brand", SearchFilters(), top_k=10)
        } == {"s1:1"}
        # vectors swapped
        by_id = {
            h.chunk_id: h.score
            for h in await store.search_vector("p4", _unit(4), SearchFilters(), top_k=10)
        }
        assert by_id["s1:1"] == pytest.approx(1.0, abs=1e-5)
        assert "s1:0" in by_id
        # source metadata updated
        got = await store.get_source("p4", "a.md")
        assert got is not None and got.content_hash == "hash2" and got.size_bytes == 500
    finally:
        await store.close()


async def test_reindex_validates_before_touching_data() -> None:
    store, src, old_chunks, _ = await _reindex_fixture()
    try:
        # wrong-dimension vectors -> StoreError, nothing changes
        with pytest.raises(StoreError, match="shape"):
            await store.reindex_source(
                "p4", src, [_chunk("s1:9", "s1", "x")], [np.ones(3, dtype=np.float32)]
            )
        # duplicate chunk ids
        with pytest.raises(StoreError, match="duplicate chunk id"):
            await store.reindex_source(
                "p4",
                src,
                [_chunk("s1:9", "s1", "x"), _chunk("s1:9", "s1", "y")],
                [_unit(0), _unit(1)],
            )
        # foreign source ids
        with pytest.raises(StoreError, match="reference source_id"):
            await store.reindex_source("p4", src, [_chunk("s1:9", "other", "x")], [_unit(0)])
        # chunks/vectors length mismatch
        with pytest.raises(StoreError, match="length mismatch"):
            await store.reindex_source("p4", src, [_chunk("s1:9", "s1", "x")], [_unit(0), _unit(1)])
        stats = await store.stats("p4")
        assert stats.chunk_count == 2
    finally:
        await store.close()


async def test_delete_source_cascades() -> None:
    store, src, old_chunks, _ = await _reindex_fixture()
    try:
        await store.delete_source("p4", "s1")
        assert await store.get_source("p4", "a.md") is None
        stats = await store.stats("p4")
        assert stats.chunk_count == 0 and stats.source_count == 0
        assert await store.search_vector("p4", _unit(0), SearchFilters(), top_k=10) == []
        assert await store.search_fts("p4", "alpha", SearchFilters(), top_k=10) == []
    finally:
        await store.close()


async def test_delete_chunk_rows() -> None:
    store, src, old_chunks, _ = await _reindex_fixture()
    try:
        await store.delete("p4", ["s1:0"])
        by_id = {
            h.chunk_id for h in await store.search_vector("p4", _unit(0), SearchFilters(), top_k=10)
        }
        assert "s1:0" not in by_id
        assert "s1:1" in by_id
        assert await store.search_fts("p4", "alpha", SearchFilters(), top_k=10) == []
        assert (await store.stats("p4")).chunk_count == 1
    finally:
        await store.close()


async def test_list_sources_sorted_with_content_type() -> None:
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    try:
        await store.upsert_source(
            "p4",
            SourceMetadata(
                id="b",
                collection_id="p4",
                path="b.md",
                content_hash="1",
                size_bytes=1,
                content_type="text",
            ),
        )
        await store.upsert_source(
            "p4",
            SourceMetadata(
                id="a",
                collection_id="p4",
                path="a.md",
                content_hash="2",
                size_bytes=2,
                content_type="markdown",
            ),
        )
        await store.upsert_source(
            "p4",
            SourceMetadata(id="c", collection_id="p4", path="c.py", content_hash="3", size_bytes=3),
        )
        sources = await store.list_sources("p4")
        assert [s.path for s in sources] == ["a.md", "b.md", "c.py"]
        assert sources[0].content_type == "markdown"
        assert sources[2].content_type is None
        # upsert by path refreshes the row (UNIQUE(collection_id, path))
        await store.upsert_source(
            "p4",
            SourceMetadata(
                id="a2",
                collection_id="p4",
                path="a.md",
                content_hash="9",
                size_bytes=9,
                content_type="pdf",
            ),
        )
        sources = await store.list_sources("p4")
        assert len(sources) == 3
        refreshed = [s for s in sources if s.path == "a.md"][0]
        assert (
            refreshed.id == "a2"
            and refreshed.content_hash == "9"
            and refreshed.content_type == "pdf"
        )
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# FTS sanitization + raw mode (plan §14)
# --------------------------------------------------------------------------- #


async def _fts_store() -> tuple[SqliteStore, list[StoredChunk]]:
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    await store.upsert_source(
        "p4",
        SourceMetadata(
            id="s",
            collection_id="p4",
            path="x.txt",
            content_hash="h",
            size_bytes=1,
            content_type="text",
        ),
    )
    chunks = [
        _chunk("d0", "s", "alpha beta"),
        _chunk("d1", "s", "alpha gamma"),
        _chunk("d2", "s", "beta gamma"),
        _chunk("d3", "s", "delta epsilon"),
    ]
    await store.add("p4", chunks, [_unit(i) for i in range(4)])
    return store, chunks


async def test_fts_metacharacter_queries_default_mode() -> None:
    """Default mode: metacharacters are stripped, tokens are literal and
    ANDed — a user's FTS5 syntax never reaches the engine."""
    store, chunks = await _fts_store()
    try:
        # "-alpha" -> literal alpha
        assert {
            h.chunk_id for h in await store.search_fts("p4", "-alpha", SearchFilters(), top_k=10)
        } == {"d0", "d1"}
        # "alpha OR beta" -> "alpha" AND "beta" (literal AND, NOT the user's OR)
        assert {
            h.chunk_id
            for h in await store.search_fts("p4", "alpha OR beta", SearchFilters(), top_k=10)
        } == {"d0"}
        # "alpha*" -> literal alpha (no prefix expansion)
        assert {
            h.chunk_id for h in await store.search_fts("p4", "alpha*", SearchFilters(), top_k=10)
        } == {"d0", "d1"}
        # token-free query -> no results, no error
        assert await store.search_fts("p4", "!!!", SearchFilters(), top_k=10) == []
    finally:
        await store.close()


async def test_fts_raw_opt_in_mode() -> None:
    """raw=True passes the query verbatim: real FTS5 boolean syntax works,
    and a malformed raw query raises StoreError (never a bare sqlite error)."""
    store, chunks = await _fts_store()
    try:
        # real boolean OR
        hits = await store.search_fts("p4", "alpha OR beta", SearchFilters(), top_k=10, raw=True)
        assert {h.chunk_id for h in hits} == {"d0", "d1", "d2"}
        # quoted phrase
        hits = await store.search_fts("p4", '"alpha gamma"', SearchFilters(), top_k=10, raw=True)
        assert {h.chunk_id for h in hits} == {"d1"}
        # malformed raw query -> typed StoreError with context
        with pytest.raises(StoreError, match="invalid FTS5 query"):
            await store.search_fts("p4", "alpha AND (", SearchFilters(), top_k=10, raw=True)
    finally:
        await store.close()


async def test_fts_porter_stemming_kept() -> None:
    store, chunks = await _fts_store()
    try:
        # porter stems 'betas' -> 'beta' inside the quoted phrase (spike-verified)
        assert {
            h.chunk_id for h in await store.search_fts("p4", "betas", SearchFilters(), top_k=10)
        } == {"d0", "d2"}
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# search_hybrid over the real store
# --------------------------------------------------------------------------- #


class _ReorderingReranker:
    model_name = "scripted"

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse
        self.documents: list[str] | None = None

    async def rerank(self, query, documents, top_k=None, instruction=None):
        self.documents = documents
        order = list(range(len(documents)))
        if self.reverse:
            order.reverse()
        return [RerankHit(index=i, score=float(len(order) - rank)) for rank, i in enumerate(order)]


async def test_search_hybrid_rrf_and_weighted() -> None:
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    await store.upsert_source(
        "p4",
        SourceMetadata(
            id="s",
            collection_id="p4",
            path="guide.md",
            content_hash="h",
            size_bytes=1,
            content_type="markdown",
        ),
    )
    chunks = [
        _chunk("s:0", "s", "token expiry rotation policy"),
        _chunk("s:1", "s", "billing invoices monthly"),
        _chunk("s:2", "s", "search index stores embeddings"),
    ]
    # content-correlated vectors (distinct directions per doc)
    await store.add("p4", chunks, [_unit(0), _unit(1), _unit(2)])
    try:
        query_vec = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        res = await search_hybrid(
            store, collection="p4", query_text="token policy", query_vector=query_vec, top_k=3
        )
        assert res.results
        assert res.results[0].chunk_id == "s:0"
        assert res.metric == Metric.RRF
        assert res.mode == "rrf"
        assert res.total_results >= 1

        res_w = await search_hybrid(
            store,
            collection="p4",
            query_text="token policy",
            query_vector=query_vec,
            top_k=3,
            mode="weighted",
        )
        assert res_w.results
        assert res_w.metric == Metric.WEIGHTED

        # rerank stage flips the order
        res_r = await search_hybrid(
            store,
            collection="p4",
            query_text="token policy",
            query_vector=query_vec,
            top_k=3,
            reranker=_ReorderingReranker(reverse=True),
        )
        assert res_r.metric == Metric.RERANK
        assert all(h.metric == Metric.RERANK for h in res_r.results)
        assert len(res_r.results) == 3

        # filters flow into the hybrid legs (pre-filter, not post-fusion)
        res_f = await search_hybrid(
            store,
            collection="p4",
            query_text="token",
            query_vector=query_vec,
            top_k=3,
            filters=SearchFilters(metadata_match=(("parent", "intro"),)),
        )
        # no chunk has parent="intro" -> no results from either leg
        assert res_f.results == []
        assert res_f.total_results == 0
    finally:
        await store.close()


async def test_search_hybrid_topk_truncation_and_total() -> None:
    store, chunks = await _recall_store()
    try:
        q = _unit(0)
        res = await search_hybrid(
            store, collection="p4", query_text="document", query_vector=q, top_k=10
        )
        assert len(res.results) == 10
        assert res.total_results >= 10  # union of the two legs' candidates
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# schema versioning + error paths
# --------------------------------------------------------------------------- #


async def test_fresh_file_store_stamps_schema_version(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "store.db"
    store = await SqliteStore.open(db_path)
    await store.close()
    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert version == 2


async def test_stale_schema_version_requires_reingest(tmp_path: Path) -> None:
    """A v1 (M3-era) database must be rejected with a clear re-ingest error —
    never silently half-initialized (there is no migration)."""
    db_path = tmp_path / "old.db"
    db = await aiosqlite.connect(str(db_path))
    await db.execute("PRAGMA user_version = 1")
    await db.close()
    with pytest.raises(SchemaError, match="re-ingest"):
        await SqliteStore.open(str(db_path))


async def test_newer_schema_version_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "future.db"
    db = await aiosqlite.connect(str(db_path))
    await db.execute("PRAGMA user_version = 99")
    await db.close()
    with pytest.raises(SchemaError, match="newer"):
        await SqliteStore.open(str(db_path))


async def test_add_validation_errors() -> None:
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    await store.upsert_source(
        "p4",
        SourceMetadata(id="s", collection_id="p4", path="a.md", content_hash="h", size_bytes=1),
    )
    try:
        with pytest.raises(StoreError, match="length mismatch"):
            await store.add("p4", [_chunk("a", "s", "x")], [])
        with pytest.raises(StoreError, match="unknown source_id"):
            await store.add("p4", [_chunk("a", "ghost", "x")], [_unit(0)])
        with pytest.raises(StoreError, match="duplicate chunk id"):
            await store.add(
                "p4", [_chunk("a", "s", "x"), _chunk("a", "s", "y")], [_unit(0), _unit(1)]
            )
        with pytest.raises(StoreError, match="shape"):
            await store.add("p4", [_chunk("a", "s", "x")], [np.ones(3, dtype=np.float32)])
        with pytest.raises(StoreError, match="unknown collection"):
            await store.search_vector("ghost", _unit(0), SearchFilters(), top_k=5)
        with pytest.raises(StoreError, match="top_k"):
            await store.search_vector("p4", _unit(0), SearchFilters(), top_k=0)
        with pytest.raises(StoreError, match="shape"):
            await store.search_vector("p4", np.ones(3, dtype=np.float32), SearchFilters(), top_k=5)
        with pytest.raises(StoreError, match="unknown collection"):
            await store.search_fts("ghost", "x", SearchFilters(), top_k=5)
        assert (await store.stats("p4")).chunk_count == 0
    finally:
        await store.close()


async def test_sourcemetadata_content_type_validation() -> None:
    with pytest.raises(ValueError, match="content_type"):
        SourceMetadata(
            id="s", collection_id="c", path="p", content_hash="h", size_bytes=1, content_type="  "
        )


async def test_more_error_paths() -> None:
    """Coverage for the remaining honest error paths: closed store,
    collection/dimension validation, empty payload no-ops, non-finite
    min_score, and the add() rollback on a cross-collection PK collision."""
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    await store.upsert_source(
        "p4",
        SourceMetadata(id="s", collection_id="p4", path="a.md", content_hash="h", size_bytes=1),
    )
    await store.add("p4", [_chunk("k1", "s", "first")], [_unit(0)])

    # empty payloads are no-ops
    await store.add("p4", [], [])
    await store.delete("p4", [])

    # collection dimension validation
    with pytest.raises(StoreError, match="dimension"):
        await store.create_collection("p4", DIM + 1)  # mismatch
    with pytest.raises(StoreError, match="dimension"):
        await store.create_collection("bad-dim", 0)
    with pytest.raises(StoreError, match="invalid collection id"):
        await store.create_collection("bad id!", DIM)

    # top_k / min_score validation
    with pytest.raises(StoreError, match="top_k"):
        await store.search_fts("p4", "first", SearchFilters(), top_k=0)
    with pytest.raises(StoreError, match="min_score"):
        await store.search_vector("p4", _unit(0), SearchFilters(), top_k=5, min_score=float("nan"))

    # count_fts with a token-free query
    assert await store.count_fts("p4", "!!!", SearchFilters()) == 0

    # add() rollback: the chunks PK is global; reusing a chunk id across
    # collections fails mid-transaction and must roll back cleanly.
    await store.create_collection("other", DIM)
    await store.upsert_source(
        "other",
        SourceMetadata(id="s2", collection_id="other", path="b.md", content_hash="h", size_bytes=1),
    )
    with pytest.raises(sqlite3.IntegrityError):  # chunks PK is global; cross-collection reuse
        await store.add("other", [_chunk("k1", "s2", "clash")], [_unit(1)])
    assert (await store.stats("other")).chunk_count == 0
    # the original collection is untouched
    assert (await store.stats("p4")).chunk_count == 1

    await store.close()

    # closed store raises a typed error on every entry point
    with pytest.raises(StoreError, match="not open"):
        await store.search_vector("p4", _unit(0), SearchFilters(), top_k=5)
    with pytest.raises(StoreError, match="not open"):
        await store.stats("p4")


async def test_reindex_empty_source_is_noop() -> None:
    """reindex_source on a source with no existing chunks: no stale rows to
    clean, new chunks land cleanly (covers the empty-id guard)."""
    store = await SqliteStore.open(":memory:")
    await store.create_collection("p4", DIM)
    src = SourceMetadata(
        id="fresh", collection_id="p4", path="new.md", content_hash="h", size_bytes=1
    )
    await store.upsert_source("p4", src)
    try:
        assert (await store.stats("p4")).chunk_count == 0
        await store.reindex_source(
            "p4", src, [_chunk("fresh:0", "fresh", "hello world")], [_unit(0)]
        )
        assert (await store.stats("p4")).chunk_count == 1
        hits = await store.search_fts("p4", "hello", SearchFilters(), top_k=10)
        assert [h.chunk_id for h in hits] == ["fresh:0"]
    finally:
        await store.close()
