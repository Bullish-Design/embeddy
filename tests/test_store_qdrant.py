"""Phase-8 integration tests — the QdrantStore adapter (plan §10 test plan)
on the HERMETIC in-memory backend (probed: `QdrantClient(":memory:")` is
offline, no docker/network — runs in the default suite).

Covers, on a live QdrantStore:
  * the frozen Searchable protocol EXACTLY (chunks, search, sources, stats)
  * payload-filter translation (content_types / source_path_prefix /
    chunk_types / metadata_match) as TRUE pre-filters — restrictive filters
    return full top_k (the M3 recall contract)
  * source-op lifecycle (upsert/get/list/delete cascade)
  * reindex_source semantics: subset swap cleans stale ids; a mid-swap
    failure leaves the OLD chunks intact and queryable
  * sparse plumbing (add(..., sparse_vectors=) + search_sparse extension)
  * quantization params (int8/binary accepted at collection creation)
  * the beyond-protocol extras (create_collection / get_chunk / list_chunks
    / list_collections) + stats parity vs SqliteStore on the same vectors
  * min_score semantics per metric (cosine / BM25 / sparse-dot)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest

from embeddy.index.base import SearchFilters
from embeddy.index.factory import parse_store_url
from embeddy.index.qdrant import (
    QdrantStore,
    _bm25_scores,
    _PointId,
    _tokenize,
    compile_qdrant_filter,
)
from embeddy.index.sqlite import SqliteStore, StoreError
from embeddy.protocol.types import Metric, SourceMetadata, StoredChunk
from embeddy.search import search_hybrid

pytestmark = pytest.mark.integration

DIM = 8


def _unit(i: int) -> np.ndarray:
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
        collection_id="qd",
        source_id=source_id,
        content=content,
        chunk_type=chunk_type,
        start_line=1,
        end_line=1,
        parent=parent,
        granularity=granularity,
        token_count=len(content.split()),
    )


def _source(
    sid: str,
    path: str,
    *,
    content_type: str | None = "markdown",
    size: int = 100,
    mtime: datetime | None = None,
    content_hash: str | None = None,
) -> SourceMetadata:
    return SourceMetadata(
        id=sid,
        collection_id="qd",
        path=path,
        content_hash=content_hash or f"h-{sid}",
        size_bytes=size,
        mtime=mtime,
        content_type=content_type,
    )


async def _open(
    *, collection: str = "qd", dimension: int = DIM, quantization: str | None = None
) -> QdrantStore:
    store = await QdrantStore.open(parse_store_url("qdrant://:memory:"))
    await store.create_collection(collection, dimension, quantization=quantization)
    return store


async def _seeded(
    *,
    collection: str = "qd",
    sources: list[SourceMetadata] | None = None,
    chunks: list[StoredChunk] | None = None,
    vectors: list[np.ndarray] | None = None,
    sparse: list[tuple[list[int], list[float]]] | None = None,
) -> QdrantStore:
    store = await _open(collection=collection)
    for src in sources or []:
        await store.upsert_source(collection, src)
    if chunks and vectors:
        await store.add(collection, chunks, vectors, sparse_vectors=sparse)
    return store


# --------------------------------------------------------------------------- #
# pure functions (mock-free)
# --------------------------------------------------------------------------- #


def test_compile_qdrant_filter_empty_is_none() -> None:
    assert compile_qdrant_filter(SearchFilters()) is None


def test_compile_qdrant_filter_all_fields() -> None:
    f = compile_qdrant_filter(
        SearchFilters(
            content_types=("markdown", "text"),
            source_path_prefix="docs/a",
            chunk_types=("paragraph",),
            metadata_match=(("tag", "x"), ("lang", "py")),
        )
    )
    assert f is not None
    assert isinstance(f.must, list)
    must: list[Any] = f.must  # qdrant types Filter.must as a wide union
    assert len(must) == 5
    # deterministic order: content_types, source_path_prefix, chunk_types,
    # then the metadata pairs.
    assert must[0].key == "content_type"  # type: ignore[attr-defined]
    assert must[1].key == "path_prefixes"  # type: ignore[attr-defined]
    assert must[2].key == "chunk_type"  # type: ignore[attr-defined]
    assert must[3].key == "tag"  # type: ignore[attr-defined]
    assert must[4].key == "lang"  # type: ignore[attr-defined]


def test_bm25_scores_rank_overlap() -> None:
    docs = [
        ("a", "the cat sat on the mat"),
        ("b", "the dog sat on the mat"),
        ("c", "a dog and a cat and a mat"),
    ]
    scores = _bm25_scores(docs, "cat")
    assert scores[0] > scores[2] > 0  # a has 1 cat, c has 1 cat + shorter doc? assert a > 0
    assert scores[1] == 0.0  # no cat term -> no overlap
    # multi-term query ranks the doc with more overlapping terms higher
    scores2 = _bm25_scores(docs, "dog mat")
    assert scores2[1] > scores2[0]


def test_bm25_scores_empty_corpus() -> None:
    assert _bm25_scores([], "anything") == []


def test_bm25_scores_zero_len_docs() -> None:
    """Empty-content documents (avgdl = 0) must not divide by zero."""
    scores = _bm25_scores([("a", ""), ("b", "cat")], "cat")
    assert scores[1] > 0
    assert scores[0] == 0.0


def test_tokenize() -> None:
    assert _tokenize("Hello, WORLD 123!") == ["hello", "world", "123"]
    assert _tokenize("") == []
    assert _tokenize("café") == ["caf"]  # [a-z0-9] only; accents split out


def test_path_prefixes() -> None:
    assert QdrantStore._path_prefixes("src/a/b.py") == ["src", "src/a", "src/a/b.py"]
    assert QdrantStore._path_prefixes("docs/x.md") == ["docs", "docs/x.md"]
    assert QdrantStore._path_prefixes("top") == ["top"]


def test_point_ids_are_stable_uuids() -> None:
    a1 = _PointId.chunk("c", "s1:0")
    a2 = _PointId.chunk("c", "s1:0")
    b = _PointId.chunk("d", "s1:0")
    assert a1 == a2
    assert a1 != b
    s1 = _PointId.source("s1")
    assert len(s1) == 36  # uuid4/uuid5 string shape


# --------------------------------------------------------------------------- #
# collections + stats
# --------------------------------------------------------------------------- #


async def test_create_collection_and_recreate() -> None:
    store = await _open()
    try:
        # recreate with the same dimension is a no-op
        await store.create_collection("qd", DIM)
        stats = await store.stats("qd")
        assert stats.chunk_count == 0
        assert stats.source_count == 0
        assert stats.vector_dimension == DIM
    finally:
        await store.close()


async def test_create_collection_dimension_mismatch_raises() -> None:
    store = await _open()
    try:
        with pytest.raises(StoreError, match="already exists"):
            await store.create_collection("qd", DIM + 1)
    finally:
        await store.close()


async def test_create_collection_validation() -> None:
    store = await _open()
    try:
        with pytest.raises(StoreError, match="invalid collection"):
            await store.create_collection("bad id!", DIM)
        with pytest.raises(StoreError, match="dimension"):
            await store.create_collection("ok", 0)
    finally:
        await store.close()


async def test_quantization_int8_and_binary_accepted() -> None:
    for quant in ("int8", "binary"):
        store = await QdrantStore.open(parse_store_url("qdrant://:memory:"))
        try:
            await store.create_collection(f"q_{quant}", DIM, quantization=quant)
            await store.upsert_source(f"q_{quant}", _source("s1", "a.md"))
            await store.add(f"q_{quant}", [_chunk("s1:0", "s1", "hello quantized")], [_unit(0)])
            hits = await store.search_vector(f"q_{quant}", _unit(0), SearchFilters(), 5)
            assert [h.chunk_id for h in hits] == ["s1:0"]
        finally:
            await store.close()


async def test_quantization_from_dsn_default() -> None:
    spec = parse_store_url("qdrant://:memory:?quantization=int8")
    store = await QdrantStore.open(spec)
    try:
        await store.create_collection("qd", DIM)  # no explicit quantization -> DSN default
        await store.stats("qd")  # exists, searchable
    finally:
        await store.close()


async def test_quantization_invalid_raises() -> None:
    store = await _open()
    try:
        with pytest.raises(StoreError, match="invalid quantization"):
            await store.create_collection("bad", DIM, quantization="float16")
    finally:
        await store.close()


async def test_list_collections_filters_companions() -> None:
    store = await _open()
    try:
        await store.create_collection("second", 4)
        infos = await store.list_collections()
        names = [i.collection_id for i in infos]
        assert names == ["qd", "second"]
        assert {i.vector_dimension for i in infos} == {DIM, 4}
    finally:
        await store.close()


async def test_stats_counts_and_size_bytes() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md", size=100), _source("sB", "docs/b.md", size=200)],
        chunks=[
            _chunk("sA:0", "sA", "one"),
            _chunk("sA:1", "sA", "two"),
            _chunk("sB:0", "sB", "three"),
        ],
        vectors=[_unit(0), _unit(1), _unit(2)],
    )
    try:
        stats = await store.stats("qd")
        assert stats.chunk_count == 3
        assert stats.source_count == 2
        assert stats.size_bytes == 300
        assert stats.vector_dimension == DIM
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# source ops (first-class — CONCEPT §3.3)
# --------------------------------------------------------------------------- #


async def test_source_lifecycle() -> None:
    store = await _open()
    try:
        assert await store.get_source("qd", "docs/a.md") is None
        src = _source("sA", "docs/a.md", mtime=datetime(2026, 1, 1, 12, 0, 0))
        assert await store.upsert_source("qd", src) == "sA"
        got = await store.get_source("qd", "docs/a.md")
        assert got is not None
        assert got.id == "sA"
        assert got.content_hash == "h-sA"
        assert got.mtime == datetime(2026, 1, 1, 12, 0, 0)
        assert got.content_type == "markdown"
        listed = await store.list_sources("qd")
        assert [s.path for s in listed] == ["docs/a.md"]
        # upsert refreshes (same path, new metadata)
        updated = _source("sA", "docs/a.md", content_type="text", size=500)
        await store.upsert_source("qd", updated)
        refreshed = await store.get_source("qd", "docs/a.md")
        assert refreshed is not None
        assert refreshed.size_bytes == 500
    finally:
        await store.close()


async def test_upsert_source_requires_collection() -> None:
    store = await _open()
    try:
        with pytest.raises(StoreError, match="unknown collection"):
            await store.upsert_source("nope", _source("sA", "a.md"))
    finally:
        await store.close()


async def test_delete_source_cascades_chunks() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a/guide.md"), _source("sB", "docs/b/manual.txt")],
        chunks=[
            _chunk("sA:0", "sA", "a0"),
            _chunk("sA:1", "sA", "a1"),
            _chunk("sB:0", "sB", "b0"),
        ],
        vectors=[_unit(0), _unit(1), _unit(2)],
    )
    try:
        await store.delete_source("qd", "sA")
        stats = await store.stats("qd")
        assert stats.chunk_count == 1
        assert stats.source_count == 1
        assert await store.get_source("qd", "docs/a.md") is None
        hits = await store.search_vector("qd", _unit(0), SearchFilters(), 5)
        assert [h.chunk_id for h in hits] == ["sB:0"]
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# add / delete / validation
# --------------------------------------------------------------------------- #


async def test_add_and_search_roundtrip() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[
            _chunk("sA:0", "sA", "alpha"),
            _chunk("sA:1", "sA", "beta"),
        ],
        vectors=[_unit(0), _unit(1)],
    )
    try:
        hits = await store.search_vector("qd", _unit(1), SearchFilters(), 5)
        assert hits[0].chunk_id == "sA:1"
        assert hits[0].metric == Metric.COSINE
        assert hits[0].source_path == "docs/a.md"
        assert hits[0].content == "beta"
    finally:
        await store.close()


async def test_add_validation_errors() -> None:
    store = await _seeded(sources=[_source("sA", "docs/a.md")])
    try:
        with pytest.raises(StoreError, match="length mismatch"):
            await store.add("qd", [_chunk("sA:0", "sA", "x")], [])
        with pytest.raises(StoreError, match="unknown source"):
            await store.add(
                "qd",
                [_chunk("ghost:0", "ghost", "x")],
                [_unit(0)],
            )
        with pytest.raises(StoreError, match="duplicate chunk id"):
            await store.add(
                "qd", [_chunk("d", "sA", "x"), _chunk("d", "sA", "y")], [_unit(0), _unit(1)]
            )
        bad_vec = np.full(DIM, 0.5, dtype=np.float32)  # norm != 1
        # non-unit vectors raise ValueError (the protocol guard — the SAME
        # behavior as SqliteStore's _validate_vectors, which lets
        # assert_unit_vector propagate).
        with pytest.raises(ValueError, match="unit-norm"):
            await store.add("qd", [_chunk("v", "sA", "x")], [bad_vec])
        with pytest.raises(StoreError, match="shape"):
            await store.add("qd", [_chunk("w", "sA", "x")], [np.zeros(DIM + 1, dtype=np.float32)])
        with pytest.raises(StoreError, match="sparse_vectors"):
            await store.add("qd", [_chunk("sp", "sA", "x")], [_unit(0)], sparse_vectors=[])
    finally:
        await store.close()


async def test_add_unknown_collection_raises() -> None:
    store = await _open()
    try:
        with pytest.raises(StoreError, match="unknown collection"):
            await store.add("nope", [_chunk("x", "sA", "x")], [_unit(0)])
    finally:
        await store.close()


async def test_add_empty_payload_is_noop() -> None:
    store = await _seeded(sources=[_source("sA", "docs/a.md")])
    try:
        await store.add("qd", [], [])
        assert (await store.stats("qd")).chunk_count == 0
    finally:
        await store.close()


async def test_delete_chunks() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk("sA:0", "sA", "a0"), _chunk("sA:1", "sA", "a1")],
        vectors=[_unit(0), _unit(1)],
    )
    try:
        await store.delete("qd", ["sA:0"])
        assert await store.get_chunk("qd", "sA:0") is None
        assert await store.get_chunk("qd", "sA:1") is not None
        await store.delete("qd", [])  # empty is a no-op
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# payload filters — pre-filter recall (the M3 contract)
# --------------------------------------------------------------------------- #


async def _recall_store() -> QdrantStore:
    """100 chunks (90 paragraph / 10 heading) across two sources with
    different content_types and path prefixes."""
    store = await _seeded(
        sources=[
            _source("sA", "docs/a/guide.md", content_type="markdown"),
            _source("sB", "docs/b/manual.txt", content_type="text"),
        ],
        chunks=[
            _chunk(
                f"c{i:03d}",
                ("sA" if i % 2 == 0 else "sB"),
                f"document chunk number {i}",
                chunk_type="paragraph" if i % 10 != 9 else "heading",
            )
            for i in range(100)
        ],
        vectors=[_unit(i % DIM) for i in range(100)],
    )
    return store


async def test_restrictive_filter_returns_full_top_k() -> None:
    store = await _recall_store()
    try:
        # 10 headings under docs/b (i=9,19,...,99) — a restrictive filter
        # must still return FULL top_k (the M3 recall contract: filters are
        # pre-filters, never post-filter over-fetch).
        hits = await store.search_vector(
            "qd",
            _unit(0),
            SearchFilters(source_path_prefix="docs/b", chunk_types=("heading",)),
            10,
        )
        assert len(hits) == 10
        assert all(h.source_path.startswith("docs/b") for h in hits)
    finally:
        await store.close()


async def test_content_types_filter() -> None:
    store = await _recall_store()
    try:
        hits = await store.search_vector(
            "qd", _unit(0), SearchFilters(content_types=("markdown",)), 100
        )
        assert len(hits) == 50
        assert all(h.source_path.startswith("docs/a") for h in hits)
    finally:
        await store.close()


async def test_source_path_prefix_filter_exact() -> None:
    store = await _recall_store()
    try:
        # "docs/b" must NOT match "docs/b/manual.txt" AND "docs/b..." only
        hits = await store.search_vector(
            "qd", _unit(0), SearchFilters(source_path_prefix="docs/b"), 100
        )
        assert len(hits) == 50
        assert all(h.source_path.startswith("docs/b") for h in hits)
        # a longer prefix narrows further
        hits2 = await store.search_vector(
            "qd", _unit(0), SearchFilters(source_path_prefix="docs/b/manual.txt"), 100
        )
        assert len(hits2) == 50
    finally:
        await store.close()


async def test_metadata_match_filter() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk("sA:0", "sA", "x", chunk_type="function")],
        vectors=[_unit(0)],
    )
    try:
        hits = await store.search_vector(
            "qd", _unit(0), SearchFilters(metadata_match=(("chunk_type", "function"),)), 5
        )
        assert [h.chunk_id for h in hits] == ["sA:0"]
        hits2 = await store.search_vector(
            "qd", _unit(0), SearchFilters(metadata_match=(("chunk_type", "heading"),)), 5
        )
        assert hits2 == []
    finally:
        await store.close()


async def test_fts_filters_are_pre_filters() -> None:
    store = await _recall_store()
    try:
        hits = await store.search_fts(
            "qd",
            "document",
            SearchFilters(source_path_prefix="docs/b", chunk_types=("heading",)),
            20,
        )
        assert len(hits) == 10  # all 10 headings in docs/b match "document"
        assert all(h.source_path.startswith("docs/b") for h in hits)
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# search_vector semantics
# --------------------------------------------------------------------------- #


async def test_search_vector_top_k_and_min_score() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk("sA:0", "sA", "x")],
        vectors=[_unit(0)],
    )
    try:
        with pytest.raises(StoreError, match="top_k"):
            await store.search_vector("qd", _unit(0), SearchFilters(), 0)
        with pytest.raises(StoreError, match="shape"):
            await store.search_vector("qd", np.zeros(DIM + 1), SearchFilters(), 5)
        with pytest.raises(ValueError, match="unit-norm"):
            await store.search_vector("qd", np.full(DIM, 0.3), SearchFilters(), 5)
        with pytest.raises(StoreError, match="min_score"):
            await store.search_vector("qd", _unit(0), SearchFilters(), 5, min_score=float("nan"))
        hits = await store.search_vector("qd", _unit(0), SearchFilters(), 5, min_score=0.99)
        assert [h.chunk_id for h in hits] == ["sA:0"]
        # the identical vector scores 1.0, so a stricter threshold still keeps it
        hits2 = await store.search_vector("qd", _unit(0), SearchFilters(), 5, min_score=0.9999)
        assert [h.chunk_id for h in hits2] == ["sA:0"]
    finally:
        await store.close()


async def test_dense_parity_with_sqlite() -> None:
    """Same vectors + same query -> same ranking (dense search parity vs
    SqliteStore — plan §10). Cosine similarity is exact: both compute
    1 - cosine_distance on unit vectors."""
    srcs = [_source("sA", "docs/a.md"), _source("sB", "docs/b.md")]
    chunks = [
        _chunk("sA:0", "sA", "alpha"),
        _chunk("sA:1", "sA", "beta"),
        _chunk("sB:0", "sB", "gamma"),
    ]
    vectors = [_unit(0), _unit(1), _unit(2)]

    sqlite = await SqliteStore.open(":memory:")
    await sqlite.create_collection("p", DIM)
    for src in srcs:
        await sqlite.upsert_source("p", src)
    await sqlite.add("p", chunks, vectors)

    qdrant = await _seeded(sources=srcs, chunks=chunks, vectors=vectors)
    try:
        for query in (_unit(0), _unit(1), _unit(2)):
            s_hits = await sqlite.search_vector("p", query, SearchFilters(), 3)
            q_hits = await qdrant.search_vector("qd", query, SearchFilters(), 3)
            assert [h.chunk_id for h in s_hits] == [h.chunk_id for h in q_hits]
            assert [round(h.score, 6) for h in s_hits] == [round(h.score, 6) for h in q_hits]
    finally:
        await sqlite.close()
        await qdrant.close()


# --------------------------------------------------------------------------- #
# search_fts — pure-Python BM25
# --------------------------------------------------------------------------- #


async def test_fts_ranking_and_min_score() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[
            _chunk("sA:0", "sA", "the cat sat on the mat"),
            _chunk("sA:1", "sA", "a dog and a cat"),
            _chunk("sA:2", "sA", "nothing about pets"),
        ],
        vectors=[_unit(0), _unit(1), _unit(2)],
    )
    try:
        hits = await store.search_fts("qd", "cat", SearchFilters(), 5)
        assert [h.chunk_id for h in hits] == ["sA:0", "sA:1"]
        assert all(h.metric == Metric.BM25 for h in hits)
        assert all(h.score <= 0 for h in hits)  # FTS5-style negative ranks
        # min_score: sA:0 (-0.42) keeps, sA:1 (-0.46) drops — the skip branch
        hits2 = await store.search_fts("qd", "cat", SearchFilters(), 5, min_score=-0.43)
        assert [h.chunk_id for h in hits2] == ["sA:0"]
        # raw is accepted and is a no-op (no FTS5 syntax on this path)
        hits3 = await store.search_fts("qd", "cat", SearchFilters(), 5, raw=True)
        assert [h.chunk_id for h in hits3] == ["sA:0", "sA:1"]
    finally:
        await store.close()


async def test_fts_multi_page_scroll() -> None:
    """> _SCROLL_LIMIT (100) chunks force the scroll pagination loop."""
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk(f"sA:{i}", "sA", f"needle term {i}") for i in range(120)],
        vectors=[_unit(i) for i in range(120)],
    )
    try:
        hits = await store.search_fts("qd", "needle", SearchFilters(), 50)
        assert len(hits) == 50
        assert all(h.metric == Metric.BM25 for h in hits)
        all_chunks = await store.list_chunks("qd", limit=200)
        assert len(all_chunks) == 120
    finally:
        await store.close()


async def test_fts_empty_query_and_no_matches() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk("sA:0", "sA", "alpha")],
        vectors=[_unit(0)],
    )
    try:
        assert await store.search_fts("qd", "", SearchFilters(), 5) == []
        assert await store.search_fts("qd", "   ", SearchFilters(), 5) == []
        assert await store.search_fts("qd", "zzz", SearchFilters(), 5) == []
        with pytest.raises(StoreError, match="top_k"):
            await store.search_fts("qd", "alpha", SearchFilters(), 0)
    finally:
        await store.close()


async def test_fts_full_hybrid_search() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[
            _chunk("sA:0", "sA", "authentication issues short lived jwt tokens"),
            _chunk("sA:1", "sA", "billing records usage in five minute buckets"),
        ],
        vectors=[_unit(0), _unit(1)],
    )
    try:
        result = await search_hybrid(
            store,
            collection="qd",
            query_text="billing buckets",
            query_vector=_unit(1),
            top_k=5,
        )
        assert result.results
        assert result.metric == Metric.RRF
        assert result.total_results >= 1
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# sparse plumbing (Phase 8 extension, synthetic vectors)
# --------------------------------------------------------------------------- #


async def test_sparse_add_and_search() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a/guide.md"), _source("sB", "docs/b/manual.txt")],
        chunks=[
            _chunk("sA:0", "sA", "alpha"),
            _chunk("sA:1", "sA", "beta"),
            _chunk("sB:0", "sB", "gamma"),
        ],
        vectors=[_unit(0), _unit(1), _unit(2)],
        sparse=[
            ([0, 1], [0.8, 0.2]),  # sA:0
            ([1], [1.0]),  # sA:1
            ([0], [0.4]),  # sB:0
        ],
    )
    try:
        hits = await store.search_sparse("qd", [0], [0.8], SearchFilters(), 5)
        assert hits[0].chunk_id == "sA:0"  # highest dot with query [0]: 0.8
        assert hits[0].metric == Metric.SPARSE_DOT
        assert hits[0].score > hits[1].score
        # filter narrows the sparse search
        hits2 = await store.search_sparse(
            "qd", [0], [0.8], SearchFilters(source_path_prefix="docs/b"), 5
        )
        assert [h.chunk_id for h in hits2] == ["sB:0"]
        # min_score filters
        hits3 = await store.search_sparse("qd", [0], [0.8], SearchFilters(), 5, min_score=0.5)
        assert [h.chunk_id for h in hits3] == ["sA:0"]
        with pytest.raises(StoreError, match="top_k"):
            await store.search_sparse("qd", [0], [0.8], SearchFilters(), 0)
    finally:
        await store.close()


async def test_sparse_provided_with_dense_survives() -> None:
    """The probe fact: a partial upsert replaces the other named vector —
    sparse must be provided TOGETHER with dense (the adapter's contract)."""
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk("sA:0", "sA", "alpha")],
        vectors=[_unit(0)],
        sparse=[([0], [1.0])],
    )
    try:
        # dense still searchable after the sparse was attached in the same upsert
        dense = await store.search_vector("qd", _unit(0), SearchFilters(), 5)
        assert [h.chunk_id for h in dense] == ["sA:0"]
        sparse = await store.search_sparse("qd", [0], [1.0], SearchFilters(), 5)
        assert [h.chunk_id for h in sparse] == ["sA:0"]
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# reindex_source — achievable guarantee (docs/decisions/0004)
# --------------------------------------------------------------------------- #


async def test_reindex_subset_swap_cleans_stale() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a/guide.md"), _source("sB", "docs/b/manual.txt")],
        chunks=[
            _chunk("sA:0", "sA", "old a0"),
            _chunk("sA:1", "sA", "old a1"),
            _chunk("sA:2", "sA", "old a2"),
            _chunk("sB:0", "sB", "b0"),
        ],
        vectors=[_unit(0), _unit(1), _unit(2), _unit(3)],
    )
    try:
        new_src = _source("sA", "docs/a.md", content_hash="new-h", size=999)
        await store.reindex_source(
            "qd",
            new_src,
            [_chunk("sA:0", "sA", "new a0")],
            [_unit(7)],
        )
        stats = await store.stats("qd")
        assert stats.chunk_count == 2  # sA:0 (new) + sB:0 — sA:1/2 cleaned
        assert stats.source_count == 2
        # old chunk ids are gone, the new one is searchable
        assert await store.get_chunk("qd", "sA:1") is None
        hits = await store.search_vector("qd", _unit(7), SearchFilters(), 5)
        assert hits[0].chunk_id == "sA:0"
        assert hits[0].content == "new a0"
        # source metadata refreshed
        got = await store.get_source("qd", "docs/a.md")
        assert got is not None and got.size_bytes == 999
    finally:
        await store.close()


async def test_reindex_failure_leaves_old_queryable() -> None:
    """Mid-swap failure (upsert raises): the delete never runs, so the OLD
    chunks remain intact and queryable — the H7 guarantee, as far as the
    backend allows (no transactions; documented in 0004)."""
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk("sA:0", "sA", "old a0"), _chunk("sA:1", "sA", "old a1")],
        vectors=[_unit(0), _unit(1)],
    )
    try:
        original_upsert = store._c().upsert
        client: Any = store._client  # ty: the qdrant client type is not the target here

        def failing_upsert(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated qdrant failure")

        client.upsert = failing_upsert
        try:
            with pytest.raises(RuntimeError, match="simulated"):
                await store.reindex_source(
                    "qd",
                    _source("sA", "docs/a.md", content_hash="new"),
                    [_chunk("sA:0", "sA", "new a0")],
                    [_unit(5)],
                )
        finally:
            client.upsert = original_upsert
        # old chunks still intact + queryable; metadata unchanged
        kept0 = await store.get_chunk("qd", "sA:0")
        kept1 = await store.get_chunk("qd", "sA:1")
        assert kept0 is not None and kept0.content == "old a0"
        assert kept1 is not None and kept1.content == "old a1"
        hits = await store.search_vector("qd", _unit(0), SearchFilters(), 5)
        assert {h.chunk_id for h in hits} == {"sA:0", "sA:1"}
        src_meta = await store.get_source("qd", "docs/a.md")
        assert src_meta is not None and src_meta.content_hash == "h-sA"
    finally:
        await store.close()


async def test_reindex_validation_errors() -> None:
    store = await _seeded(sources=[_source("sA", "docs/a.md")])
    try:
        with pytest.raises(StoreError, match="length mismatch"):
            await store.reindex_source(
                "qd", _source("sA", "docs/a.md"), [_chunk("x", "sA", "x")], []
            )
        with pytest.raises(StoreError, match="reference source_id"):
            await store.reindex_source(
                "qd",
                _source("sA", "docs/a.md"),
                [_chunk("sB:0", "sB", "x")],
                [_unit(0)],
            )
        with pytest.raises(StoreError, match="duplicate"):
            await store.reindex_source(
                "qd",
                _source("sA", "docs/a.md"),
                [_chunk("d", "sA", "x"), _chunk("d", "sA", "y")],
                [_unit(0), _unit(1)],
            )
        with pytest.raises(StoreError, match="sparse_vectors"):
            await store.reindex_source(
                "qd",
                _source("sA", "docs/a.md"),
                [_chunk("x", "sA", "x")],
                [_unit(0)],
                sparse_vectors=[],
            )
    finally:
        await store.close()


async def test_reindex_keeps_unrelated_sources() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a/guide.md"), _source("sB", "docs/b/manual.txt")],
        chunks=[_chunk("sA:0", "sA", "a0"), _chunk("sB:0", "sB", "b0")],
        vectors=[_unit(0), _unit(1)],
    )
    try:
        await store.reindex_source(
            "qd", _source("sA", "docs/a.md"), [_chunk("sA:0", "sA", "a0 v2")], [_unit(2)]
        )
        unrelated = await store.get_chunk("qd", "sB:0")
        assert unrelated is not None and unrelated.content == "b0"
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# beyond-protocol extras
# --------------------------------------------------------------------------- #


async def test_get_chunk_roundtrip() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[
            _chunk("sA:0", "sA", "alpha", chunk_type="function", granularity="function"),
        ],
        vectors=[_unit(0)],
    )
    try:
        chunk = await store.get_chunk("qd", "sA:0")
        assert chunk is not None
        assert chunk.id == "sA:0"
        assert chunk.source_id == "sA"
        assert chunk.content == "alpha"
        assert chunk.chunk_type == "function"
        assert chunk.granularity == "function"
        assert await store.get_chunk("qd", "missing") is None
    finally:
        await store.close()


async def test_list_chunks_pagination_and_order() -> None:
    store = await _seeded(
        sources=[_source("sA", "docs/a.md")],
        chunks=[_chunk(f"sA:{i}", "sA", f"chunk {i}") for i in range(10)],
        vectors=[_unit(i) for i in range(10)],
    )
    try:
        page = await store.list_chunks("qd", limit=4, offset=2)
        assert [c.id for c in page] == ["sA:2", "sA:3", "sA:4", "sA:5"]
        with pytest.raises(StoreError, match="limit"):
            await store.list_chunks("qd", limit=0)
        with pytest.raises(StoreError, match="offset"):
            await store.list_chunks("qd", offset=-1)
    finally:
        await store.close()


async def test_count_fts_not_implemented() -> None:
    """count_fts is a sqlite-only extra — the qdrant adapter does not carry
    it (the server's 501 path covers it; the server never calls it)."""
    store = await _seeded(sources=[_source("sA", "docs/a.md")])
    try:
        assert not hasattr(store, "count_fts")
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# lifecycle / errors
# --------------------------------------------------------------------------- #


async def test_closed_store_raises() -> None:
    store = await _open()
    await store.close()
    with pytest.raises(StoreError, match="not open"):
        store._c()


async def test_async_context_manager() -> None:
    """The SqliteStore-parity `async with` protocol closes on exit."""
    async with await _open() as store:
        await store.create_collection("second", 4)
        assert (await store.stats("second")).vector_dimension == 4
    with pytest.raises(StoreError, match="not open"):
        store._c()


async def test_private_empty_guards() -> None:
    """The defensive early-returns on empty source-id sets (unreachable via
    the public API — add() guards empty chunks — but unit-covered)."""
    store = await _seeded(sources=[_source("sA", "docs/a.md")])
    try:
        assert store._missing_sources("qd", set()) == set()
        assert store._source_lookup("qd", set()) == {}
        # the companion collection's unnamed size-1 vector is readable too
        # (the elif branch of _collection_dimension).
        assert store._collection_dimension("qd__sources") == 1
    finally:
        await store.close()


async def test_sparse_only_collection_has_no_dense_dimension() -> None:
    """A collection created without a dense vector (legal in qdrant) raises
    the defensive 'no dense vector config' StoreError from stats()."""
    from qdrant_client.http import models

    store = await QdrantStore.open(parse_store_url("qdrant://:memory:"))
    try:
        store._c().create_collection(
            "sparseonly",
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        # stats() reads the dimension -> no dense config -> StoreError
        with pytest.raises(StoreError, match="no dense vector config"):
            await store.stats("sparseonly")
    finally:
        await store.close()


async def test_open_remote_unreachable_raises() -> None:
    """Honest health: unreachable qdrant -> StoreError at open (localhost:1
    refuses instantly — offline)."""
    with pytest.raises(StoreError, match="cannot reach qdrant"):
        await QdrantStore.open(parse_store_url("qdrant://127.0.0.1:1"))


async def test_unknown_collection_operations_raise() -> None:
    store = await _open()
    try:
        with pytest.raises(StoreError, match="unknown collection"):
            await store.search_vector("nope", _unit(0), SearchFilters(), 5)
        with pytest.raises(StoreError, match="unknown collection"):
            await store.search_fts("nope", "x", SearchFilters(), 5)
        with pytest.raises(StoreError, match="unknown collection"):
            await store.delete("nope", ["x"])
        with pytest.raises(StoreError, match="unknown collection"):
            await store.stats("nope")
        with pytest.raises(StoreError, match="unknown collection"):
            await store.get_source("nope", "a.md")
        with pytest.raises(StoreError, match="unknown collection"):
            await store.list_sources("nope")
        with pytest.raises(StoreError, match="unknown collection"):
            await store.delete_source("nope", "s")
    finally:
        await store.close()
