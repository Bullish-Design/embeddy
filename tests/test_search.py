# tests/test_search.py
"""Tests for the search layer.

TDD: Tests are written before the implementation.
Covers: SearchService construction, vector search, fulltext search, hybrid
search (RRF + weighted fusion), find_similar, min_score filtering, error
handling, SearchResults construction.

All tests mock the Embedder and VectorStore to avoid heavy dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from embeddy.models import (
    Embedding,
    FusionStrategy,
    SearchFilters,
    SearchMode,
    SearchResult,
    SearchResults,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_embedder(dimension: int = 128, model_name: str = "test-model"):
    """Create a mock Embedder for search tests."""
    embedder = AsyncMock()
    embedder.dimension = dimension
    embedder.model_name = model_name

    async def _encode(texts):
        return [
            Embedding(
                vector=[0.1] * dimension,
                model_name=model_name,
                normalized=True,
            )
            for _ in texts
        ]

    embedder.encode.side_effect = _encode

    async def _encode_query(text):
        return Embedding(
            vector=[0.1] * dimension,
            model_name=model_name,
            normalized=True,
        )

    embedder.encode_query = AsyncMock(side_effect=_encode_query)
    return embedder


def _make_knn_results(n: int = 5) -> list[dict]:
    """Create mock KNN results from VectorStore.search_knn."""
    return [
        {
            "chunk_id": f"chunk-{i}",
            "score": 1.0 - (i * 0.1),
            "content": f"Content of chunk {i}",
            "content_type": "python",
            "chunk_type": "function",
            "source_path": f"/src/file_{i}.py",
            "start_line": i * 10,
            "end_line": (i + 1) * 10,
            "name": f"func_{i}",
            "parent": None,
            "metadata": {},
            "content_hash": f"hash_{i}",
        }
        for i in range(n)
    ]


def _make_fts_results(n: int = 5, offset: int = 0) -> list[dict]:
    """Create mock FTS results from VectorStore.search_fts.

    offset allows creating results with different chunk IDs for testing fusion.
    """
    return [
        {
            "chunk_id": f"chunk-{i + offset}",
            "score": 2.0 - (i * 0.2),
            "content": f"Content of chunk {i + offset}",
            "content_type": "python",
            "chunk_type": "function",
            "source_path": f"/src/file_{i + offset}.py",
            "start_line": (i + offset) * 10,
            "end_line": (i + offset + 1) * 10,
            "name": f"func_{i + offset}",
            "parent": None,
            "metadata": {},
            "content_hash": f"hash_{i + offset}",
        }
        for i in range(n)
    ]


def _make_mock_store(knn_results=None, fts_results=None):
    """Create a mock VectorStore with search methods."""
    store = AsyncMock()
    store.search_knn = AsyncMock(return_value=knn_results or [])
    store.search_fts = AsyncMock(return_value=fts_results or [])
    store.get = AsyncMock(return_value=None)
    return store


# ---------------------------------------------------------------------------
# SearchService — construction
# ---------------------------------------------------------------------------


class TestSearchServiceConstruction:
    """Tests for SearchService instantiation."""

    def test_accepts_embedder_and_store(self):
        """SearchService can be constructed with embedder and store."""
        from embeddy.search.search_service import SearchService

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        svc = SearchService(embedder=embedder, store=store)
        assert svc is not None

    def test_stores_dependencies(self):
        """SearchService stores the injected embedder and store."""
        from embeddy.search.search_service import SearchService

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        svc = SearchService(embedder=embedder, store=store)
        assert svc._embedder is embedder
        assert svc._store is store


# ---------------------------------------------------------------------------
# SearchService — vector search
# ---------------------------------------------------------------------------


class TestSearchVector:
    """Tests for pure vector (semantic) search."""

    async def test_vector_search_returns_search_results(self):
        """search_vector returns a SearchResults object."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(3)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("test query", collection="test")
        assert isinstance(results, SearchResults)
        assert len(results.results) == 3
        assert results.mode == SearchMode.VECTOR

    async def test_vector_search_calls_encode_query(self):
        """search_vector calls embedder.encode_query for the query."""
        from embeddy.search.search_service import SearchService

        store = _make_mock_store(knn_results=_make_knn_results(1))
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        await svc.search_vector("hello world", collection="test")
        embedder.encode_query.assert_called_once()

    async def test_vector_search_calls_store_knn(self):
        """search_vector delegates to store.search_knn."""
        from embeddy.search.search_service import SearchService

        store = _make_mock_store(knn_results=_make_knn_results(2))
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        await svc.search_vector("query", collection="my_coll", top_k=5)
        store.search_knn.assert_called_once()
        call_args = store.search_knn.call_args
        assert call_args[1]["collection_name"] == "my_coll" or call_args[0][0] == "my_coll"

    async def test_vector_search_results_sorted_descending(self):
        """Results are sorted by score descending."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(5)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("query", collection="test")
        scores = [r.score for r in results.results]
        assert scores == sorted(scores, reverse=True)

    async def test_vector_search_passes_filters(self):
        """search_vector passes filters to store.search_knn."""
        from embeddy.search.search_service import SearchService

        store = _make_mock_store(knn_results=[])
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        filters = SearchFilters(source_path_prefix="/src/")
        await svc.search_vector("query", collection="test", filters=filters)
        call_kwargs = store.search_knn.call_args
        # Filters should be passed through
        assert call_kwargs[1].get("filters") is filters or (len(call_kwargs[0]) > 3 and call_kwargs[0][3] is filters)

    async def test_vector_search_respects_top_k(self):
        """search_vector limits results to top_k."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(10)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("query", collection="test", top_k=3)
        assert len(results.results) <= 3


# ---------------------------------------------------------------------------
# SearchService — fulltext search
# ---------------------------------------------------------------------------


class TestSearchFulltext:
    """Tests for pure full-text (BM25) search."""

    async def test_fulltext_search_returns_search_results(self):
        """search_fulltext returns a SearchResults object."""
        from embeddy.search.search_service import SearchService

        fts = _make_fts_results(3)
        store = _make_mock_store(fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_fulltext("test", collection="test")
        assert isinstance(results, SearchResults)
        assert len(results.results) == 3
        assert results.mode == SearchMode.FULLTEXT

    async def test_fulltext_does_not_call_embedder(self):
        """search_fulltext does NOT call the embedder (no encoding needed)."""
        from embeddy.search.search_service import SearchService

        fts = _make_fts_results(2)
        store = _make_mock_store(fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        await svc.search_fulltext("keyword", collection="test")
        embedder.encode_query.assert_not_called()
        embedder.encode.assert_not_called()

    async def test_fulltext_calls_store_fts(self):
        """search_fulltext delegates to store.search_fts."""
        from embeddy.search.search_service import SearchService

        store = _make_mock_store(fts_results=_make_fts_results(1))
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        await svc.search_fulltext("keyword", collection="coll1", top_k=7)
        store.search_fts.assert_called_once()

    async def test_fulltext_results_sorted_descending(self):
        """FTS results are sorted by score descending."""
        from embeddy.search.search_service import SearchService

        fts = _make_fts_results(4)
        store = _make_mock_store(fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_fulltext("query", collection="test")
        scores = [r.score for r in results.results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# SearchService — hybrid search
# ---------------------------------------------------------------------------


class TestSearchHybrid:
    """Tests for hybrid (vector + fulltext) search with score fusion."""

    async def test_hybrid_search_calls_both_backends(self):
        """Hybrid search calls both search_knn and search_fts."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(3)
        fts = _make_fts_results(3, offset=3)
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        await svc.search("query", collection="test", mode=SearchMode.HYBRID)
        store.search_knn.assert_called_once()
        store.search_fts.assert_called_once()

    async def test_hybrid_returns_search_results(self):
        """Hybrid search returns a SearchResults object."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(3)
        fts = _make_fts_results(3, offset=3)
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.HYBRID)
        assert isinstance(results, SearchResults)
        assert results.mode == SearchMode.HYBRID

    async def test_hybrid_deduplicates_results(self):
        """Hybrid search deduplicates results that appear in both KNN and FTS."""
        from embeddy.search.search_service import SearchService

        # Create overlapping results — chunk-0 appears in both
        knn = _make_knn_results(3)  # chunk-0, chunk-1, chunk-2
        fts = _make_fts_results(3, offset=0)  # chunk-0, chunk-1, chunk-2 (same IDs)
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.HYBRID)
        # Should be deduplicated — no duplicate chunk_ids
        chunk_ids = [r.chunk_id for r in results.results]
        assert len(chunk_ids) == len(set(chunk_ids))

    async def test_hybrid_rrf_fusion(self):
        """RRF fusion produces valid fused scores."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(3)
        fts = _make_fts_results(3, offset=2)  # partial overlap
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search(
            "query",
            collection="test",
            mode=SearchMode.HYBRID,
            fusion=FusionStrategy.RRF,
        )
        assert len(results.results) > 0
        # Results should be sorted descending
        scores = [r.score for r in results.results]
        assert scores == sorted(scores, reverse=True)

    async def test_hybrid_weighted_fusion(self):
        """Weighted fusion produces valid fused scores."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(3)
        fts = _make_fts_results(3, offset=2)
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search(
            "query",
            collection="test",
            mode=SearchMode.HYBRID,
            fusion=FusionStrategy.WEIGHTED,
            hybrid_alpha=0.7,
        )
        assert len(results.results) > 0
        scores = [r.score for r in results.results]
        assert scores == sorted(scores, reverse=True)

    async def test_hybrid_respects_top_k(self):
        """Hybrid search limits final results to top_k."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(5)
        fts = _make_fts_results(5, offset=5)  # 10 unique chunks total
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.HYBRID, top_k=3)
        assert len(results.results) <= 3

    async def test_hybrid_empty_knn_still_returns_fts(self):
        """If KNN returns nothing, hybrid still returns FTS results."""
        from embeddy.search.search_service import SearchService

        fts = _make_fts_results(3)
        store = _make_mock_store(knn_results=[], fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.HYBRID)
        assert len(results.results) > 0

    async def test_hybrid_empty_fts_still_returns_knn(self):
        """If FTS returns nothing, hybrid still returns KNN results."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(3)
        store = _make_mock_store(knn_results=knn, fts_results=[])
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.HYBRID)
        assert len(results.results) > 0


# ---------------------------------------------------------------------------
# SearchService — main search() dispatch
# ---------------------------------------------------------------------------


class TestSearchDispatch:
    """Tests for the main search() method that dispatches by mode."""

    async def test_search_vector_mode(self):
        """search(mode=VECTOR) uses only vector search."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(2)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.VECTOR)
        assert results.mode == SearchMode.VECTOR
        store.search_knn.assert_called_once()
        store.search_fts.assert_not_called()

    async def test_search_fulltext_mode(self):
        """search(mode=FULLTEXT) uses only fulltext search."""
        from embeddy.search.search_service import SearchService

        fts = _make_fts_results(2)
        store = _make_mock_store(fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test", mode=SearchMode.FULLTEXT)
        assert results.mode == SearchMode.FULLTEXT
        store.search_fts.assert_called_once()
        store.search_knn.assert_not_called()

    async def test_search_default_mode_is_hybrid(self):
        """search() defaults to hybrid mode."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(2)
        fts = _make_fts_results(2, offset=2)
        store = _make_mock_store(knn_results=knn, fts_results=fts)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("query", collection="test")
        assert results.mode == SearchMode.HYBRID


# ---------------------------------------------------------------------------
# SearchService — min_score filtering
# ---------------------------------------------------------------------------


class TestSearchMinScore:
    """Tests for min_score threshold filtering."""

    async def test_min_score_filters_low_scores(self):
        """Results below min_score are excluded."""
        from embeddy.search.search_service import SearchService

        # Create results with scores: 1.0, 0.9, 0.8, 0.7, 0.6
        knn = _make_knn_results(5)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("query", collection="test", min_score=0.85)
        for r in results.results:
            assert r.score >= 0.85

    async def test_no_min_score_returns_all(self):
        """Without min_score, all results are returned."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(5)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("query", collection="test")
        assert len(results.results) == 5


# ---------------------------------------------------------------------------
# SearchService — find_similar
# ---------------------------------------------------------------------------


class TestFindSimilar:
    """Tests for finding chunks similar to an existing chunk."""

    async def test_find_similar_returns_results(self):
        """find_similar returns SearchResults."""
        from embeddy.search.search_service import SearchService

        # Mock store.get to return a chunk with its vector
        knn = _make_knn_results(3)
        store = _make_mock_store(knn_results=knn)
        store.get = AsyncMock(
            return_value={
                "chunk_id": "target-chunk",
                "content": "Some content",
                "content_type": "python",
                "chunk_type": "function",
                "source_path": "/src/file.py",
                "start_line": 1,
                "end_line": 10,
                "name": "func",
                "parent": None,
                "metadata": {},
                "content_hash": "hash",
            }
        )
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.find_similar(
            chunk_id="target-chunk",
            collection="test",
            top_k=3,
        )
        assert isinstance(results, SearchResults)

    async def test_find_similar_excludes_self(self):
        """find_similar excludes the source chunk by default."""
        from embeddy.search.search_service import SearchService

        # KNN results include the source chunk itself
        knn = [
            {
                "chunk_id": "target-chunk",
                "score": 1.0,
                "content": "Self",
                "content_type": "python",
                "chunk_type": "function",
                "source_path": "/src/file.py",
                "start_line": 1,
                "end_line": 10,
                "name": "func",
                "parent": None,
                "metadata": {},
                "content_hash": "hash",
            },
            *_make_knn_results(3),
        ]
        store = _make_mock_store(knn_results=knn)
        store.get = AsyncMock(
            return_value={
                "chunk_id": "target-chunk",
                "content": "Self",
                "content_type": "python",
                "chunk_type": "function",
                "source_path": "/src/file.py",
                "start_line": 1,
                "end_line": 10,
                "name": "func",
                "parent": None,
                "metadata": {},
                "content_hash": "hash",
            }
        )
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.find_similar("target-chunk", collection="test")
        chunk_ids = [r.chunk_id for r in results.results]
        assert "target-chunk" not in chunk_ids

    async def test_find_similar_nonexistent_chunk(self):
        """find_similar for a nonexistent chunk returns empty results."""
        from embeddy.search.search_service import SearchService

        store = _make_mock_store()
        store.get = AsyncMock(return_value=None)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.find_similar("nonexistent", collection="test")
        assert len(results.results) == 0


# ---------------------------------------------------------------------------
# SearchService — error handling
# ---------------------------------------------------------------------------


class TestSearchErrors:
    """Tests for error handling in search."""

    async def test_empty_query_returns_empty_results(self):
        """An empty query string returns empty results."""
        from embeddy.search.search_service import SearchService

        store = _make_mock_store()
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search("", collection="test")
        assert len(results.results) == 0

    async def test_search_elapsed_ms(self):
        """SearchResults includes elapsed_ms >= 0."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(2)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("query", collection="test")
        assert results.elapsed_ms >= 0.0

    async def test_search_populates_query_and_collection(self):
        """SearchResults has the query and collection fields set."""
        from embeddy.search.search_service import SearchService

        knn = _make_knn_results(1)
        store = _make_mock_store(knn_results=knn)
        embedder = _make_mock_embedder()
        svc = SearchService(embedder=embedder, store=store)

        results = await svc.search_vector("my query", collection="my_coll")
        assert results.query == "my query"
        assert results.collection == "my_coll"
