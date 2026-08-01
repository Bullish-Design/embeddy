"""Unit tests for search_hybrid — the orchestration layer over Searchable.

Uses a scripted stub store (no sqlite) so the pure orchestration logic —
leg retrieval, fusion mode selection, total_results semantics, the rerank
stage — is tested in isolation. Real-store hybrid behavior lives in
tests/test_store_phase4.py ([integration]).
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddy.protocol.rerank import RerankerProvider, RerankHit
from embeddy.protocol.types import CollectionStats, Metric, ScoredDocument, Vector
from embeddy.search import fuse_rrf, fuse_weighted, search_hybrid


def _hit(chunk_id: str, score: float, metric: Metric) -> ScoredDocument:
    return ScoredDocument(
        chunk_id=chunk_id,
        collection_id="acme",
        source_id="s1",
        source_path="a.md",
        content=f"content-{chunk_id}",
        score=score,
        metric=metric,
    )


class StubStore:
    """Scripted Searchable: returns pre-arranged leg results and records the
    arguments each leg was called with (structural match to Searchable)."""

    def __init__(self, vec: list[ScoredDocument], fts: list[ScoredDocument]) -> None:
        self.vec = vec
        self.fts = fts
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def search_vector(self, collection, query_vector, filters, top_k, *, min_score=None):
        self.calls.append(("vector", (collection, top_k, min_score)))
        return self.vec[:top_k]

    async def search_fts(self, collection, query, filters, top_k, *, min_score=None, raw=False):
        self.calls.append(("fts", (collection, top_k, min_score, raw)))
        return self.fts[:top_k]

    # not used by search_hybrid — declared so the stub is a complete Searchable;
    # raising keeps the runtime honest if a future test calls them.
    async def add(self, collection: str, chunks: list, vectors: list) -> None:
        raise NotImplementedError

    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    async def stats(self, collection: str) -> CollectionStats:
        raise NotImplementedError

    async def upsert_source(self, collection: str, source: object) -> str:
        raise NotImplementedError

    async def get_source(self, collection: str, path: str) -> None:
        raise NotImplementedError

    async def reindex_source(
        self, collection: str, source: object, chunks: list, vectors: list
    ) -> None:
        raise NotImplementedError

    async def delete_source(self, collection: str, source_id: str) -> None:
        raise NotImplementedError

    async def list_sources(self, collection: str) -> list:
        raise NotImplementedError


def _query_vector() -> Vector:
    return np.ones(4, dtype=np.float32) / 2.0


async def test_rrf_fusion_and_total_results() -> None:
    vec = [_hit("a", 0.9, Metric.COSINE), _hit("b", 0.8, Metric.COSINE)]
    fts = [_hit("b", -1.0, Metric.BM25), _hit("c", -2.0, Metric.BM25)]
    store = StubStore(vec, fts)
    res = await search_hybrid(
        store, collection="acme", query_text="alpha", query_vector=_query_vector(), top_k=3
    )
    # RRF over [[a,b],[b,c]]: b=2/61, a=c=1/62 -> b, then a/c (insertion order)
    assert [h.chunk_id for h in res.results] == ["b", "a", "c"]
    assert res.metric == Metric.RRF
    assert res.mode == "rrf"
    assert res.total_results == 3  # union {a, b, c}
    assert store.calls == [("vector", ("acme", 50, None)), ("fts", ("acme", 50, None, False))]


async def test_weighted_mode_and_weights() -> None:
    vec = [_hit("a", 0.9, Metric.COSINE)]
    fts = [_hit("b", -1.0, Metric.BM25)]
    store = StubStore(vec, fts)
    res = await search_hybrid(
        store,
        collection="acme",
        query_text="q",
        query_vector=_query_vector(),
        top_k=2,
        mode="weighted",
        weights=(2.0, 1.0),
    )
    assert res.metric == Metric.WEIGHTED
    assert res.mode == "weighted"
    assert {h.chunk_id for h in res.results} == {"a", "b"}


async def test_retrieve_k_and_min_score_passthrough() -> None:
    store = StubStore([], [])
    await search_hybrid(
        store,
        collection="acme",
        query_text="q",
        query_vector=_query_vector(),
        top_k=5,
        retrieve_k=20,
        min_score=0.7,
        raw=True,
    )
    assert store.calls == [("vector", ("acme", 20, 0.7)), ("fts", ("acme", 20, 0.7, True))]


class ScriptedReranker(RerankerProvider):
    model_name = "scripted"

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.last_documents: list[str] | None = None
        self.last_top_k: int | None = None

    async def rerank(self, query, documents, top_k=None, instruction=None):
        self.last_documents = documents
        self.last_top_k = top_k
        hits = [RerankHit(index=i, score=s) for i, s in self.scores.items()]
        hits.sort(key=lambda hit: hit.score, reverse=True)  # contract: descending
        return hits[:top_k] if top_k is not None else hits


async def test_rerank_stage_reorders_and_marks_metric() -> None:
    vec = [_hit("a", 0.9, Metric.COSINE), _hit("b", 0.8, Metric.COSINE)]
    fts = [_hit("b", -1.0, Metric.BM25), _hit("c", -2.0, Metric.BM25)]
    store = StubStore(vec, fts)
    # fused (RRF) order is [b, a, c] (indices 0,1,2); the reranker decides the
    # cross-encoder prefers c > a > b, flipping the fusion order.
    reranker = ScriptedReranker(scores={2: 3.0, 1: 2.0, 0: 1.0})
    res = await search_hybrid(
        store,
        collection="acme",
        query_text="q",
        query_vector=_query_vector(),
        top_k=3,
        reranker=reranker,
        rerank_top_k=3,
        instruction="query-instr",
    )
    assert [h.chunk_id for h in res.results] == ["c", "a", "b"]
    assert all(h.metric == Metric.RERANK for h in res.results)
    assert res.metric == Metric.RERANK
    assert res.total_results == 3  # preserved from pre-rerank candidates
    assert reranker.last_documents == [
        "content-b",
        "content-a",
        "content-c",
    ]  # fused order before rerank
    assert reranker.last_top_k == 3


async def test_rerank_truncates_to_top_k() -> None:
    vec = [_hit("a", 0.9, Metric.COSINE), _hit("b", 0.8, Metric.COSINE)]
    store = StubStore(vec, [])
    reranker = ScriptedReranker(scores={0: 2.0, 1: 1.0})
    res = await search_hybrid(
        store,
        collection="acme",
        query_text="q",
        query_vector=_query_vector(),
        top_k=1,
        reranker=reranker,
    )
    assert len(res.results) == 1
    assert res.results[0].chunk_id == "a"
    assert reranker.last_top_k == 1  # default: rerank_top_k == top_k


async def test_validation_errors() -> None:
    store = StubStore([], [])
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        await search_hybrid(
            store, collection="c", query_text="q", query_vector=_query_vector(), top_k=0
        )
    with pytest.raises(ValueError, match="retrieve_k"):
        await search_hybrid(
            store,
            collection="c",
            query_text="q",
            query_vector=_query_vector(),
            top_k=10,
            retrieve_k=5,
        )
    with pytest.raises(ValueError, match="mode"):
        await search_hybrid(
            store, collection="c", query_text="q", query_vector=_query_vector(), mode="minmax"
        )


async def test_empty_inputs() -> None:
    assert fuse_rrf([]) == []
    assert fuse_weighted([]) == []
    store = StubStore([], [])
    res = await search_hybrid(store, collection="c", query_text="q", query_vector=_query_vector())
    assert res.results == []
    assert res.total_results == 0


async def test_fuse_weighted_mixed_empty_lists() -> None:
    """A non-empty list followed by an empty list must not break min-max
    normalization (the empty list contributes nothing)."""
    vec = [_hit("a", 0.9, Metric.COSINE)]
    fused = fuse_weighted([vec, []])
    assert [h.chunk_id for h in fused] == ["a"]
    assert fused[0].score == pytest.approx(1.0 * 0.5, abs=1e-9)  # equal weights, single list
    # and the empty-first order
    assert [h.chunk_id for h in fuse_weighted([[], vec])] == ["a"]
