"""Keystone vertical slice (plan §3): text -> ingest -> chunk -> fake vectors
-> store -> hybrid search -> assert. Tagged [e2e].

Covers the Phase-1 acceptance points:
  * cosine score semantics correct (identical unit vector -> score 1.0)
  * no dict[str, Any] in public results (typed records end to end)
  * RRF / weighted fusion over typed results
  * MRL resolve_dimension matrix + truncate_and_renormalize
  * ValidatedChunker invariants (non-empty, token budget, line ranges)
"""

from __future__ import annotations

import numpy as np
import pytest

from chonkai import ChunkBudget, IngestResult, ParagraphChunker, ValidatedChunker
from embeddy import (
    Metric,
    ModelSpec,
    ScoredDocument,
    assert_unit_vector,
    fuse_rrf,
    fuse_weighted,
    resolve_dimension,
    truncate_and_renormalize,
)
from embeddy.index.base import SearchFilters
from embeddy.index.sqlite import SqliteStore, StoreError
from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.protocol.types import SourceMetadata, StoredChunk
from embeddy.providers.fake import FakeProvider

pytestmark = pytest.mark.e2e

CORPUS = """\
The acme authentication service issues short-lived JWTs for API access.
Tokens expire after 15 minutes and are refreshed via a rotating key.

The acme billing API records usage in 5-minute buckets and invoices monthly.
Refunds are processed within three business days of the request.

The acme search index stores embeddings in an embedded sqlite-vec store.
Hybrid search combines dense vector similarity with full-text ranking.
"""


def _ingest_and_chunk(text: str) -> tuple[IngestResult, list]:
    ingest = IngestResult.from_text(text, path="docs/guide.md", content_type="markdown")
    chunker = ValidatedChunker(
        ParagraphChunker(),
        budget=ChunkBudget(max_tokens=128),
        token_counter=lambda s: len(s.split()),
    )
    return ingest, chunker.chunk(ingest)


async def _seed_store(store: SqliteStore) -> tuple[FakeProvider, list[StoredChunk], list]:
    """Upsert one source and add all corpus chunks+vectors; return provider,
    stored chunks and vectors for reuse."""
    ingest, chunks = _ingest_and_chunk(CORPUS)
    provider = FakeProvider()
    vectors = await provider.encode([c.content for c in chunks])
    stored = [
        StoredChunk(
            id=f"src1:{i}",
            collection_id="acme",
            source_id="src1",
            content=c.content,
            chunk_type=c.chunk_type,
            start_line=c.start_line,
            end_line=c.end_line,
            parent=c.parent,
            granularity=None,
            token_count=c.token_count,
            vector=v,
        )
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    await store.create_collection("acme", provider.dimension)
    await store.upsert_source(
        "acme",
        SourceMetadata(
            id="src1",
            collection_id="acme",
            path="docs/guide.md",
            content_hash=ingest.source.content_hash or "",
            size_bytes=ingest.source.size_bytes,
        ),
    )
    await store.add("acme", stored, vectors)
    return provider, stored, vectors


async def test_keystone_vertical_slice() -> None:
    ingest, chunks = _ingest_and_chunk(CORPUS)
    assert len(chunks) >= 3, "corpus should split into >= 3 paragraphs"
    # ValidatedChunker invariants
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.start_line >= 1 and chunk.end_line >= chunk.start_line
        assert chunk.chunk_type == "paragraph"
        assert 0 < chunk.token_count <= 128

    provider = FakeProvider()
    assert isinstance(provider, EmbeddingProvider), "FakeProvider must satisfy the protocol"

    async with await SqliteStore.open(":memory:") as store:
        _, stored, vectors = await _seed_store(store)

        # typed vectors: float32, dim 8, unit-norm
        for v in vectors:
            assert v.shape == (8,)
            assert v.dtype == np.float32
            assert_unit_vector(v)

        stats = await store.stats("acme")
        assert stats.chunk_count == len(chunks)
        assert stats.source_count == 1
        assert stats.vector_dimension == 8
        assert stats.size_bytes == ingest.source.size_bytes

        # --- vector search: cosine semantics ---
        exact = (await provider.encode([chunks[0].content]))[0]
        hits = await store.search_vector("acme", exact, SearchFilters(), top_k=5)
        assert hits, "vector search returned nothing"
        assert hits[0].chunk_id == stored[0].id
        assert hits[0].score == pytest.approx(1.0, abs=1e-5), (
            "identical unit vector must score 1.0 under cosine"
        )
        assert all(hit.metric == Metric.COSINE for hit in hits)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True), "hits must be rank-ordered"

        # --- FTS search: BM25 + porter stemming ---
        fts = await store.search_fts("acme", "tokens", SearchFilters(), top_k=5)
        assert fts, "FTS returned nothing for 'tokens'"
        assert all(hit.metric == Metric.BM25 for hit in fts)
        assert "token" in fts[0].content.lower(), "porter should stem 'tokens' -> 'token'"

        # --- hybrid fusion ---
        query = (await provider.encode(["token expiry and rotation policy"]))[0]
        vec_hits = await store.search_vector("acme", query, SearchFilters(), top_k=5)
        hybrid = fuse_rrf([vec_hits, fts])
        assert hybrid, "RRF over non-empty inputs must be non-empty"
        assert all(hit.metric == Metric.RRF for hit in hybrid)
        assert all(0.0 < hit.score <= 2.0 / 61.0 for hit in hybrid), (
            "RRF k=60: each hit scores at most 2 * 1/61"
        )
        weighted = fuse_weighted([vec_hits, fts])
        assert weighted and all(hit.metric == Metric.WEIGHTED for hit in weighted)

        # typed results: no dict[str, Any] leaks (all fields are typed records)
        assert isinstance(hybrid[0], ScoredDocument)
        assert isinstance(hits[0], ScoredDocument)

        # honest errors on unknown collections
        with pytest.raises(StoreError):
            await store.search_vector("nope", query, SearchFilters(), top_k=5)
        with pytest.raises(StoreError):
            await store.stats("nope")


async def test_cosine_score_semantics() -> None:
    """score = 1.0 - distance only under cosine; hand-computed pair check."""
    async with await SqliteStore.open(":memory:") as store:
        await store.create_collection("geo", 4)
        await store.upsert_source(
            "geo",
            SourceMetadata(
                id="s1",
                collection_id="geo",
                path="a.txt",
                content_hash="h",
                size_bytes=1,
            ),
        )
        c1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c2 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        c3 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        # the store enforces unit-norm vectors at the boundary (protocol
        # guard); normalize the raw pair vectors before storing
        c2 = c2 / np.linalg.norm(c2)
        chunks = [
            StoredChunk(
                id="c1",
                collection_id="geo",
                source_id="s1",
                content="alpha beta",
                chunk_type="paragraph",
                start_line=1,
                end_line=1,
            ),
            StoredChunk(
                id="c2",
                collection_id="geo",
                source_id="s1",
                content="alpha gamma",
                chunk_type="paragraph",
                start_line=2,
                end_line=2,
            ),
            StoredChunk(
                id="c3",
                collection_id="geo",
                source_id="s1",
                content="gamma delta",
                chunk_type="heading",
                start_line=3,
                end_line=3,
                parent="intro",
            ),
        ]
        await store.add("geo", chunks, [c1, c2, c3])

        hits = await store.search_vector("geo", c1, SearchFilters(), top_k=3)
        by_id = {hit.chunk_id: hit.score for hit in hits}
        # hand-computed cosine of unit vectors == dot product
        assert by_id["c1"] == pytest.approx(1.0, abs=1e-5)
        assert by_id["c2"] == pytest.approx(float(np.dot(c1, c2)), abs=1e-5)
        assert by_id["c3"] == pytest.approx(float(np.dot(c1, c3)), abs=1e-5)
        assert list(by_id) == ["c1", "c2", "c3"], "ranked by descending cosine"


def test_resolve_dimension_mrl_matrix() -> None:
    """Exact CONCEPT §5.2 logic (ported from the spike's 22-test matrix)."""
    mrl = ModelSpec(
        id="m",
        native_dimension=1024,
        mrl_range=range(32, 1025),
        context_length=1000,
        instructions={},
    )
    assert resolve_dimension(mrl, None) == 1024
    assert resolve_dimension(mrl, 512) == 512
    with pytest.raises(ValueError):
        resolve_dimension(mrl, 16)  # below MRL range
    with pytest.raises(ValueError):
        resolve_dimension(mrl, 1025)  # above MRL range

    plain = ModelSpec(
        id="p",
        native_dimension=8,
        mrl_range=None,
        context_length=1000,
        instructions={},
    )
    assert resolve_dimension(plain, None) == 8
    assert resolve_dimension(plain, 8) == 8
    with pytest.raises(ValueError):
        resolve_dimension(plain, 4)  # non-MRL + wrong dim -> explicit error

    vec = np.asarray([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    truncated = truncate_and_renormalize(vec, 2)
    assert truncated.shape == (2,)
    assert_unit_vector(truncated)  # sliced vectors must be re-normalized
    with pytest.raises(ValueError):
        truncate_and_renormalize(vec, 16)  # cannot grow


def test_fusion_pure_functions() -> None:
    def hit(chunk_id: str, score: float, metric: Metric) -> ScoredDocument:
        return ScoredDocument(
            chunk_id=chunk_id,
            collection_id="acme",
            source_id="s1",
            source_path="a.txt",
            content="x",
            score=score,
            metric=metric,
        )

    vec = [hit("a", 0.9, Metric.COSINE), hit("b", 0.8, Metric.COSINE)]
    fts = [hit("b", -1.0, Metric.BM25), hit("c", -2.0, Metric.BM25)]

    rrf = fuse_rrf([vec, fts])
    assert [h.chunk_id for h in rrf] == ["b", "a", "c"], "b ranks 1+1, a ranks 2"
    assert all(h.metric == Metric.RRF for h in rrf)

    weighted = fuse_weighted([vec, fts])
    assert weighted[0].chunk_id in {"a", "b"}
    assert all(h.metric == Metric.WEIGHTED for h in weighted)

    assert fuse_rrf([]) == []
    assert fuse_weighted([]) == []
    with pytest.raises(ValueError):
        fuse_rrf([vec], k=0)
    with pytest.raises(ValueError):
        fuse_weighted([vec, fts], weights=[1.0])  # wrong weight count
