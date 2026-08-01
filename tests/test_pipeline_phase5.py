"""Phase-5 integration tests — the real SqliteStore behind the pipeline.

Plan §7 test plan, integration layer (real sqlite-vec/FTS5, real chonkai
Ingestor + chunker routing; the only mock is the deterministic FakeProvider).
Covers:
  * ingest_directory over a real store: stats, source paths, searchability
  * dedup: identical content at two paths = two sources (real store UNIQUE)
  * pipeline-level reindex atomicity: a MID-SWAP failure injected at the
    pipeline's write phase leaves the OLD chunks, vectors, FTS rows and
    metadata fully intact and queryable (the Phase-4 failure pattern
    repeated at the pipeline level — plan §7)
  * incremental sync: new / modified / deleted diff
  * reindex success: atomic swap, stale FTS terms gone
  * reindex_unchanged dedup policy (skip vs reindex)
  * ingest_text -> search_hybrid finds the content
  * auto chunker routing: a .py file gets tree-sitter code chunks
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from chonkai import ParagraphChunker
from chonkai.models import compute_content_hash
from embeddy import (
    IngestPipeline,
    SourcePhase,
    generate_source_id,
)
from embeddy.index.base import Searchable, SearchFilters
from embeddy.index.sqlite import SqliteStore
from embeddy.protocol.types import SourceMetadata, StoredChunk
from embeddy.providers.fake import FakeProvider
from embeddy.search import search_hybrid

pytestmark = pytest.mark.integration

DIM = 8
tok = lambda s: len(s.split())  # noqa: E731 — deterministic word-count counter


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def _pipeline(store: Searchable, **kw) -> IngestPipeline:
    """A pipeline over the real store with a deterministic chunker +
    FakeProvider (the only realistic mock). `instruction` is injected so the
    FakeProvider (non-registry model_name) never hits the registry."""
    kw.setdefault("chunker", ParagraphChunker(token_counter=tok))
    kw.setdefault("token_counter", tok)
    return IngestPipeline(
        store=store,
        provider=FakeProvider(),
        instruction="doc",
        concurrency=3,
        **kw,
    )


async def test_ingest_directory_real_store(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("The quick brown fox jumps over the lazy dog.")
    (tmp_path / "b.txt").write_text("Apples and oranges are both fruit.")
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = _pipeline(store)
        stats = await pipeline.ingest_directory(tmp_path, collection="acme")
        assert stats.files_indexed == 2
        assert stats.files_attempted == 2
        assert stats.errors == ()

        cs = await store.stats("acme")
        assert cs.source_count == 2
        assert cs.chunk_count == 2  # one semchunk paragraph each

        # searchable content + canonical collection-relative paths
        hits = await store.search_fts("acme", "fox", SearchFilters(), top_k=5)
        assert hits and "fox" in hits[0].content
        assert [s.path for s in await store.list_sources("acme")] == ["a.txt", "b.txt"]
        # source ids are the stable hash(collection, path)
        assert (await store.list_sources("acme"))[0].id == generate_source_id("acme", "a.txt")
    finally:
        await store.close()


async def test_dedup_identical_content_two_paths_real(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("identical bytes here")
    (tmp_path / "b.txt").write_text("identical bytes here")
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = _pipeline(store)
        stats = await pipeline.ingest_directory(tmp_path, collection="acme")
        assert stats.files_indexed == 2
        assert (await store.stats("acme")).source_count == 2  # two sources
        # re-ingest: unchanged content -> skip (no-op)
        again = await pipeline.ingest_directory(tmp_path, collection="acme")
        assert again.files_skipped == 2 and again.files_indexed == 0
        assert (await store.stats("acme")).source_count == 2
    finally:
        await store.close()


async def test_reindex_atomic_at_pipeline_level_mid_swap_failure(tmp_path: Path) -> None:
    """The Phase-4 failure-injection pattern repeated at the PIPELINE level:
    a chunk-id PK collision with ANOTHER source's chunk fires MID-SWAP inside
    store.reindex_source (after the old rows are gone, before commit). The
    rollback must leave the old chunks, vectors, FTS rows AND source metadata
    fully intact and queryable, and the failure must be COLLECTED in
    IngestStats — never raised out of the pool."""
    src = tmp_path / "a.txt"
    src.write_text("old alpha beta content")
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = _pipeline(store)
        first = await pipeline.ingest_file(src, collection="acme")
        assert first.files_indexed == 1
        old_hash = compute_content_hash(b"old alpha beta content")
        got = await store.get_source("acme", str(src))
        assert got is not None
        assert got.content_hash == old_hash

        # a second source owns chunk id "foreign:0" — the poison below
        # reuses it, colliding with the GLOBAL chunks PK mid-transaction.
        await store.upsert_source(
            "acme",
            SourceMetadata(
                id="foreign",
                collection_id="acme",
                path="foreign.txt",
                content_hash="h",
                size_bytes=1,
            ),
        )
        await store.add(
            "acme",
            [
                StoredChunk(
                    id="foreign:0",
                    collection_id="acme",
                    source_id="foreign",
                    content="foreign stuff",
                    chunk_type="paragraph",
                    start_line=1,
                    end_line=1,
                )
            ],
            [_unit(3)],
        )

        # now the content changes; reindex through a wrapper that injects
        # the colliding chunk into the payload (the pipeline's write call
        # reaches the REAL store transaction and rolls back atomically).
        src.write_text("new gamma delta content")

        class PoisonStore:
            def __init__(self, inner: SqliteStore) -> None:
                self.inner = inner

            async def reindex_source(self, collection, source, chunks, vectors):
                poison = StoredChunk(
                    id="foreign:0",
                    collection_id=collection,
                    source_id=source.id,
                    content="clobber",
                    chunk_type="paragraph",
                    start_line=1,
                    end_line=1,
                )
                await self.inner.reindex_source(
                    collection, source, [*chunks, poison], [*vectors, _unit(7)]
                )

            def __getattr__(self, name):
                return getattr(self.inner, name)

        poisoned = _pipeline(cast(Searchable, PoisonStore(store)))
        stats = await poisoned.ingest_file(src, collection="acme")
        assert stats.files_indexed == 0
        (err,) = stats.errors
        assert err.phase is SourcePhase.STORE
        assert err.error_type == "IntegrityError"

        # --- the OLD index is fully intact and queryable ---
        # FTS: old term present, new term absent
        old_hits = await store.search_fts("acme", "alpha", SearchFilters(), top_k=10)
        assert any("old alpha beta" in h.content for h in old_hits)
        assert await store.search_fts("acme", "gamma", SearchFilters(), top_k=10) == []
        # vectors: the old chunk still scores 1.0 against its own text
        q = (await FakeProvider().encode(["old alpha beta content"]))[0]
        vec_hits = await store.search_vector("acme", q, SearchFilters(), top_k=10)
        sid = generate_source_id("acme", str(src))
        assert {h.chunk_id for h in vec_hits} == {f"{sid}:0", "foreign:0"}
        assert any(
            h.chunk_id == f"{sid}:0" and h.score == pytest.approx(1.0, abs=1e-5) for h in vec_hits
        )
        # stats + metadata untouched
        assert (await store.stats("acme")).chunk_count == 2
        got = await store.get_source("acme", str(src))
        assert got is not None and got.content_hash == old_hash
    finally:
        await store.close()


async def test_reindex_success_swaps_atomically(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("old gamma term")
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = _pipeline(store)
        await pipeline.ingest_file(src, collection="acme")
        src.write_text("brand new term delta")
        stats = await pipeline.ingest_file(src, collection="acme")
        assert stats.files_indexed == 1 and stats.files_skipped == 0
        # old FTS terms gone, new present (no stale rows)
        assert await store.search_fts("acme", "gamma", SearchFilters(), top_k=10) == []
        hits = await store.search_fts("acme", "brand", SearchFilters(), top_k=10)
        assert any("brand new term delta" in h.content for h in hits)
        got = await store.get_source("acme", str(src))
        assert got is not None
        assert got.content_hash == compute_content_hash(b"brand new term delta")
    finally:
        await store.close()


async def test_reindex_unchanged_policy_real(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("content")
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        skip_pipeline = _pipeline(store)
        s1 = await skip_pipeline.ingest_file(src, collection="acme")
        s2 = await skip_pipeline.ingest_file(src, collection="acme")
        assert s1.files_indexed == 1
        assert s2.files_skipped == 1 and s2.files_indexed == 0

        reindex_pipeline = _pipeline(store, reindex_unchanged=True)
        s3 = await reindex_pipeline.ingest_file(src, collection="acme")
        assert s3.files_indexed == 1 and s3.files_skipped == 0
        assert (await store.stats("acme")).chunk_count == 1
    finally:
        await store.close()


async def test_sync_incremental_real_store(tmp_path: Path) -> None:
    # seed OUTSIDE the synced tree (sync walks tmp_path recursively)
    seed_dir = tmp_path.parent / "seed-dir"
    seed_dir.mkdir(exist_ok=True)
    (seed_dir / "a.txt").write_text("alpha")
    (seed_dir / "old.txt").write_text("gone")
    (seed_dir / "c.txt").write_text("c one")
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = _pipeline(store)
        await pipeline.ingest_directory(seed_dir, collection="acme")

        # live dir: a.txt unchanged, c.txt modified, new.txt added, old.txt gone
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "c.txt").write_text("c TWO")
        (tmp_path / "new.txt").write_text("brand new")
        stats = await pipeline.sync(tmp_path, collection="acme")
        assert stats.files_deleted == 1
        assert stats.files_indexed == 2  # c.txt reindexed + new.txt
        assert stats.files_skipped == 1  # a.txt unchanged
        assert stats.chunks_indexed == 2

        paths = {s.path for s in await store.list_sources("acme")}
        assert paths == {"a.txt", "c.txt", "new.txt"}
        assert (await store.stats("acme")).source_count == 3
        # modified content is searchable
        hits = await store.search_fts("acme", "TWO", SearchFilters(), top_k=5)
        assert hits
        # the deleted source is fully gone (no stale FTS rows)
        assert await store.search_fts("acme", "gone", SearchFilters(), top_k=5) == []
    finally:
        await store.close()


async def test_ingest_text_real_store_searchable() -> None:
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = _pipeline(store)
        stats = await pipeline.ingest_text(
            "token expiry and rotation policy",
            path="guide.md",
            content_type="markdown",
            collection="acme",
        )
        assert stats.files_indexed == 1 and stats.chunks_indexed == 1

        q = (await FakeProvider().encode(["token expiry policy"]))[0]
        res = await search_hybrid(
            store,
            collection="acme",
            query_text="token policy",
            query_vector=q,
            top_k=5,
        )
        assert res.results
        assert res.results[0].chunk_id.startswith(generate_source_id("acme", "guide.md"))
        assert res.total_results >= 1
    finally:
        await store.close()


async def test_auto_chunker_routing_python(tmp_path: Path) -> None:
    """ingest_directory with NO fixed chunker routes a .py file to the
    tree-sitter chunker (get_chunker auto): the function/class/method split
    (3 chunks) is impossible for the paragraph chunker — proving content-
    type routing ran."""
    (tmp_path / "mod.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
        "\nclass Foo:\n    def bar(self):\n        return 42\n"
    )
    store = await SqliteStore.open(":memory:")
    try:
        await store.create_collection("acme", DIM)
        pipeline = IngestPipeline(
            store=store,
            provider=FakeProvider(),
            instruction="doc",
            concurrency=2,
            token_counter=tok,
        )  # chunker=None -> get_chunker(auto)
        stats = await pipeline.ingest_directory(tmp_path, collection="acme")
        assert stats.files_indexed == 1
        assert (await store.stats("acme")).chunk_count == 3  # function+class+method
        hits = await store.search_fts("acme", "bar", SearchFilters(), top_k=10)
        assert hits and "def bar" in hits[0].content
    finally:
        await store.close()
