# tests/test_store.py
"""Tests for the storage layer (Phase 3).

Covers: VectorStore initialization, collection CRUD, chunk CRUD,
KNN search (sqlite-vec), FTS search (BM25), metadata filtering,
stats, and transactional integrity.

All tests use a real in-memory SQLite database — no mocking needed
since sqlite-vec and FTS5 are available.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
import pytest

from embeddy.config import StoreConfig
from embeddy.exceptions import StoreError
from embeddy.models import (
    Chunk,
    Collection,
    CollectionStats,
    ContentType,
    DistanceMetric,
    Embedding,
    SearchFilters,
    SourceMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(dim: int = 128, seed: int = 42, normalized: bool = True) -> Embedding:
    """Create a deterministic fake Embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    if normalized:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    return Embedding(
        vector=vec.tolist(),
        model_name="test-model",
        normalized=normalized,
    )


def _make_chunk(
    content: str = "def hello(): pass",
    content_type: ContentType = ContentType.PYTHON,
    chunk_type: str = "function",
    source_path: str | None = "/src/test.py",
    name: str | None = "hello",
    content_hash: str | None = None,
    metadata: dict | None = None,
) -> Chunk:
    """Create a Chunk with a unique ID."""
    return Chunk(
        id=str(uuid.uuid4()),
        content=content,
        content_type=content_type,
        chunk_type=chunk_type,
        source=SourceMetadata(file_path=source_path, content_hash=content_hash),
        name=name,
        metadata=metadata or {},
    )


def _similar_vector(base: Embedding, noise: float = 0.1, seed: int = 99) -> Embedding:
    """Create a vector similar to `base` by adding small noise."""
    rng = np.random.default_rng(seed)
    arr = np.array(base.to_list(), dtype=np.float32)
    arr += rng.standard_normal(arr.shape).astype(np.float32) * noise
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return Embedding(
        vector=arr.tolist(),
        model_name=base.model_name,
        normalized=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_config() -> StoreConfig:
    """In-memory SQLite for tests."""
    return StoreConfig(db_path=":memory:")


@pytest.fixture
async def store(store_config: StoreConfig):
    """Create and initialize a VectorStore, yield it, then close."""
    from embeddy.store.vector_store import VectorStore

    vs = VectorStore(store_config)
    await vs.initialize()
    yield vs
    await vs.close()


@pytest.fixture
async def collection(store) -> Collection:
    """Create a default test collection."""
    return await store.create_collection(
        name="test",
        dimension=128,
        model_name="test-model",
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    """Test VectorStore initialization and database setup."""

    async def test_initialize_creates_tables(self, store) -> None:
        """After init, collections table should exist."""
        # If we can create a collection, the tables exist
        coll = await store.create_collection(name="init-test", dimension=64, model_name="m")
        assert coll.name == "init-test"

    async def test_initialize_idempotent(self, store_config: StoreConfig) -> None:
        """Calling initialize() twice should not raise."""
        from embeddy.store.vector_store import VectorStore

        vs = VectorStore(store_config)
        await vs.initialize()
        await vs.initialize()  # Should not raise
        await vs.close()

    async def test_wal_mode_enabled(self, tmp_path) -> None:
        """WAL mode should be enabled when wal_mode=True (requires file-based DB)."""
        from embeddy.store.vector_store import VectorStore

        config = StoreConfig(db_path=str(tmp_path / "test_wal.db"), wal_mode=True)
        vs = VectorStore(config)
        await vs.initialize()
        mode = await vs._get_journal_mode()
        assert mode.lower() == "wal"
        await vs.close()


# ---------------------------------------------------------------------------
# Collection CRUD
# ---------------------------------------------------------------------------


class TestCollectionCRUD:
    """Test collection create/read/list/delete operations."""

    async def test_create_collection(self, store) -> None:
        coll = await store.create_collection(
            name="my-collection",
            dimension=256,
            model_name="test-model",
            distance_metric=DistanceMetric.COSINE,
        )
        assert coll.name == "my-collection"
        assert coll.dimension == 256
        assert coll.model_name == "test-model"
        assert coll.id is not None

    async def test_create_duplicate_name_raises(self, store) -> None:
        await store.create_collection(name="dup", dimension=128, model_name="m")
        with pytest.raises(StoreError, match="[Aa]lready exists|[Dd]uplicate|UNIQUE"):
            await store.create_collection(name="dup", dimension=128, model_name="m")

    async def test_get_collection(self, store) -> None:
        created = await store.create_collection(name="get-test", dimension=128, model_name="m")
        fetched = await store.get_collection("get-test")
        assert fetched is not None
        assert fetched.name == "get-test"
        assert fetched.id == created.id

    async def test_get_nonexistent_collection_returns_none(self, store) -> None:
        result = await store.get_collection("nonexistent")
        assert result is None

    async def test_list_collections_empty(self, store) -> None:
        colls = await store.list_collections()
        assert colls == []

    async def test_list_collections(self, store) -> None:
        await store.create_collection(name="a", dimension=128, model_name="m")
        await store.create_collection(name="b", dimension=256, model_name="m")
        colls = await store.list_collections()
        names = {c.name for c in colls}
        assert names == {"a", "b"}

    async def test_delete_collection(self, store) -> None:
        await store.create_collection(name="del-test", dimension=128, model_name="m")
        await store.delete_collection("del-test")
        result = await store.get_collection("del-test")
        assert result is None

    async def test_delete_nonexistent_collection_raises(self, store) -> None:
        with pytest.raises(StoreError, match="[Nn]ot found|[Dd]oes not exist"):
            await store.delete_collection("ghost")

    async def test_delete_collection_removes_chunks(self, store) -> None:
        """Deleting a collection should also delete its chunks, vectors, and FTS data."""
        coll = await store.create_collection(name="del-chunks", dimension=128, model_name="m")
        chunk = _make_chunk()
        emb = _make_embedding(dim=128)
        await store.add("del-chunks", [chunk], [emb])

        # Verify chunk exists
        count = await store.count("del-chunks")
        assert count == 1

        # Delete collection
        await store.delete_collection("del-chunks")

        # Collection should be gone
        assert await store.get_collection("del-chunks") is None


# ---------------------------------------------------------------------------
# Chunk CRUD
# ---------------------------------------------------------------------------


class TestChunkCRUD:
    """Test chunk add/get/delete operations."""

    async def test_add_single_chunk(self, store, collection: Collection) -> None:
        chunk = _make_chunk()
        emb = _make_embedding(dim=128)
        await store.add("test", [chunk], [emb])

        count = await store.count("test")
        assert count == 1

    async def test_add_multiple_chunks(self, store, collection: Collection) -> None:
        chunks = [_make_chunk(content=f"def func_{i}(): pass", name=f"func_{i}") for i in range(5)]
        embs = [_make_embedding(dim=128, seed=i) for i in range(5)]
        await store.add("test", chunks, embs)

        count = await store.count("test")
        assert count == 5

    async def test_add_mismatched_lengths_raises(self, store, collection: Collection) -> None:
        chunks = [_make_chunk()]
        embs = [_make_embedding(dim=128), _make_embedding(dim=128, seed=1)]
        with pytest.raises((StoreError, ValueError)):
            await store.add("test", chunks, embs)

    async def test_add_to_nonexistent_collection_raises(self, store) -> None:
        with pytest.raises(StoreError):
            await store.add("nonexistent", [_make_chunk()], [_make_embedding(dim=128)])

    async def test_get_chunk(self, store, collection: Collection) -> None:
        chunk = _make_chunk(content="class Foo: pass", name="Foo", chunk_type="class")
        emb = _make_embedding(dim=128)
        await store.add("test", [chunk], [emb])

        result = await store.get("test", chunk.id)
        assert result is not None
        assert result["content"] == "class Foo: pass"
        assert result["name"] == "Foo"
        assert result["chunk_type"] == "class"

    async def test_get_nonexistent_chunk_returns_none(self, store, collection: Collection) -> None:
        result = await store.get("test", "nonexistent-id")
        assert result is None

    async def test_delete_chunks(self, store, collection: Collection) -> None:
        chunks = [_make_chunk(content=f"def f{i}(): pass", name=f"f{i}") for i in range(3)]
        embs = [_make_embedding(dim=128, seed=i) for i in range(3)]
        await store.add("test", chunks, embs)

        # Delete first two
        await store.delete("test", [chunks[0].id, chunks[1].id])
        count = await store.count("test")
        assert count == 1

        # Remaining chunk should still be retrievable
        result = await store.get("test", chunks[2].id)
        assert result is not None

    async def test_delete_by_source(self, store, collection: Collection) -> None:
        c1 = _make_chunk(content="def a(): pass", source_path="/src/a.py", name="a")
        c2 = _make_chunk(content="def b(): pass", source_path="/src/a.py", name="b")
        c3 = _make_chunk(content="def c(): pass", source_path="/src/b.py", name="c")
        embs = [_make_embedding(dim=128, seed=i) for i in range(3)]
        await store.add("test", [c1, c2, c3], embs)

        deleted = await store.delete_by_source("test", "/src/a.py")
        assert deleted == 2
        assert await store.count("test") == 1


# ---------------------------------------------------------------------------
# KNN Search
# ---------------------------------------------------------------------------


class TestKNNSearch:
    """Test vector similarity search via sqlite-vec."""

    async def test_search_knn_basic(self, store, collection: Collection) -> None:
        """Insert chunks, search with a similar vector, get ranked results."""
        # Create base embedding and chunks
        base_emb = _make_embedding(dim=128, seed=10)
        similar_emb = _similar_vector(base_emb, noise=0.05, seed=20)
        distant_emb = _make_embedding(dim=128, seed=999)  # Very different

        c_similar = _make_chunk(content="similar content", name="similar")
        c_distant = _make_chunk(content="distant content", name="distant")
        c_base = _make_chunk(content="base content", name="base")

        await store.add("test", [c_base, c_similar, c_distant], [base_emb, similar_emb, distant_emb])

        # Search with base vector — should find base first, then similar, then distant
        results = await store.search_knn("test", base_emb.to_list(), top_k=3)
        assert len(results) == 3
        # First result should be the base chunk (exact match)
        assert results[0]["chunk_id"] == c_base.id

    async def test_search_knn_top_k(self, store, collection: Collection) -> None:
        """top_k limits the number of results."""
        chunks = [_make_chunk(content=f"chunk {i}", name=f"c{i}") for i in range(10)]
        embs = [_make_embedding(dim=128, seed=i) for i in range(10)]
        await store.add("test", chunks, embs)

        results = await store.search_knn("test", embs[0].to_list(), top_k=3)
        assert len(results) == 3

    async def test_search_knn_empty_collection(self, store, collection: Collection) -> None:
        results = await store.search_knn("test", _make_embedding(dim=128).to_list(), top_k=5)
        assert results == []

    async def test_search_knn_with_content_type_filter(self, store, collection: Collection) -> None:
        """Filter by content type during KNN search."""
        c_py = _make_chunk(content="python code", content_type=ContentType.PYTHON, name="py")
        c_md = _make_chunk(content="markdown text", content_type=ContentType.MARKDOWN, name="md")
        embs = [_make_embedding(dim=128, seed=i) for i in range(2)]
        await store.add("test", [c_py, c_md], embs)

        filters = SearchFilters(content_types=[ContentType.PYTHON])
        results = await store.search_knn("test", embs[0].to_list(), top_k=10, filters=filters)
        assert len(results) == 1
        assert results[0]["content_type"] == "python"

    async def test_search_knn_with_source_prefix_filter(self, store, collection: Collection) -> None:
        """Filter by source path prefix."""
        c1 = _make_chunk(content="def a(): pass", source_path="/src/core/a.py", name="a")
        c2 = _make_chunk(content="def b(): pass", source_path="/src/utils/b.py", name="b")
        embs = [_make_embedding(dim=128, seed=i) for i in range(2)]
        await store.add("test", [c1, c2], embs)

        filters = SearchFilters(source_path_prefix="/src/core")
        results = await store.search_knn("test", embs[0].to_list(), top_k=10, filters=filters)
        assert len(results) == 1
        assert results[0]["source_path"] == "/src/core/a.py"


# ---------------------------------------------------------------------------
# FTS Search (BM25)
# ---------------------------------------------------------------------------


class TestFTSSearch:
    """Test full-text search via FTS5."""

    async def test_search_fts_basic(self, store, collection: Collection) -> None:
        """Full-text search finds chunks by content keywords."""
        c1 = _make_chunk(content="The quick brown fox jumps over the lazy dog", name="fox")
        c2 = _make_chunk(content="A slow red turtle crawls under the fence", name="turtle")
        embs = [_make_embedding(dim=128, seed=i) for i in range(2)]
        await store.add("test", [c1, c2], embs)

        results = await store.search_fts("test", "quick brown fox", top_k=10)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == c1.id

    async def test_search_fts_by_name(self, store, collection: Collection) -> None:
        """FTS should also search the `name` column."""
        c1 = _make_chunk(content="def calculate_total(): return sum(items)", name="calculate_total")
        c2 = _make_chunk(content="def process_data(): return data", name="process_data")
        embs = [_make_embedding(dim=128, seed=i) for i in range(2)]
        await store.add("test", [c1, c2], embs)

        results = await store.search_fts("test", "calculate_total", top_k=10)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == c1.id

    async def test_search_fts_empty_results(self, store, collection: Collection) -> None:
        c1 = _make_chunk(content="hello world", name="hello")
        embs = [_make_embedding(dim=128)]
        await store.add("test", [c1], embs)

        results = await store.search_fts("test", "xyznonexistent", top_k=10)
        assert results == []

    async def test_search_fts_with_filter(self, store, collection: Collection) -> None:
        """FTS results should respect content_type filters."""
        c_py = _make_chunk(content="python function implementation", content_type=ContentType.PYTHON, name="py_func")
        c_md = _make_chunk(content="markdown function documentation", content_type=ContentType.MARKDOWN, name="md_func")
        embs = [_make_embedding(dim=128, seed=i) for i in range(2)]
        await store.add("test", [c_py, c_md], embs)

        filters = SearchFilters(content_types=[ContentType.PYTHON])
        results = await store.search_fts("test", "function", top_k=10, filters=filters)
        assert len(results) == 1
        assert results[0]["content_type"] == "python"


# ---------------------------------------------------------------------------
# Stats & Info
# ---------------------------------------------------------------------------


class TestStoreStats:
    """Test store info/stats methods."""

    async def test_count_empty(self, store, collection: Collection) -> None:
        count = await store.count("test")
        assert count == 0

    async def test_count_after_add(self, store, collection: Collection) -> None:
        chunks = [_make_chunk(content=f"c{i}", name=f"n{i}") for i in range(7)]
        embs = [_make_embedding(dim=128, seed=i) for i in range(7)]
        await store.add("test", chunks, embs)
        assert await store.count("test") == 7

    async def test_list_sources(self, store, collection: Collection) -> None:
        c1 = _make_chunk(content="def a(): pass", source_path="/src/a.py", name="a")
        c2 = _make_chunk(content="def b(): pass", source_path="/src/a.py", name="b")
        c3 = _make_chunk(content="def c(): pass", source_path="/src/b.py", name="c")
        embs = [_make_embedding(dim=128, seed=i) for i in range(3)]
        await store.add("test", [c1, c2, c3], embs)

        sources = await store.list_sources("test")
        assert set(sources) == {"/src/a.py", "/src/b.py"}

    async def test_stats(self, store, collection: Collection) -> None:
        chunks = [
            _make_chunk(content="def a(): pass", source_path="/src/a.py", name="a"),
            _make_chunk(content="def b(): pass", source_path="/src/b.py", name="b"),
        ]
        embs = [_make_embedding(dim=128, seed=i) for i in range(2)]
        await store.add("test", chunks, embs)

        stats = await store.stats("test")
        assert isinstance(stats, CollectionStats)
        assert stats.name == "test"
        assert stats.chunk_count == 2
        assert stats.source_count == 2
        assert stats.dimension == 128
        assert stats.model_name == "test-model"

    async def test_count_nonexistent_collection_raises(self, store) -> None:
        with pytest.raises(StoreError):
            await store.count("nonexistent")


# ---------------------------------------------------------------------------
# Content Hash Dedup Support
# ---------------------------------------------------------------------------


class TestContentHashSupport:
    """Test that content_hash is stored and queryable for dedup."""

    async def test_content_hash_stored(self, store, collection: Collection) -> None:
        chunk = _make_chunk(content="hashable content", content_hash="abc123", name="hashed")
        emb = _make_embedding(dim=128)
        await store.add("test", [chunk], [emb])

        result = await store.get("test", chunk.id)
        assert result is not None
        assert result["content_hash"] == "abc123"

    async def test_has_content_hash(self, store, collection: Collection) -> None:
        """Store should provide a way to check if a content_hash exists."""
        chunk = _make_chunk(content="unique content", content_hash="hash_xyz", name="unique")
        emb = _make_embedding(dim=128)
        await store.add("test", [chunk], [emb])

        assert await store.has_content_hash("test", "hash_xyz") is True
        assert await store.has_content_hash("test", "nonexistent_hash") is False


# ---------------------------------------------------------------------------
# Transactional Integrity
# ---------------------------------------------------------------------------


class TestTransactionalIntegrity:
    """Test that add/delete operations are atomic across all three stores."""

    async def test_add_populates_all_three_stores(self, store, collection: Collection) -> None:
        """Adding a chunk should insert into chunks table, vec table, and FTS table."""
        chunk = _make_chunk(content="transactional test content", name="txn_test")
        emb = _make_embedding(dim=128)
        await store.add("test", [chunk], [emb])

        # Chunk table
        result = await store.get("test", chunk.id)
        assert result is not None

        # Vec table (KNN should find it)
        knn_results = await store.search_knn("test", emb.to_list(), top_k=1)
        assert len(knn_results) == 1
        assert knn_results[0]["chunk_id"] == chunk.id

        # FTS table (full-text should find it)
        fts_results = await store.search_fts("test", "transactional test", top_k=1)
        assert len(fts_results) == 1
        assert fts_results[0]["chunk_id"] == chunk.id

    async def test_delete_removes_from_all_three_stores(self, store, collection: Collection) -> None:
        """Deleting a chunk should remove from chunks, vec, and FTS."""
        chunk = _make_chunk(content="delete me from all stores", name="del_all")
        emb = _make_embedding(dim=128)
        await store.add("test", [chunk], [emb])

        await store.delete("test", [chunk.id])

        # All three should be empty
        assert await store.get("test", chunk.id) is None
        assert await store.search_knn("test", emb.to_list(), top_k=1) == []
        assert await store.search_fts("test", "delete stores", top_k=1) == []
