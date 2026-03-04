# tests/test_client.py
"""Tests for the EmbeddyClient — async HTTP client for the embeddy server.

TDD: Tests are written before the implementation.
Covers: all client methods mirroring the server API, error handling,
connection errors, timeout handling, context manager usage.

Strategy: Build a real FastAPI app with mocked dependencies, then point the
EmbeddyClient at it via httpx.ASGITransport.  This validates that the client
sends correct requests and correctly parses the server's responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from embeddy.client import EmbeddyClient
from embeddy.exceptions import EmbeddyError
from embeddy.models import (
    Collection,
    CollectionStats,
    ContentType,
    Embedding,
    IngestStats,
    SearchMode,
    SearchResult,
    SearchResults,
)
from embeddy.models import IngestError as IngestErrorModel
from embeddy.server import create_app


# ---------------------------------------------------------------------------
# Helpers — mock factories (same pattern as test_server.py)
# ---------------------------------------------------------------------------


def _make_mock_embedder(dimension: int = 128, model_name: str = "test-model"):
    """Create a mock Embedder."""
    embedder = AsyncMock()
    embedder.dimension = dimension
    embedder.model_name = model_name

    async def _encode(inputs, instruction=None):
        return [Embedding(vector=[0.1] * dimension, model_name=model_name) for _ in inputs]

    embedder.encode = AsyncMock(side_effect=_encode)

    async def _encode_query(text):
        return Embedding(vector=[0.2] * dimension, model_name=model_name)

    embedder.encode_query = AsyncMock(side_effect=_encode_query)
    return embedder


def _make_mock_store():
    """Create a mock VectorStore."""
    store = AsyncMock()
    store.list_collections = AsyncMock(return_value=[])
    store.get_collection = AsyncMock(return_value=None)
    store.create_collection = AsyncMock()
    store.delete_collection = AsyncMock()
    store.get = AsyncMock(return_value=None)
    store.delete = AsyncMock()
    store.count = AsyncMock(return_value=0)
    store.list_sources = AsyncMock(return_value=[])
    store.stats = AsyncMock(
        return_value=CollectionStats(
            name="default",
            chunk_count=0,
            source_count=0,
            dimension=128,
            model_name="test-model",
        )
    )
    return store


def _make_mock_pipeline():
    """Create a mock Pipeline."""
    pipeline = AsyncMock()
    pipeline.ingest_text = AsyncMock(
        return_value=IngestStats(files_processed=1, chunks_created=3, chunks_embedded=3, chunks_stored=3)
    )
    pipeline.ingest_file = AsyncMock(
        return_value=IngestStats(files_processed=1, chunks_created=5, chunks_embedded=5, chunks_stored=5)
    )
    pipeline.ingest_directory = AsyncMock(
        return_value=IngestStats(
            files_processed=4, chunks_created=20, chunks_embedded=20, chunks_stored=18, chunks_skipped=2
        )
    )
    pipeline.reindex_file = AsyncMock(
        return_value=IngestStats(files_processed=1, chunks_created=5, chunks_embedded=5, chunks_stored=5)
    )
    pipeline.delete_source = AsyncMock(return_value=3)
    return pipeline


def _make_mock_search_service():
    """Create a mock SearchService."""
    svc = AsyncMock()

    _results = SearchResults(
        results=[
            SearchResult(chunk_id="c1", content="hello world", score=0.95, source_path="/a.py", content_type="python"),
            SearchResult(
                chunk_id="c2", content="goodbye world", score=0.80, source_path="/b.md", content_type="markdown"
            ),
        ],
        query="hello",
        collection="default",
        mode=SearchMode.HYBRID,
        total_results=2,
        elapsed_ms=12.5,
    )
    svc.search = AsyncMock(return_value=_results)
    svc.find_similar = AsyncMock(return_value=_results)
    return svc


def _build_app():
    """Build a FastAPI app with all mocked deps."""
    embedder = _make_mock_embedder()
    store = _make_mock_store()
    pipeline = _make_mock_pipeline()
    search_service = _make_mock_search_service()
    app = create_app(embedder=embedder, store=store, pipeline=pipeline, search_service=search_service)
    return app, embedder, store, pipeline, search_service


@pytest.fixture
def app_and_mocks():
    """Create a test app with mocked dependencies."""
    return _build_app()


@pytest.fixture
async def client(app_and_mocks):
    """Create an EmbeddyClient pointing at the test app via ASGI transport."""
    app, *_ = app_and_mocks
    transport = ASGITransport(app=app)
    async with EmbeddyClient(base_url="http://test", transport=transport) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_health(self, client: EmbeddyClient) -> None:
        result = await client.health()
        assert result["status"] == "ok"

    async def test_info(self, client: EmbeddyClient) -> None:
        result = await client.info()
        assert result["version"] is not None
        assert result["model_name"] == "test-model"
        assert result["dimension"] == 128


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


class TestEmbed:
    async def test_embed_single(self, client: EmbeddyClient) -> None:
        result = await client.embed(["hello world"])
        assert "embeddings" in result
        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) == 128
        assert result["dimension"] == 128
        assert result["model"] == "test-model"
        assert "elapsed_ms" in result

    async def test_embed_batch(self, client: EmbeddyClient) -> None:
        result = await client.embed(["hello", "world", "foo"])
        assert len(result["embeddings"]) == 3

    async def test_embed_with_instruction(self, client: EmbeddyClient) -> None:
        result = await client.embed(["hello"], instruction="Classify this text")
        assert len(result["embeddings"]) == 1

    async def test_embed_empty_input_returns_error(self, client: EmbeddyClient) -> None:
        with pytest.raises(EmbeddyError):
            await client.embed([])

    async def test_embed_query(self, client: EmbeddyClient) -> None:
        result = await client.embed_query("search query")
        assert "embedding" in result
        assert len(result["embedding"]) == 128
        assert result["dimension"] == 128
        assert result["model"] == "test-model"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_default(self, client: EmbeddyClient) -> None:
        result = await client.search("hello", collection="default")
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["chunk_id"] == "c1"
        assert result["results"][0]["score"] == 0.95
        assert result["query"] == "hello"
        assert result["collection"] == "default"
        assert result["total_results"] == 2

    async def test_search_with_options(self, client: EmbeddyClient) -> None:
        result = await client.search(
            "hello",
            collection="default",
            top_k=5,
            mode="vector",
            min_score=0.5,
        )
        assert "results" in result

    async def test_search_hybrid_with_fusion(self, client: EmbeddyClient) -> None:
        result = await client.search(
            "hello",
            collection="default",
            mode="hybrid",
            hybrid_alpha=0.8,
            fusion="weighted",
        )
        assert "results" in result

    async def test_find_similar(self, client: EmbeddyClient) -> None:
        result = await client.find_similar("c1", collection="default", top_k=5)
        assert "results" in result
        assert len(result["results"]) == 2


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    async def test_ingest_text(self, client: EmbeddyClient) -> None:
        result = await client.ingest_text("some text", collection="default")
        assert result["files_processed"] == 1
        assert result["chunks_created"] == 3
        assert result["chunks_embedded"] == 3
        assert result["chunks_stored"] == 3

    async def test_ingest_text_with_source_and_type(self, client: EmbeddyClient) -> None:
        result = await client.ingest_text("some code", collection="code", source="test.py", content_type="python")
        assert result["files_processed"] == 1

    async def test_ingest_file(self, client: EmbeddyClient) -> None:
        result = await client.ingest_file("/path/to/file.py", collection="default")
        assert result["files_processed"] == 1
        assert result["chunks_created"] == 5

    async def test_ingest_directory(self, client: EmbeddyClient) -> None:
        result = await client.ingest_directory("/path/to/dir", collection="default")
        assert result["files_processed"] == 4
        assert result["chunks_created"] == 20
        assert result["chunks_skipped"] == 2

    async def test_ingest_directory_with_options(self, client: EmbeddyClient) -> None:
        result = await client.ingest_directory(
            "/path/to/dir",
            collection="default",
            include=["*.py", "*.md"],
            exclude=["__pycache__"],
            recursive=False,
        )
        assert result["files_processed"] == 4

    async def test_reindex(self, client: EmbeddyClient) -> None:
        result = await client.reindex("/path/to/file.py", collection="default")
        assert result["files_processed"] == 1
        assert result["chunks_created"] == 5

    async def test_delete_source(self, client: EmbeddyClient) -> None:
        result = await client.delete_source("/path/to/file.py", collection="default")
        assert result["deleted_count"] == 3


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class TestCollections:
    async def test_list_collections_empty(self, client: EmbeddyClient) -> None:
        result = await client.list_collections()
        assert result["collections"] == []

    async def test_list_collections_with_items(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.list_collections = AsyncMock(
            return_value=[
                Collection(id="id1", name="col1", dimension=128, model_name="test-model"),
                Collection(id="id2", name="col2", dimension=128, model_name="test-model"),
            ]
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.list_collections()
            assert len(result["collections"]) == 2
            assert result["collections"][0]["name"] == "col1"

    async def test_create_collection(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.create_collection = AsyncMock(
            return_value=Collection(id="new-id", name="my-col", dimension=128, model_name="test-model")
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.create_collection("my-col")
            assert result["name"] == "my-col"
            assert result["dimension"] == 128

    async def test_create_collection_with_metadata(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.create_collection = AsyncMock(
            return_value=Collection(
                id="new-id", name="my-col", dimension=128, model_name="test-model", metadata={"key": "val"}
            )
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.create_collection("my-col", metadata={"key": "val"})
            assert result["name"] == "my-col"

    async def test_get_collection_found(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.get_collection = AsyncMock(
            return_value=Collection(id="id1", name="default", dimension=128, model_name="test-model")
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.get_collection("default")
            assert result["name"] == "default"

    async def test_get_collection_not_found(self, client: EmbeddyClient) -> None:
        with pytest.raises(EmbeddyError, match="not_found"):
            await client.get_collection("nonexistent")

    async def test_delete_collection_found(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.get_collection = AsyncMock(
            return_value=Collection(id="id1", name="default", dimension=128, model_name="test-model")
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            # Should not raise
            await c.delete_collection("default")

    async def test_delete_collection_not_found(self, client: EmbeddyClient) -> None:
        with pytest.raises(EmbeddyError, match="not_found"):
            await client.delete_collection("nonexistent")

    async def test_collection_sources(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.get_collection = AsyncMock(
            return_value=Collection(id="id1", name="default", dimension=128, model_name="test-model")
        )
        store.list_sources = AsyncMock(return_value=["/a.py", "/b.md"])
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.collection_sources("default")
            assert result["sources"] == ["/a.py", "/b.md"]

    async def test_collection_sources_not_found(self, client: EmbeddyClient) -> None:
        with pytest.raises(EmbeddyError, match="not_found"):
            await client.collection_sources("nonexistent")

    async def test_collection_stats(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.get_collection = AsyncMock(
            return_value=Collection(id="id1", name="default", dimension=128, model_name="test-model")
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.collection_stats("default")
            assert result["name"] == "default"
            assert result["chunk_count"] == 0

    async def test_collection_stats_not_found(self, client: EmbeddyClient) -> None:
        with pytest.raises(EmbeddyError, match="not_found"):
            await client.collection_stats("nonexistent")


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


class TestChunks:
    async def test_get_chunk_found(self, app_and_mocks) -> None:
        app, embedder, store, pipeline, search_service = app_and_mocks
        store.get = AsyncMock(
            return_value={
                "chunk_id": "c1",
                "content": "hello world",
                "content_type": "python",
                "chunk_type": "function",
                "source_path": "/a.py",
                "start_line": 1,
                "end_line": 10,
                "name": "my_func",
                "parent": None,
                "metadata": {},
                "content_hash": "abc123",
            }
        )
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.get_chunk("c1", collection="default")
            assert result["chunk_id"] == "c1"
            assert result["content"] == "hello world"
            assert result["name"] == "my_func"

    async def test_get_chunk_not_found(self, client: EmbeddyClient) -> None:
        with pytest.raises(EmbeddyError, match="not_found"):
            await client.get_chunk("nonexistent", collection="default")

    async def test_delete_chunk(self, client: EmbeddyClient) -> None:
        # delete_chunk doesn't check existence, just calls store.delete
        await client.delete_chunk("c1", collection="default")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_server_error_raises_embeddy_error(self, app_and_mocks) -> None:
        """Test that non-2xx responses raise EmbeddyError."""
        app, embedder, store, pipeline, search_service = app_and_mocks
        # Make search raise an exception from the server side
        from embeddy.exceptions import SearchError

        search_service.search = AsyncMock(side_effect=SearchError("search failed"))
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            with pytest.raises(EmbeddyError, match="search failed"):
                await c.search("hello", collection="default")

    async def test_validation_error_raises_embeddy_error(self, client: EmbeddyClient) -> None:
        """Test that 400 responses raise EmbeddyError."""
        with pytest.raises(EmbeddyError):
            await client.embed([])


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    async def test_async_context_manager(self) -> None:
        """Test that EmbeddyClient works as an async context manager."""
        app, *_ = _build_app()
        transport = ASGITransport(app=app)
        async with EmbeddyClient(base_url="http://test", transport=transport) as c:
            result = await c.health()
            assert result["status"] == "ok"

    async def test_explicit_close(self) -> None:
        """Test that the client can be closed explicitly."""
        app, *_ = _build_app()
        transport = ASGITransport(app=app)
        c = EmbeddyClient(base_url="http://test", transport=transport)
        result = await c.health()
        assert result["status"] == "ok"
        await c.close()


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_default_base_url(self) -> None:
        c = EmbeddyClient()
        assert c.base_url == "http://localhost:8585"

    def test_custom_base_url(self) -> None:
        c = EmbeddyClient(base_url="http://myhost:9999")
        assert c.base_url == "http://myhost:9999"

    def test_custom_timeout(self) -> None:
        c = EmbeddyClient(timeout=60.0)
        assert c.timeout == 60.0
