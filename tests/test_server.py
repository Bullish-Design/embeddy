# tests/test_server.py
"""Tests for the FastAPI server layer.

TDD: Tests are written before the implementation.
Covers: health/info, embed, search, ingest, collections, chunks, error handling.

All tests mock the Embedder, VectorStore, Pipeline, and SearchService
to avoid heavy dependencies (no real model or database needed).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from embeddy.models import (
    Collection,
    CollectionStats,
    ContentType,
    Embedding,
    FusionStrategy,
    IngestStats,
    SearchMode,
    SearchResult,
    SearchResults,
)


# ---------------------------------------------------------------------------
# Helpers — mock factories
# ---------------------------------------------------------------------------


def _make_mock_embedder(dimension: int = 128, model_name: str = "test-model"):
    """Create a mock Embedder."""
    embedder = AsyncMock()
    embedder.dimension = dimension
    embedder.model_name = model_name

    async def _encode(inputs, instruction=None):
        return [Embedding(vector=[0.1] * dimension, model_name=model_name, normalized=True) for _ in inputs]

    embedder.encode = AsyncMock(side_effect=_encode)

    async def _encode_query(text):
        return Embedding(vector=[0.1] * dimension, model_name=model_name, normalized=True)

    embedder.encode_query = AsyncMock(side_effect=_encode_query)

    async def _encode_document(text):
        return Embedding(vector=[0.1] * dimension, model_name=model_name, normalized=True)

    embedder.encode_document = AsyncMock(side_effect=_encode_document)
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
        return_value=IngestStats(
            files_processed=1,
            chunks_created=3,
            chunks_embedded=3,
            chunks_stored=3,
            chunks_skipped=0,
            elapsed_seconds=0.5,
        )
    )
    pipeline.ingest_file = AsyncMock(
        return_value=IngestStats(
            files_processed=1,
            chunks_created=5,
            chunks_embedded=5,
            chunks_stored=5,
            chunks_skipped=0,
            elapsed_seconds=1.0,
        )
    )
    pipeline.ingest_directory = AsyncMock(
        return_value=IngestStats(
            files_processed=10,
            chunks_created=50,
            chunks_embedded=50,
            chunks_stored=48,
            chunks_skipped=2,
            elapsed_seconds=5.0,
        )
    )
    pipeline.reindex_file = AsyncMock(
        return_value=IngestStats(
            files_processed=1,
            chunks_created=5,
            chunks_embedded=5,
            chunks_stored=5,
            chunks_skipped=0,
            elapsed_seconds=1.2,
        )
    )
    pipeline.delete_source = AsyncMock(return_value=5)
    return pipeline


def _make_mock_search_service():
    """Create a mock SearchService."""
    svc = AsyncMock()

    async def _search(query, collection, **kwargs):
        return SearchResults(
            results=[
                SearchResult(chunk_id="chunk-1", content="Result 1", score=0.95),
                SearchResult(chunk_id="chunk-2", content="Result 2", score=0.85),
            ],
            query=query,
            collection=collection,
            mode=kwargs.get("mode", SearchMode.HYBRID),
            total_results=2,
            elapsed_ms=15.0,
        )

    svc.search = AsyncMock(side_effect=_search)

    async def _find_similar(chunk_id, collection, **kwargs):
        return SearchResults(
            results=[
                SearchResult(chunk_id="chunk-3", content="Similar 1", score=0.88),
            ],
            query=f"similar:{chunk_id}",
            collection=collection,
            mode=SearchMode.VECTOR,
            total_results=1,
            elapsed_ms=10.0,
        )

    svc.find_similar = AsyncMock(side_effect=_find_similar)
    return svc


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_deps():
    """Create all mock dependencies."""
    return {
        "embedder": _make_mock_embedder(),
        "store": _make_mock_store(),
        "pipeline": _make_mock_pipeline(),
        "search_service": _make_mock_search_service(),
    }


@pytest.fixture
async def client(mock_deps):
    """Create an async test client with mocked dependencies."""
    from embeddy.server.app import create_app

    app = create_app(
        embedder=mock_deps["embedder"],
        store=mock_deps["store"],
        pipeline=mock_deps["pipeline"],
        search_service=mock_deps["search_service"],
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for the /health endpoint."""

    async def test_health_returns_200(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_health_has_status_field(self, client):
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "ok"


class TestInfo:
    """Tests for the /info endpoint."""

    async def test_info_returns_200(self, client):
        resp = await client.get("/api/v1/info")
        assert resp.status_code == 200

    async def test_info_includes_version(self, client):
        resp = await client.get("/api/v1/info")
        data = resp.json()
        assert "version" in data

    async def test_info_includes_model_name(self, client, mock_deps):
        resp = await client.get("/api/v1/info")
        data = resp.json()
        assert data["model_name"] == "test-model"

    async def test_info_includes_dimension(self, client, mock_deps):
        resp = await client.get("/api/v1/info")
        data = resp.json()
        assert data["dimension"] == 128


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


class TestEmbed:
    """Tests for /embed endpoints."""

    async def test_embed_single_text(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/embed",
            json={"inputs": [{"text": "hello world"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert data["dimension"] == 128

    async def test_embed_batch(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/embed",
            json={"inputs": [{"text": "hello"}, {"text": "world"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embeddings"]) == 2

    async def test_embed_query(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/embed/query",
            json={"input": {"text": "search query"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding" in data
        assert data["dimension"] == 128

    async def test_embed_empty_inputs_returns_400(self, client):
        resp = await client.post(
            "/api/v1/embed",
            json={"inputs": []},
        )
        assert resp.status_code == 400

    async def test_embed_includes_model_name(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/embed",
            json={"inputs": [{"text": "test"}]},
        )
        data = resp.json()
        assert data["model"] == "test-model"

    async def test_embed_includes_elapsed_ms(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/embed",
            json={"inputs": [{"text": "test"}]},
        )
        data = resp.json()
        assert "elapsed_ms" in data
        assert data["elapsed_ms"] >= 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for /search endpoints."""

    async def test_search_returns_results(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test query", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2

    async def test_search_returns_scores(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test query", "collection": "default"},
        )
        data = resp.json()
        for result in data["results"]:
            assert "score" in result
            assert "chunk_id" in result
            assert "content" in result

    async def test_search_supports_mode(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test", "collection": "default", "mode": "vector"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "vector"

    async def test_search_supports_top_k(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test", "collection": "default", "top_k": 5},
        )
        assert resp.status_code == 200

    async def test_search_similar(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search/similar",
            json={"chunk_id": "chunk-1", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    async def test_search_includes_elapsed_ms(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test", "collection": "default"},
        )
        data = resp.json()
        assert "elapsed_ms" in data
        assert data["elapsed_ms"] >= 0

    async def test_search_includes_collection(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test", "collection": "my_coll"},
        )
        data = resp.json()
        assert data["collection"] == "my_coll"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    """Tests for /ingest endpoints."""

    async def test_ingest_text(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/ingest/text",
            json={"text": "Hello world", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_created"] == 3
        assert data["files_processed"] == 1

    async def test_ingest_file(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/ingest/file",
            json={"path": "/tmp/test.py", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_created"] == 5

    async def test_ingest_directory(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/ingest/directory",
            json={"path": "/tmp/project", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_processed"] == 10

    async def test_ingest_reindex(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/ingest/reindex",
            json={"path": "/tmp/test.py", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_created"] == 5

    async def test_ingest_delete_source(self, client, mock_deps):
        resp = await client.request(
            "DELETE",
            "/api/v1/ingest/source",
            json={"source_path": "/tmp/test.py", "collection": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_count"] == 5

    async def test_ingest_text_includes_elapsed(self, client, mock_deps):
        resp = await client.post(
            "/api/v1/ingest/text",
            json={"text": "test", "collection": "default"},
        )
        data = resp.json()
        assert "elapsed_seconds" in data


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class TestCollections:
    """Tests for /collections endpoints."""

    async def test_list_collections_empty(self, client, mock_deps):
        resp = await client.get("/api/v1/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collections"] == []

    async def test_list_collections_with_data(self, client, mock_deps):
        mock_deps["store"].list_collections = AsyncMock(
            return_value=[
                Collection(
                    id="c1",
                    name="test-coll",
                    dimension=128,
                    model_name="test-model",
                ),
            ]
        )
        resp = await client.get("/api/v1/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["collections"]) == 1
        assert data["collections"][0]["name"] == "test-coll"

    async def test_create_collection(self, client, mock_deps):
        mock_deps["store"].create_collection = AsyncMock(
            return_value=Collection(
                id="new-id",
                name="my-coll",
                dimension=128,
                model_name="test-model",
            )
        )
        resp = await client.post(
            "/api/v1/collections",
            json={"name": "my-coll"},
        )
        assert resp.status_code == 201

    async def test_get_collection(self, client, mock_deps):
        mock_deps["store"].get_collection = AsyncMock(
            return_value=Collection(
                id="c1",
                name="test-coll",
                dimension=128,
                model_name="test-model",
            )
        )
        resp = await client.get("/api/v1/collections/test-coll")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-coll"

    async def test_get_collection_not_found(self, client, mock_deps):
        mock_deps["store"].get_collection = AsyncMock(return_value=None)
        resp = await client.get("/api/v1/collections/nonexistent")
        assert resp.status_code == 404

    async def test_delete_collection(self, client, mock_deps):
        # Ensure it exists first
        mock_deps["store"].get_collection = AsyncMock(
            return_value=Collection(
                id="c1",
                name="test-coll",
                dimension=128,
                model_name="test-model",
            )
        )
        resp = await client.delete("/api/v1/collections/test-coll")
        assert resp.status_code == 200

    async def test_collection_sources(self, client, mock_deps):
        mock_deps["store"].get_collection = AsyncMock(
            return_value=Collection(
                id="c1",
                name="test-coll",
                dimension=128,
                model_name="test-model",
            )
        )
        mock_deps["store"].list_sources = AsyncMock(return_value=["/src/file1.py", "/src/file2.py"])
        resp = await client.get("/api/v1/collections/test-coll/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sources"]) == 2

    async def test_collection_stats(self, client, mock_deps):
        mock_deps["store"].get_collection = AsyncMock(
            return_value=Collection(
                id="c1",
                name="test-coll",
                dimension=128,
                model_name="test-model",
            )
        )
        resp = await client.get("/api/v1/collections/test-coll/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "chunk_count" in data


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


class TestChunks:
    """Tests for /chunks endpoints."""

    async def test_get_chunk(self, client, mock_deps):
        mock_deps["store"].get = AsyncMock(
            return_value={
                "chunk_id": "chunk-1",
                "content": "def hello(): pass",
                "content_type": "python",
                "chunk_type": "function",
                "source_path": "/src/file.py",
                "start_line": 1,
                "end_line": 2,
                "name": "hello",
                "parent": None,
                "metadata": {},
                "content_hash": "abc123",
                "created_at": "2025-01-01T00:00:00",
            }
        )
        resp = await client.get("/api/v1/chunks/chunk-1?collection=default")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_id"] == "chunk-1"
        assert data["content"] == "def hello(): pass"

    async def test_get_chunk_not_found(self, client, mock_deps):
        mock_deps["store"].get = AsyncMock(return_value=None)
        resp = await client.get("/api/v1/chunks/nonexistent?collection=default")
        assert resp.status_code == 404

    async def test_delete_chunk(self, client, mock_deps):
        resp = await client.delete("/api/v1/chunks/chunk-1?collection=default")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and exception mapping."""

    async def test_validation_error_returns_422(self, client):
        """Missing required field in request body."""
        resp = await client.post("/api/v1/search", json={})
        # FastAPI returns 422 for validation errors
        assert resp.status_code == 422

    async def test_embeddy_error_returns_500(self, client, mock_deps):
        """EmbeddyError maps to 500 with structured error response."""
        from embeddy.exceptions import EncodingError

        mock_deps["embedder"].encode = AsyncMock(side_effect=EncodingError("Model failed"))
        resp = await client.post(
            "/api/v1/embed",
            json={"inputs": [{"text": "test"}]},
        )
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "message" in data

    async def test_search_error_returns_500(self, client, mock_deps):
        """SearchError in search endpoint maps to 500."""
        from embeddy.exceptions import SearchError

        mock_deps["search_service"].search = AsyncMock(side_effect=SearchError("Search failed"))
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test", "collection": "default"},
        )
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "search_error"

    async def test_store_error_returns_500(self, client, mock_deps):
        """StoreError maps to 500."""
        from embeddy.exceptions import StoreError

        mock_deps["store"].list_collections = AsyncMock(side_effect=StoreError("DB corrupt"))
        resp = await client.get("/api/v1/collections")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "store_error"
