"""Phase-8 contract tests — the server with a QdrantStore backend (plan
§10/§12 M7 gate). The qdrant in-memory backend is hermetic (probed), so
these run in the default suite like the sqlite contract tests.

Covers:
  * store selection via `store_settings.url` (the one config line): the
    lifespan opens a QdrantStore when url="qdrant://:memory:", the sqlite
    fallback when url=None (bare-server behavior unchanged)
  * the honest-health contract with an unreachable qdrant (not-ready +
    reason, never a half-open store)
  * the full product surface on a QdrantStore: create collection, ingest
    (pipeline), hybrid search, similar, collections, chunks, stats —
    WITHOUT any wire-shape change (the server is backend-agnostic)
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from embeddy.config import EmbedderSettings, PipelineSettings, ServerSettings, StoreSettings
from embeddy.index.factory import parse_store_url
from embeddy.index.qdrant import QdrantStore
from embeddy.protocol.types import Metric
from embeddy.server import create_app

from .conftest import (
    RecordingProvider,
)

pytestmark = pytest.mark.integration


async def _make_client(app: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    await client.__aenter__()
    return client


# --------------------------------------------------------------------------- #
# injected QdrantStore — the DI seam is backend-agnostic
# --------------------------------------------------------------------------- #


async def test_health_ready_with_injected_qdrant_store() -> None:
    store = await QdrantStore.open(parse_store_url("qdrant://:memory:"))
    app = create_app(
        store=store,
        provider=RecordingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.get("/health/ready")
            assert r.status_code == 200
            assert r.json()["ready"] is True
        finally:
            await client.aclose()
    await store.close()


async def test_full_product_surface_on_qdrant() -> None:
    """create collection -> ingest text (pipeline) -> hybrid search ->
    similar -> collections/chunks/stats. No wire change: the same routes
    the sqlite backend serves, with a QdrantStore underneath."""
    store = await QdrantStore.open(parse_store_url("qdrant://:memory:"))
    app = create_app(
        store=store,
        provider=RecordingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            # create a collection (the store extra maps to qdrant)
            r = await client.post(
                "/api/v1/collections", json={"collection": "acme", "dimension": 8}
            )
            assert r.status_code == 200, r.text
            assert r.json()["dimension"] == 8

            # ingest text through the pipeline (FakeProvider vectors)
            r = await client.post(
                "/api/v1/ingest/text",
                json={
                    "text": "The acme billing API records usage in buckets.",
                    "collection": "acme",
                    "path": "docs/billing.md",
                    "content_type": "markdown",
                },
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["chunks_indexed"] >= 1

            # hybrid search
            r = await client.post(
                "/api/v1/search",
                json={"query": "billing buckets", "collection": "acme", "top_k": 5},
            )
            assert r.status_code == 200, r.text
            result = r.json()
            assert result["collection"] == "acme"
            assert result["metric"] in (Metric.RRF.value, Metric.WEIGHTED.value)
            assert result["total_results"] >= 1

            # similar (re-embed an existing chunk; get_chunk is a qdrant extra)
            chunk_id = result["results"][0]["chunk_id"]
            r = await client.post(
                "/api/v1/similar",
                json={"chunk_id": chunk_id, "collection": "acme", "top_k": 3},
            )
            assert r.status_code == 200, r.text

            # collections listing + stats
            r = await client.get("/api/v1/collections")
            assert r.status_code == 200, r.text
            assert [c["collection"] for c in r.json()["collections"]] == ["acme"]
            r = await client.get("/api/v1/collections/acme")
            assert r.status_code == 200, r.text
            stats = r.json()
            assert stats["chunk_count"] >= 1
            assert stats["source_count"] == 1
            assert stats["vector_dimension"] == 8

            # chunks listing
            r = await client.get("/api/v1/chunks", params={"collection": "acme"})
            assert r.status_code == 200, r.text
            assert len(r.json()["chunks"]) >= 1
        finally:
            await client.aclose()
    await store.close()


async def test_missing_collection_404_on_qdrant() -> None:
    store = await QdrantStore.open(parse_store_url("qdrant://:memory:"))
    app = create_app(
        store=store,
        provider=RecordingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/api/v1/search",
                json={"query": "x", "collection": "nope", "top_k": 5},
            )
            assert r.status_code == 404, r.text
            assert r.json()["error"]["type"] == "not_found"
        finally:
            await client.aclose()
    await store.close()


# --------------------------------------------------------------------------- #
# store selection via the lifespan (no injected store)
# --------------------------------------------------------------------------- #


async def test_lifespan_opens_qdrant_from_store_url() -> None:
    """`store.url = "qdrant://:memory:"` — the ONE config line — makes the
    bare server open a QdrantStore (Phase 8 work item 3)."""
    app = create_app(
        provider=RecordingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
        store_settings=StoreSettings(url="qdrant://:memory:"),
    )
    async with app.router.lifespan_context(app):
        state = cast(Any, app.state).server
        assert isinstance(state.store, QdrantStore)
        client = await _make_client(app)
        try:
            r = await client.get("/health/ready")
            assert r.status_code == 200
            assert r.json()["ready"] is True
            # a real qdrant-backed route works
            r = await client.post("/api/v1/collections", json={"collection": "c", "dimension": 8})
            assert r.status_code == 200, r.text
        finally:
            await client.aclose()


async def test_lifespan_store_url_none_uses_sqlite_store_path() -> None:
    """`store.url = None` falls back to the server's sqlite store_path — the
    bare `app = create_app()` behavior is unchanged (M5/M6 contract)."""
    from embeddy.index.sqlite import SqliteStore

    app = create_app(
        provider=RecordingProvider(),
        settings=ServerSettings(store_path=":memory:"),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
        store_settings=StoreSettings(url=None),
    )
    async with app.router.lifespan_context(app):
        state = cast(Any, app.state).server
        assert isinstance(state.store, SqliteStore)
        client = await _make_client(app)
        try:
            r = await client.get("/health/ready")
            assert r.status_code == 200
            assert r.json()["ready"] is True
        finally:
            await client.aclose()


async def test_lifespan_unreachable_qdrant_reports_not_ready() -> None:
    """The honest-health contract with a dead qdrant: the lifespan records
    not-ready + reason (never a half-open store); /health/ready is 503."""
    app = create_app(
        provider=RecordingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
        store_settings=StoreSettings(url="qdrant://127.0.0.1:1"),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.get("/health/live")
            assert r.status_code == 200
            assert r.json()["ready"] is False
            r = await client.get("/health/ready")
            assert r.status_code == 503
            body = r.json()
            assert body["ready"] is False
            assert "StoreError" in (body["reason"] or "")
            assert "cannot reach qdrant" in (body["reason"] or "")
        finally:
            await client.aclose()
