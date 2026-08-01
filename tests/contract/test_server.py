"""ASGI contract tests for embeddy/server.py — plan §8/§11.

INJECTED mocks (RecordingProvider / in-memory SqliteStore / FakeReranker)
prove the dependency-injection seam, not the real model. Covers:
  * honest health (live always, ready only when the provider is loaded)
  * the error map (400 / 404 / 413 / 422 / 500 / 501 / 503 structured shapes)
  * the request/batch size limits (OOM/crash-prevention, CONCEPT §9.7)
  * the H2 role-discipline regression (query role on embed/search routes,
    document role on ingest routes — never the other way)
  * collections / chunks / search / similar / ingest surfaces
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from embeddy.config import EmbedderSettings, PipelineSettings, ServerSettings
from embeddy.index.base import Searchable
from embeddy.registry import resolve_instruction
from embeddy.server import create_app

from .conftest import (
    REGISTRY_MODEL,
    FailingProvider,
    FakeReranker,
    RaisingProvider,
    RecordingProvider,
    StatsOnlyStore,
    seed_corpus,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


async def _make_client(app: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    await client.__aenter__()
    return client


# --------------------------------------------------------------------------- #
# health — honest readiness
# --------------------------------------------------------------------------- #


async def test_health_live_and_ready_when_ready(client: httpx.AsyncClient) -> None:
    r = await client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "live", "ready": True}
    r = await client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["ready"] is True


async def test_health_not_ready_when_provider_load_fails(
    store: Any, settings: ServerSettings
) -> None:
    """Injected state: the provider's load() raises -> the lifespan records
    not-ready and KEEPS SERVING (the honest contract). /health/live stays
    200; /health/ready is 503 with the reason."""
    app = create_app(
        store=store,
        provider=FailingProvider(),
        settings=settings,
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.get("/health/live")
            assert r.status_code == 200
            assert r.json() == {"status": "live", "ready": False}
            r = await client.get("/health/ready")
            assert r.status_code == 503
            body = r.json()
            assert body["status"] == "not_ready"
            assert body["ready"] is False
            assert "ProviderError" in (body["reason"] or "")
        finally:
            await client.aclose()


async def test_non_health_route_503_when_not_ready(store: Any) -> None:
    app = create_app(
        store=store,
        provider=FailingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post("/api/v1/search", json={"query": "q", "collection": "acme"})
            assert r.status_code == 503
            assert r.json()["error"]["type"] == "service_unavailable"
        finally:
            await client.aclose()


async def test_module_level_app_bare_state() -> None:
    """`uvicorn embeddy.server:app` works bare: the module-level app exists
    and is built from settings (store_path default, ready=False until its
    lifespan runs)."""
    from embeddy.server import app as module_app

    assert module_app.state.server.settings.store_path == "embeddy.db"
    assert module_app.state.server.ready is False


# --------------------------------------------------------------------------- #
# error map — structured shapes
# --------------------------------------------------------------------------- #


async def test_422_request_validation_shape(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/search", json={"query": 5, "collection": "acme"})
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["type"] == "validation_error"
    assert isinstance(error["detail"], list)


async def test_404_unknown_collection(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/search", json={"query": "q", "collection": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found"


async def test_404_unknown_collection_on_ingest(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/ingest/text", json={"text": "hi", "collection": "nope"})
    assert r.status_code == 404


async def test_404_unknown_chunk_in_similar(store: Any, provider: RecordingProvider) -> None:
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/api/v1/similar", json={"chunk_id": "missing", "collection": "acme"}
            )
            assert r.status_code == 404
            assert r.json()["error"]["type"] == "not_found"
        finally:
            await client.aclose()


async def test_400_retrieve_k_lt_top_k(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    r = await client.post(
        "/api/v1/search",
        json={"query": "q", "collection": "acme", "top_k": 10, "retrieve_k": 2},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "bad_request"
    assert "retrieve_k" in r.json()["error"]["message"]


async def test_400_weights_wrong_length(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    r = await client.post(
        "/api/v1/search", json={"query": "q", "collection": "acme", "weights": [1.0]}
    )
    assert r.status_code == 400
    assert "weights" in r.json()["error"]["message"]


async def test_400_unknown_model_on_embeddings(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/embeddings", json={"input": "hi", "model": "no/such-model"})
    assert r.status_code == 400
    assert "no/such-model" in r.json()["error"]["message"]


async def test_500_embeddy_error_is_structured(store: Any, settings: ServerSettings) -> None:
    """An EmbeddyError escaping a route -> 500 with type embeddy_error (the
    documented map; the RaisingProvider raises ProviderError in encode)."""
    app = create_app(
        store=store,
        provider=RaisingProvider(),
        settings=settings,
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post("/v1/embeddings", json={"input": "hi", "model": REGISTRY_MODEL})
            assert r.status_code == 500
            error = r.json()["error"]
            assert error["type"] == "embeddy_error"
            assert "encode exploded" in error["message"]
        finally:
            await client.aclose()


async def test_500_internal_error_for_non_registry_provider(
    store: Any, settings: ServerSettings
) -> None:
    """A provider whose model_name is NOT in the registry cannot resolve a
    role -> the embed route's role resolution raises RegistryError -> the
    generic 500 structured shape (documented: inject a registry-named
    provider for tests; the bare server always uses a registry model)."""
    from embeddy.providers.fake import FakeProvider

    app = create_app(
        store=store,
        provider=FakeProvider(),
        settings=settings,
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/v1/embeddings",
                json={"input": "hi", "model": "fake/deterministic-dim8"},
            )
            assert r.status_code == 500
            assert r.json()["error"]["type"] == "internal_error"
        finally:
            await client.aclose()


async def test_501_missing_store_extra() -> None:
    """A backend without the sqlite extras -> 501 (not a crash)."""
    app = create_app(
        store=cast(Searchable, StatsOnlyStore()),
        provider=RecordingProvider(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post("/api/v1/collections", json={"collection": "c"})
            assert r.status_code == 501
            assert r.json()["error"]["type"] == "not_implemented"
            r = await client.get("/api/v1/chunks", params={"collection": "c"})
            assert r.status_code == 501
        finally:
            await client.aclose()


async def test_503_no_reranker_configured(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/rerank", json={"query": "q", "texts": ["a"]})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "service_unavailable"
    r = await client.post("/api/v1/rerank", json={"query": "q", "collection": "acme"})
    assert r.status_code == 503


async def test_cors_headers_from_config() -> None:
    """CORS is applied FROM config when cors_origins is set (CONCEPT §5.7:
    "CORS applied from config when set") — on normal AND on error responses
    (the CORS middleware is outermost)."""
    app = create_app(
        store=cast(Searchable, StatsOnlyStore()),
        provider=RecordingProvider(),
        settings=ServerSettings(cors_origins=("http://allowed.example",), max_body_bytes=64),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.get("/health/live", headers={"origin": "http://allowed.example"})
            assert r.status_code == 200
            assert r.headers["access-control-allow-origin"] == "http://allowed.example"
            # error responses get the CORS header too (the 413 from the
            # body-limit middleware is inside the CORS layer)
            r = await client.post(
                "/v1/embeddings",
                json={"input": ["x" * 200], "model": REGISTRY_MODEL},
                headers={"origin": "http://allowed.example"},
            )
            assert r.status_code == 413
            assert r.headers["access-control-allow-origin"] == "http://allowed.example"
        finally:
            await client.aclose()


async def test_ingest_error_collection_on_the_wire(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    """A per-source failure is COLLECTED into the IngestStats wire shape
    (never raised): a missing file -> a READ-phase SourceError in `errors`."""
    assert (
        await client.post("/api/v1/collections", json={"collection": "acme"})
    ).status_code == 200
    r = await client.post(
        "/api/v1/ingest/file", json={"path": "/no/such/file.txt", "collection": "acme"}
    )
    assert r.status_code == 200
    stats = r.json()
    assert stats["files_indexed"] == 0
    (error,) = stats["errors"]
    assert error["phase"] == "read"
    assert error["path"] == "/no/such/file.txt"
    assert error["error_type"]


# --------------------------------------------------------------------------- #
# request/batch size limits (OOM/crash-prevention ONLY — CONCEPT §9.7)
# --------------------------------------------------------------------------- #


async def test_413_body_too_large() -> None:
    app = create_app(
        store=cast(Searchable, StatsOnlyStore()),
        provider=RecordingProvider(),
        settings=ServerSettings(max_body_bytes=64),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/v1/embeddings",
                json={"input": ["x" * 100], "model": REGISTRY_MODEL},
            )
            assert r.status_code == 413
            error = r.json()["error"]
            assert error["type"] == "payload_too_large"
            assert "max_body_bytes" in error["message"]
        finally:
            await client.aclose()


async def test_413_too_many_embed_inputs(store: Any, provider: RecordingProvider) -> None:
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        settings=ServerSettings(max_embed_inputs=2),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/v1/embeddings",
                json={"input": ["a", "b", "c"], "model": REGISTRY_MODEL},
            )
            assert r.status_code == 413
            assert "max_embed_inputs" in r.json()["error"]["message"]
        finally:
            await client.aclose()


async def test_413_top_k_exceeded(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    r = await client.post("/api/v1/search", json={"query": "q", "collection": "acme", "top_k": 101})
    assert r.status_code == 413
    assert "max_top_k" in r.json()["error"]["message"]


async def test_413_retrieve_k_exceeded(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    r = await client.post(
        "/api/v1/search", json={"query": "q", "collection": "acme", "retrieve_k": 500}
    )
    assert r.status_code == 413


async def test_413_rerank_top_n_exceeded(store: Any, provider: RecordingProvider) -> None:
    app = create_app(
        store=store,
        provider=provider,
        reranker=FakeReranker(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post("/v1/rerank", json={"query": "q", "texts": ["a"], "top_n": 500})
            assert r.status_code == 413
        finally:
            await client.aclose()


# --------------------------------------------------------------------------- #
# role discipline regression (H2) — query vs document role never leaks
# --------------------------------------------------------------------------- #


async def test_embed_route_resolves_query_role(store: Any, provider: RecordingProvider) -> None:
    """/v1/embeddings must hand the QUERY-role instruction to encode — never
    the document role's string (H2; the regression the plan demands)."""
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    query_string = resolve_instruction(REGISTRY_MODEL, "query")
    document_string = resolve_instruction(REGISTRY_MODEL, "document")
    assert query_string != document_string
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/v1/embeddings", json={"input": ["hello"], "model": REGISTRY_MODEL}
            )
            assert r.status_code == 200
        finally:
            await client.aclose()
    (inputs, instruction) = provider.calls[-1]
    assert instruction == query_string
    assert instruction != document_string


async def test_search_route_resolves_query_role(store: Any, provider: RecordingProvider) -> None:
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    query_string = resolve_instruction(REGISTRY_MODEL, "query")
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/api/v1/search", json={"query": "token expiry", "collection": "acme"}
            )
            assert r.status_code == 200
        finally:
            await client.aclose()
    (inputs, instruction) = provider.calls[-1]
    assert instruction == query_string


async def test_ingest_route_resolves_document_role(store: Any, provider: RecordingProvider) -> None:
    """The DOCUMENT role is used ONLY by the pipeline (ingest routes) — the
    query side never sees it and ingest never sees the query string."""
    await seed_corpus(store, provider)  # pipeline resolved the DOCUMENT role
    document_string = resolve_instruction(REGISTRY_MODEL, "document")
    query_string = resolve_instruction(REGISTRY_MODEL, "query")
    assert provider.calls, "seed ingest must have embedded via the pipeline"
    for _inputs, instruction in provider.calls:
        assert instruction == document_string, (
            "ingest (pipeline) must never leak the query role's string into encode"
        )
    assert document_string != query_string


async def test_prompt_role_override_changes_query_instruction(
    store: Any, provider: RecordingProvider
) -> None:
    """embedder.prompt_role selects the role: with prompt_role="retrieval"
    the embed route resolves the retrieval string (the harness default is
    "query"; the registry maps query/retrieval for this model)."""
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="retrieval"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/v1/embeddings", json={"input": "hello", "model": REGISTRY_MODEL}
            )
            assert r.status_code == 200
        finally:
            await client.aclose()
    assert provider.calls[-1][1] == resolve_instruction(REGISTRY_MODEL, "retrieval")


# --------------------------------------------------------------------------- #
# collections surface
# --------------------------------------------------------------------------- #


async def test_create_collection_default_dimension(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/collections", json={"collection": "c1"})
    assert r.status_code == 200
    assert r.json() == {"collection": "c1", "dimension": 8}  # provider dimension


async def test_create_collection_explicit_dimension(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/collections", json={"collection": "c2", "dimension": 4})
    assert r.status_code == 200
    assert r.json()["dimension"] == 4


async def test_create_collection_bad_id_400(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/collections", json={"collection": "bad id!"})
    assert r.status_code == 400


async def test_create_collection_dimension_mismatch_400(
    client: httpx.AsyncClient,
) -> None:
    assert (await client.post("/api/v1/collections", json={"collection": "c"})).status_code == 200
    r = await client.post("/api/v1/collections", json={"collection": "c", "dimension": 4})
    assert r.status_code == 400


async def test_list_collections(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.get("/api/v1/collections")
    assert r.status_code == 200
    collections = r.json()["collections"]
    assert {"collection": "acme", "dimension": 8} in collections


async def test_collection_stats(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.get("/api/v1/collections/acme")
    assert r.status_code == 200
    stats = r.json()
    assert stats["collection_id"] == "acme"
    assert stats["source_count"] == 1
    assert stats["chunk_count"] >= 2
    assert stats["vector_dimension"] == 8


async def test_collection_stats_404(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/collections/nope")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# chunks surface
# --------------------------------------------------------------------------- #


async def test_list_chunks(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.get("/api/v1/chunks", params={"collection": "acme", "limit": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["collection"] == "acme"
    assert len(body["chunks"]) == 1
    chunk = body["chunks"][0]
    assert chunk["chunk_type"] == "paragraph"
    assert "acme" in chunk["content"] or "JWT" in chunk["content"]


async def test_list_chunks_404(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/chunks", params={"collection": "nope"})
    assert r.status_code == 404


async def test_list_chunks_bad_params_400(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    r = await client.get("/api/v1/chunks", params={"collection": "acme", "limit": 0})
    assert r.status_code == 400
    r = await client.get("/api/v1/chunks", params={"collection": "acme", "offset": -1})
    assert r.status_code == 400


async def test_list_chunks_limit_over_max_413(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    r = await client.get("/api/v1/chunks", params={"collection": "acme", "limit": 500})
    assert r.status_code == 413


async def test_search_chunks_fts(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.post("/api/v1/chunks/search", json={"collection": "acme", "query": "JWT"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    assert any("JWT" in hit["content"] for hit in results)
    assert results[0]["metric"] == "bm25"


async def test_search_chunks_404(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/chunks/search", json={"collection": "nope", "query": "x"})
    assert r.status_code == 404


async def test_search_chunks_min_score_invalid_400(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    await seed_corpus(store, provider)
    # raw body: httpx refuses to serialize NaN, but the server must reject
    # it with a clean 400 (min_score must be finite).
    import json

    payload = json.dumps({"collection": "acme", "query": "x", "min_score": float("nan")})
    r = await client.post(
        "/api/v1/chunks/search", content=payload, headers={"content-type": "application/json"}
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# search surface
# --------------------------------------------------------------------------- #


async def test_search_hybrid(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.post(
        "/api/v1/search", json={"query": "token expiry policy", "collection": "acme"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["metric"] == "rrf"
    assert body["mode"] == "rrf"
    assert body["query"] == "token expiry policy"
    assert body["collection"] == "acme"
    assert body["total_results"] >= 1
    assert body["results"]
    for hit in body["results"]:
        assert hit["metric"] == "rrf"
    # RRF fusion surfaces both corpus paragraphs; the FTS leg (porter
    # stemming) recalls the token paragraph for "token".
    assert any("token" in hit["content"].lower() for hit in body["results"])
    scores = [hit["score"] for hit in body["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_search_weighted_mode(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.post(
        "/api/v1/search",
        json={"query": "token", "collection": "acme", "mode": "weighted"},
    )
    assert r.status_code == 200
    assert r.json()["metric"] == "weighted"


async def test_search_with_filters(
    store: Any, provider: RecordingProvider, client: httpx.AsyncClient
) -> None:
    await seed_corpus(store, provider)
    r = await client.post(
        "/api/v1/search",
        json={
            "query": "token",
            "collection": "acme",
            "filters": {"content_types": ["markdown"]},
        },
    )
    assert r.status_code == 200
    # a filter that excludes everything -> empty results, honest total
    r = await client.post(
        "/api/v1/search",
        json={
            "query": "token",
            "collection": "acme",
            "filters": {"content_types": ["python"]},
        },
    )
    assert r.status_code == 200
    assert r.json()["results"] == []


async def test_similar_excludes_query_chunk(store: Any, provider: RecordingProvider) -> None:
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            chunks = (await client.get("/api/v1/chunks", params={"collection": "acme"})).json()[
                "chunks"
            ]
            assert len(chunks) >= 2
            target = chunks[0]["chunk_id"]
            r = await client.post(
                "/api/v1/similar",
                json={"chunk_id": target, "collection": "acme", "top_k": 5},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["chunk_id"] == target
            assert all(hit["chunk_id"] != target for hit in body["results"])
            assert len(body["results"]) == len(chunks) - 1
        finally:
            await client.aclose()


async def test_search_rerank_runs_rerank_stage(store: Any, provider: RecordingProvider) -> None:
    await seed_corpus(store, provider)
    app = create_app(
        store=store,
        provider=provider,
        reranker=FakeReranker(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post("/api/v1/rerank", json={"query": "token", "collection": "acme"})
            assert r.status_code == 200
            body = r.json()
            assert body["metric"] == "rerank"
            assert all(hit["metric"] == "rerank" for hit in body["results"])
            assert body["results"]
            scores = [hit["score"] for hit in body["results"]]
            assert scores == sorted(scores, reverse=True)
        finally:
            await client.aclose()


async def test_v1_rerank_tei_jina_shape(store: Any, provider: RecordingProvider) -> None:
    app = create_app(
        store=store,
        provider=provider,
        reranker=FakeReranker(),
        settings=ServerSettings(),
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with app.router.lifespan_context(app):
        client = await _make_client(app)
        try:
            r = await client.post(
                "/v1/rerank", json={"query": "q", "texts": ["a", "b", "c"], "top_n": 2}
            )
            assert r.status_code == 200
            body = r.json()
            assert [hit["index"] for hit in body["results"]] == [2, 1]  # reversed, top 2
        finally:
            await client.aclose()


# --------------------------------------------------------------------------- #
# ingest surface
# --------------------------------------------------------------------------- #


async def test_ingest_text_route(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    assert (
        await client.post("/api/v1/collections", json={"collection": "acme"})
    ).status_code == 200
    r = await client.post(
        "/api/v1/ingest/text",
        json={"text": "hello embeddy world", "collection": "acme", "path": "m.md"},
    )
    assert r.status_code == 200
    stats = r.json()
    assert stats["files_attempted"] == 1
    assert stats["files_indexed"] == 1
    assert stats["chunks_indexed"] == 1
    assert stats["errors"] == []
    # dedup: identical content at the same path -> skipped
    r = await client.post(
        "/api/v1/ingest/text",
        json={"text": "hello embeddy world", "collection": "acme", "path": "m.md"},
    )
    assert r.json()["files_skipped"] == 1
    # searchable
    r = await client.post("/api/v1/chunks/search", json={"collection": "acme", "query": "embeddy"})
    assert r.json()["results"]


async def test_ingest_text_404_missing_collection(client: httpx.AsyncClient) -> None:
    r = await client.post("/api/v1/ingest/text", json={"text": "hi", "collection": "nope"})
    assert r.status_code == 404


async def test_ingest_file_route(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider, tmp_path: Path
) -> None:
    assert (
        await client.post("/api/v1/collections", json={"collection": "acme"})
    ).status_code == 200
    f = tmp_path / "a.txt"
    f.write_text("ingest file content here")
    r = await client.post("/api/v1/ingest/file", json={"path": str(f), "collection": "acme"})
    assert r.status_code == 200
    assert r.json()["files_indexed"] == 1
    r = await client.post("/api/v1/chunks/search", json={"collection": "acme", "query": "ingest"})
    assert r.json()["results"]


async def test_ingest_directory_route(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider, tmp_path: Path
) -> None:
    assert (
        await client.post("/api/v1/collections", json={"collection": "acme"})
    ).status_code == 200
    (tmp_path / "a.txt").write_text("alpha content")
    (tmp_path / "b.txt").write_text("beta content")
    r = await client.post(
        "/api/v1/ingest/dir", json={"directory": str(tmp_path), "collection": "acme"}
    )
    assert r.status_code == 200
    assert r.json()["files_indexed"] == 2


async def test_ingest_reindex_and_delete_routes(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider, tmp_path: Path
) -> None:
    assert (
        await client.post("/api/v1/collections", json={"collection": "acme"})
    ).status_code == 200
    f = tmp_path / "a.txt"
    f.write_text("original words")
    assert (
        await client.post("/api/v1/ingest/file", json={"path": str(f), "collection": "acme"})
    ).json()["files_indexed"] == 1
    f.write_text("replaced words entirely")
    r = await client.post("/api/v1/ingest/reindex", json={"path": str(f), "collection": "acme"})
    assert r.json()["files_indexed"] == 1
    # old FTS terms are gone (atomic swap)
    r = await client.post("/api/v1/chunks/search", json={"collection": "acme", "query": "original"})
    assert r.json()["results"] == []
    r = await client.post("/api/v1/ingest/delete", json={"path": str(f), "collection": "acme"})
    assert r.json() == {"deleted": True}
    r = await client.post("/api/v1/ingest/delete", json={"path": str(f), "collection": "acme"})
    assert r.json() == {"deleted": False}


async def test_ingest_sync_route(
    client: httpx.AsyncClient, store: Any, provider: RecordingProvider, tmp_path: Path
) -> None:
    assert (
        await client.post("/api/v1/collections", json={"collection": "acme"})
    ).status_code == 200
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "a.txt").write_text("alpha")
    (seed / "old.txt").write_text("gone")
    r = await client.post("/api/v1/ingest/dir", json={"directory": str(seed), "collection": "acme"})
    assert r.json()["files_indexed"] == 2
    (seed / "old.txt").unlink()
    r = await client.post(
        "/api/v1/ingest/sync", json={"directory": str(seed), "collection": "acme"}
    )
    assert r.status_code == 200
    assert r.json()["files_deleted"] == 1
