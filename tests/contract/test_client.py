"""Contract tests: EmbeddyClient <-> live app for EVERY endpoint — plan
§8/§11 (payload parity: the client and server speak one protocol; the
client's embed() shares build_embeddings_request with the HTTPProvider).

The app runs in-process over ASGITransport with injected mocks; the client
talks to it exactly as it would to a real uvicorn server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from embeddy.client import EmbeddyClient
from embeddy.config import EmbedderSettings, PipelineSettings, ServerSettings
from embeddy.errors import ClientError
from embeddy.server import create_app

from .conftest import (
    REGISTRY_MODEL,
    FakeReranker,
    RecordingProvider,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def contract_app(store: Any, provider: RecordingProvider, settings: ServerSettings) -> Any:
    application = create_app(
        store=store,
        provider=provider,
        reranker=FakeReranker(),
        settings=settings,
        embedder=EmbedderSettings(prompt_role="query"),
        pipeline=PipelineSettings(concurrency=2),
    )
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def contract_client(contract_app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=contract_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_contract_every_endpoint(
    store: Any, provider: RecordingProvider, contract_client: httpx.AsyncClient
) -> None:
    """One end-to-end contract walk exercising EVERY client method against
    the live app: health -> collections -> ingest -> search -> similar ->
    rerank -> chunks -> delete."""

    # health ----------------------------------------------------------------
    ready = await contract_client_health(contract_client)
    assert ready["ready"] is True

    # --- create + list + stats
    created = await contract_client_create_collection(contract_client, "acme")
    assert created == {"collection": "acme", "dimension": 8}
    listed = await contract_client_list_collections(contract_client)
    assert {"collection": "acme", "dimension": 8} in listed["collections"]
    stats = await contract_client_stats(contract_client, "acme")
    assert stats["collection_id"] == "acme"

    # --- ingest text (document role) + verify searchable
    ingested = await contract_client_ingest_text(contract_client, "acme")
    assert ingested["files_indexed"] == 1
    assert ingested["chunks_indexed"] >= 1

    # --- chunks list + chunks search
    chunks = await contract_client_list_chunks(contract_client, "acme")
    assert chunks["chunks"]
    hit = await contract_client_search_chunks(contract_client, "acme", "JWT")
    assert hit["results"]

    # --- search + similar + rerank
    result = await contract_client_search(contract_client, "acme", "token expiry")
    assert result["results"]
    assert result["metric"] == "rrf"
    target = chunks["chunks"][0]["chunk_id"]
    similar = await contract_client_similar(contract_client, "acme", target)
    assert all(h["chunk_id"] != target for h in similar["results"])
    reranked = await contract_client_rerank(contract_client, "acme", "token")
    assert reranked["metric"] == "rerank"

    # --- /v1/embeddings + /v1/rerank (OpenAI-compatible + TEI/Jina)
    embedded = await contract_client_embed(contract_client)
    assert len(embedded["data"][0]["embedding"]) == 8
    tei = await contract_client_v1_rerank(contract_client)
    assert tei["results"]

    # --- file + dir ingest, reindex, sync, delete
    await contract_client_file_and_dir(contract_client, store, provider)

    # --- error surface through the client
    with pytest.raises(ClientError) as excinfo:
        await contract_client_collection_stats(contract_client, "missing")
    assert excinfo.value.status_code == 404


# Each client method is exercised through the REAL EmbeddyClient. The
# helpers below keep the walk readable; the assertions pin the wire shapes.


async def contract_client_health(c: httpx.AsyncClient) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    live = await client.health_live()
    assert live["status"] == "live"
    return await client.health_ready()


async def contract_client_create_collection(c: httpx.AsyncClient, name: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.create_collection(name)


async def contract_client_list_collections(c: httpx.AsyncClient) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.list_collections()


async def contract_client_stats(c: httpx.AsyncClient, name: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.collection_stats(name)


async def contract_client_ingest_text(c: httpx.AsyncClient, name: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.ingest_text(
        "The acme authentication service issues short-lived JWTs.",
        name,
        path="docs/auth.md",
        content_type="markdown",
    )


async def contract_client_list_chunks(c: httpx.AsyncClient, name: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.list_chunks(name, limit=10)


async def contract_client_search_chunks(
    c: httpx.AsyncClient, name: str, query: str
) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.search_chunks(name, query, top_k=5)


async def contract_client_search(c: httpx.AsyncClient, name: str, query: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.search(query, name, top_k=5)


async def contract_client_similar(c: httpx.AsyncClient, name: str, chunk_id: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.similar(chunk_id, name, top_k=5)


async def contract_client_rerank(c: httpx.AsyncClient, name: str, query: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.search_rerank(query, name, top_k=5)


async def contract_client_embed(c: httpx.AsyncClient) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.embed(["hello world"], model=REGISTRY_MODEL)


async def contract_client_v1_rerank(c: httpx.AsyncClient) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.rerank("q", ["a", "b", "c"], top_n=2)


async def contract_client_collection_stats(c: httpx.AsyncClient, name: str) -> dict[str, Any]:
    client = EmbeddyClient("http://test", http_client=c)
    return await client.collection_stats(name)


async def contract_client_file_and_dir(
    c: httpx.AsyncClient, store: Any, provider: RecordingProvider
) -> None:
    import tempfile

    client = EmbeddyClient("http://test", http_client=c)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        f = root / "a.txt"
        f.write_text("contract file ingest")
        # file ingest
        stats = await client.ingest_file(str(f), "acme")
        assert stats["files_indexed"] == 1
        # reindex after a content change (atomic swap; unchanged content is
        # skipped by the pipeline dedup policy)
        f.write_text("contract file ingest CHANGED")
        stats = await client.reindex(str(f), "acme")
        assert stats["files_indexed"] == 1
        # dir ingest: the file route stored the ABSOLUTE path, the dir
        # route stores collection-relative paths -> both files are new
        # sources (dedup is per (collection, path))
        (root / "b.txt").write_text("second contract file")
        stats = await client.ingest_directory(str(root), "acme")
        assert stats["files_indexed"] == 2  # a.txt (rel) + b.txt
        # sync: b.txt removed -> deleted (the out-of-tree sources too:
        # docs/auth.md and the absolute-path a.txt are not under `root`)
        (root / "b.txt").unlink()
        stats = await client.sync(str(root), "acme")
        assert stats["files_deleted"] == 3
        # delete the remaining relative-path source
        deleted = await client.delete_source("a.txt", "acme")
        assert deleted["deleted"] is True
        deleted = await client.delete_source("a.txt", "acme")
        assert deleted["deleted"] is False
