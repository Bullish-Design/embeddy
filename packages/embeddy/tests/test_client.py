"""EmbeddyClient unit tests — httpx MockTransport (plan §11 contract layer,
no live server). Covers: the shared wire shape with the HTTPProvider
(build_embeddings_request — one protocol, the C5 class is impossible),
response parsing, ClientError on non-2xx, health answers, and every route's
request/response mapping.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from embeddy.client import EmbeddyClient
from embeddy.errors import ClientError
from embeddy.providers.http import build_embeddings_request


def _client(
    handler: Any,
) -> tuple[EmbeddyClient, httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://srv")
    return EmbeddyClient("http://srv", http_client=http_client), http_client


async def test_embed_wire_parity_with_http_provider() -> None:
    """The client's embed() and the HTTPProvider build the EXACT same wire
    request (one protocol — the C5 class of bug is structurally impossible).
    The provider and client share build_embeddings_request; here we capture
    what the CLIENT sends and assert it equals what the PROVIDER would send."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0, 0.0, 0.0]}]})

    client, http_client = _client(handler)
    try:
        result = await client.embed(
            ["alpha", "beta"], model="m", instruction="some-resolved-string"
        )
        assert result["data"][0]["embedding"] == [1.0, 0.0, 0.0, 0.0]
        (request,) = captured
        assert request.method == "POST"
        assert str(request.url).endswith("/v1/embeddings")
        expected = build_embeddings_request(
            base_url="http://srv",
            model="m",
            inputs=["alpha", "beta"],
            instruction="some-resolved-string",
        )
        assert request.content == expected.content
        body = json.loads(request.content)
        assert body == {
            "input": ["alpha", "beta"],
            "model": "m",
            "instruction": "some-resolved-string",
        }
    finally:
        await http_client.aclose()


async def test_embed_single_string_wrapped() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0, 0, 0]}]})

    client, http_client = _client(handler)
    try:
        await client.embed("solo", model="m")
        assert captured[0]["input"] == ["solo"]
    finally:
        await http_client.aclose()


async def test_rerank_tei_jina_shape() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, json={"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]}
        )

    client, http_client = _client(handler)
    try:
        result = await client.rerank("q", ["a", "b"], top_n=5)
        assert result["results"][0]["index"] == 1
        body = captured[0]
        assert body == {"query": "q", "texts": ["a", "b"], "top_n": 5}
    finally:
        await http_client.aclose()


async def test_search_request_shape() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [], "total_results": 0, "metric": "rrf"})

    client, http_client = _client(handler)
    try:
        await client.search(
            "hello",
            "acme",
            top_k=7,
            weights=[0.5, 0.5],
            filters={"content_types": ["markdown"], "metadata_match": {"k": "v"}},
        )
        body = captured[0]
        assert body["query"] == "hello"
        assert body["collection"] == "acme"
        assert body["top_k"] == 7
        assert body["weights"] == [0.5, 0.5]
        assert body["filters"] == {
            "content_types": ["markdown"],
            "metadata_match": {"k": "v"},
        }
        assert "min_score" not in body  # omitted when None
    finally:
        await http_client.aclose()


async def test_ingest_text_request_shape() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "files_attempted": 1,
                "files_indexed": 1,
                "files_skipped": 0,
                "files_deleted": 0,
                "chunks_indexed": 2,
                "errors": [],
            },
        )

    client, http_client = _client(handler)
    try:
        from datetime import datetime

        stats = await client.ingest_text(
            "hello world", "acme", path="p.txt", content_type="markdown"
        )
        assert stats["files_indexed"] == 1
        assert captured[0] == {
            "text": "hello world",
            "collection": "acme",
            "path": "p.txt",
            "content_type": "markdown",
        }
        await client.ingest_text("x", "acme", mtime=datetime(2026, 1, 1, 12, 30))
        assert captured[1]["mtime"] == "2026-01-01T12:30:00"
    finally:
        await http_client.aclose()


async def test_health_live_and_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "live", "ready": True})
        return httpx.Response(503, json={"status": "not_ready", "ready": False, "reason": "x"})

    client, http_client = _client(handler)
    try:
        live = await client.health_live()
        assert live["status"] == "live"
        ready = await client.health_ready()
        assert ready["ready"] is False  # a 503 is an ANSWER, not an error
    finally:
        await http_client.aclose()


async def test_non_2xx_raises_client_error_with_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": {"type": "not_found", "message": "unknown collection"}}
        )

    client, http_client = _client(handler)
    try:
        with pytest.raises(ClientError) as excinfo:
            await client.search("q", "nope")
        assert excinfo.value.status_code == 404
        assert "unknown collection" in str(excinfo.value)
    finally:
        await http_client.aclose()


async def test_transport_error_raises_client_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, http_client = _client(handler)
    try:
        with pytest.raises(ClientError, match="failed"):
            await client.list_collections()
    finally:
        await http_client.aclose()


async def test_collection_stats_path_encoded() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"collection_id": "acme", "chunk_count": 0})

    client, http_client = _client(handler)
    try:
        await client.collection_stats("acme")
        assert seen == ["/api/v1/collections/acme"]
    finally:
        await http_client.aclose()


async def test_list_chunks_params() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.url.query.decode()))
        return httpx.Response(200, json={"collection": "acme", "chunks": []})

    client, http_client = _client(handler)
    try:
        await client.list_chunks("acme", limit=5, offset=10)
        assert seen == [("/api/v1/chunks", "collection=acme&limit=5&offset=10")]
    finally:
        await http_client.aclose()


async def test_delete_and_sync_shapes() -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"deleted": True})

    client, http_client = _client(handler)
    try:
        await client.delete_source("a.txt", "acme")
        assert captured[-1] == ("/api/v1/ingest/delete", {"path": "a.txt", "collection": "acme"})
        await client.sync("/data", "acme")
        assert captured[-1] == ("/api/v1/ingest/sync", {"directory": "/data", "collection": "acme"})
    finally:
        await http_client.aclose()


async def test_close_closes_injected_client() -> None:
    """close() closes the injected http_client (the CLI/contract pattern)."""
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    http_client = httpx.AsyncClient(transport=transport)
    client = EmbeddyClient("http://srv", http_client=http_client)
    await client.close()
    assert http_client.is_closed


async def test_owned_client_creation_and_close() -> None:
    """EmbeddyClient WITHOUT an injected http_client creates (and closes) its
    OWN AsyncClient — the real CLI path (embeddy/cli.py _make_client).
    Nothing listens on port 1: the transport error surfaces as ClientError
    and close() releases the owned client."""
    client = EmbeddyClient("http://127.0.0.1:1")
    try:
        with pytest.raises(ClientError, match="failed"):
            async with client:
                await client.health_live()
    finally:
        await client.close()
    assert client._owns_client is None or client._owns_client.is_closed


async def test_similar_request_shape() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [], "collection": "acme", "chunk_id": "c1"})

    client, http_client = _client(handler)
    try:
        await client.similar("c1", "acme", top_k=3, filters={"chunk_types": ["function"]})
        assert captured[0] == {
            "chunk_id": "c1",
            "collection": "acme",
            "top_k": 3,
            "filters": {"chunk_types": ["function"]},
        }
        await client.similar("c1", "acme")
        assert "filters" not in captured[1]  # omitted when None
    finally:
        await http_client.aclose()


async def test_search_rerank_request_shape() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [], "metric": "rerank"})

    client, http_client = _client(handler)
    try:
        await client.search_rerank("q", "acme", rerank_top_k=3, top_k=5)
        body = captured[0]
        assert body["rerank_top_k"] == 3
        assert body["top_k"] == 5
        await client.search_rerank("q", "acme")
        assert "rerank_top_k" not in captured[1]
    finally:
        await http_client.aclose()


async def test_create_collection_with_and_without_dimension() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"collection": "c", "dimension": 8})

    client, http_client = _client(handler)
    try:
        await client.create_collection("c")
        assert captured[0] == {"collection": "c"}
        await client.create_collection("c", dimension=4)
        assert captured[1] == {"collection": "c", "dimension": 4}
    finally:
        await http_client.aclose()


async def test_list_collections() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method + " " + request.url.path)
        return httpx.Response(200, json={"collections": [{"collection": "c", "dimension": 8}]})

    client, http_client = _client(handler)
    try:
        result = await client.list_collections()
        assert result["collections"][0]["collection"] == "c"
        assert seen == ["GET /api/v1/collections"]
    finally:
        await http_client.aclose()


async def test_search_chunks_with_min_score_and_filters() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"collection": "acme", "query": "x", "results": []})

    client, http_client = _client(handler)
    try:
        await client.search_chunks(
            "acme", "x", top_k=4, min_score=-5.0, filters={"content_types": ["md"]}
        )
        body = captured[0]
        assert body["min_score"] == -5.0
        assert body["filters"] == {"content_types": ["md"]}
        await client.search_chunks("acme", "x")
        assert "min_score" not in captured[1]
    finally:
        await http_client.aclose()


async def test_ingest_file_directory_reindex() -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"files_indexed": 1})

    client, http_client = _client(handler)
    try:
        await client.ingest_file("/a.txt", "acme")
        assert captured[0] == ("/api/v1/ingest/file", {"path": "/a.txt", "collection": "acme"})
        await client.ingest_directory("/dir", "acme")
        assert captured[1] == ("/api/v1/ingest/dir", {"directory": "/dir", "collection": "acme"})
        await client.reindex("/a.txt", "acme")
        assert captured[2] == (
            "/api/v1/ingest/reindex",
            {"path": "/a.txt", "collection": "acme"},
        )
    finally:
        await http_client.aclose()


async def test_rerank_without_top_n() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    client, http_client = _client(handler)
    try:
        await client.rerank("q", ["a", "b"])
        assert captured[0] == {"query": "q", "texts": ["a", "b"]}  # no top_n
    finally:
        await http_client.aclose()


async def test_health_ready_non_json_payload() -> None:
    """health_ready tolerates a non-JSON / non-dict body (a 503 with a plain
    text body must still surface as an answer, never a crash)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"not json at all")

    client, http_client = _client(handler)
    try:
        payload = await client.health_ready()
        assert payload["ready"] is False
    finally:
        await http_client.aclose()


async def test_embed_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("downstream down")

    client, http_client = _client(handler)
    try:
        with pytest.raises(ClientError, match="embeddings request failed"):
            await client.embed("hi", model="m")
    finally:
        await http_client.aclose()


async def test_error_message_fallbacks() -> None:
    """ClientError message extraction falls back when the server's error
    shape is missing or malformed (never crashes on a weird body)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"plain text upstream error")

    client, http_client = _client(handler)
    try:
        with pytest.raises(ClientError) as excinfo:
            await client.list_collections()
        assert excinfo.value.status_code == 500
        assert "plain text upstream error" in str(excinfo.value)
    finally:
        await http_client.aclose()
