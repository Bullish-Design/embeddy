"""embeddy/client.py — httpx EmbeddyClient mirroring EVERY server route.

Plan §8 / CONCEPT §5.8: the client and the server speak ONE wire protocol.
`embed` reuses the SHARED request builder `build_embeddings_request`
(providers/http.py) so the HTTPProvider and the client can never drift — the
C5 class of bug (two incompatible wire shapes) is structurally impossible.
`/v1/rerank` uses the TEI/Jina shape (`{"query","texts","top_n"}`), never a
non-existent OpenAI /v1/rerank standard.

Every route gets a method with the same request/response shapes the server
defines (embeddy/server.py). Non-2xx responses raise `ClientError` carrying
the status code and the server's structured error detail — except
`health_ready()`, where a 503 is an ANSWER (not an error) and is returned
as `{"status": "not_ready", "ready": False, ...}`.

httpx is an `embeddy[client]` extra and is imported LAZILY inside the
methods (the providers/http.py pattern); `embeddy/__init__.py` never imports
this module, so zero-extras `import embeddy` stays clean.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime
from typing import Any, cast

from embeddy.errors import ClientError
from embeddy.providers.http import build_embeddings_request

__all__ = ["EmbeddyClient"]

_DEFAULT_TIMEOUT = 60.0


class EmbeddyClient:
    """httpx async client for the embeddy API server.

    `http_client` injects an httpx.AsyncClient (tests: MockTransport or an
    ASGITransport over a live app). Without one, the client owns and closes
    its own AsyncClient.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        api_key: str | None = None,
        http_client: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        self._http_client = http_client
        self._owns_client: Any = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
        if self._owns_client is not None:
            await self._owns_client.aclose()
            self._owns_client = None

    async def __aenter__(self) -> EmbeddyClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        if self._owns_client is None:
            import httpx

            self._owns_client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._owns_client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
    ) -> Any:
        import httpx

        try:
            response = await self._client().request(method, path, json=json, params=params)
        except httpx.HTTPError as exc:
            raise ClientError(f"request {method} {path!r} failed: {exc}") from exc
        return _handle(response, path=path)

    # ------------------------------------------------------------------ #
    # health
    # ------------------------------------------------------------------ #

    async def health_live(self) -> dict[str, Any]:
        """GET /health/live — the process is up (200 always)."""
        payload = await self._request("GET", "/health/live")
        return cast(dict[str, Any], payload)

    async def health_ready(self) -> dict[str, Any]:
        """GET /health/ready — provider loaded and store open. A 503 here is
        an ANSWER (not-ready), returned as a dict, never raised."""
        import httpx

        try:
            response = await self._client().get("/health/ready")
        except httpx.HTTPError as exc:
            raise ClientError(f"health request failed: {exc}") from exc
        try:
            payload = response.json()
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            payload = {"status": "unexpected", "ready": False}
        return payload

    # ------------------------------------------------------------------ #
    # OpenAI-compatible surface (shared request builder — one protocol)
    # ------------------------------------------------------------------ #

    async def embed(
        self,
        inputs: str | list[str],
        *,
        model: str,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        """POST /v1/embeddings. Built with `build_embeddings_request` — the
        EXACT request shape the HTTPProvider sends (one protocol, C5 class
        impossible). `model` must match the server's provider."""
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        request = build_embeddings_request(
            base_url=self.base_url,
            model=model,
            inputs=texts,
            instruction=instruction,
            api_key=self._api_key,
        )
        import httpx

        try:
            response = await self._client().send(request)
        except httpx.HTTPError as exc:
            raise ClientError(f"embeddings request failed: {exc}") from exc
        return cast(dict[str, Any], _handle(response, path="/v1/embeddings"))

    async def rerank(
        self,
        query: str,
        texts: list[str],
        *,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """POST /v1/rerank — the TEI/Jina shape (CONCEPT §7)."""
        body: dict[str, object] = {"query": query, "texts": texts}
        if top_n is not None:
            body["top_n"] = top_n
        return cast(dict[str, Any], await self._request("POST", "/v1/rerank", json=body))

    # ------------------------------------------------------------------ #
    # search surface
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        collection: str,
        *,
        top_k: int = 10,
        mode: str = "rrf",
        weights: list[float] | None = None,
        retrieve_k: int = 50,
        min_score: float | None = None,
        raw: bool = False,
        filters: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/search — hybrid search (the server resolves the
        QUERY role and embeds `query`)."""
        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/api/v1/search",
                json=_search_body(
                    query=query,
                    collection=collection,
                    top_k=top_k,
                    mode=mode,
                    weights=weights,
                    retrieve_k=retrieve_k,
                    min_score=min_score,
                    raw=raw,
                    filters=filters,
                ),
            ),
        )

    async def similar(
        self,
        chunk_id: str,
        collection: str,
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/similar — re-embed the chunk at `chunk_id` and
        return its nearest neighbours (excluding itself)."""
        body: dict[str, object] = {"chunk_id": chunk_id, "collection": collection, "top_k": top_k}
        if filters:
            body["filters"] = filters
        return cast(dict[str, Any], await self._request("POST", "/api/v1/similar", json=body))

    async def search_rerank(
        self,
        query: str,
        collection: str,
        *,
        rerank_top_k: int | None = None,
        top_k: int = 10,
        mode: str = "rrf",
        weights: list[float] | None = None,
        retrieve_k: int = 50,
        min_score: float | None = None,
        raw: bool = False,
        filters: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/rerank — hybrid search plus the rerank stage."""
        body = _search_body(
            query=query,
            collection=collection,
            top_k=top_k,
            mode=mode,
            weights=weights,
            retrieve_k=retrieve_k,
            min_score=min_score,
            raw=raw,
            filters=filters,
        )
        if rerank_top_k is not None:
            body["rerank_top_k"] = rerank_top_k
        return cast(dict[str, Any], await self._request("POST", "/api/v1/rerank", json=body))

    # ------------------------------------------------------------------ #
    # ingest surface
    # ------------------------------------------------------------------ #

    async def ingest_text(
        self,
        text: str,
        collection: str,
        *,
        path: str = "<memory>",
        content_type: str | None = None,
        mtime: datetime | str | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/ingest/text — raw text as one source."""
        body: dict[str, object] = {"text": text, "collection": collection, "path": path}
        if content_type is not None:
            body["content_type"] = content_type
        if mtime is not None:
            body["mtime"] = mtime.isoformat() if isinstance(mtime, datetime) else mtime
        return cast(dict[str, Any], await self._request("POST", "/api/v1/ingest/text", json=body))

    async def ingest_file(self, path: str, collection: str) -> dict[str, Any]:
        """POST /api/v1/ingest/file — one server-side file."""
        return cast(
            dict[str, Any],
            await self._request(
                "POST", "/api/v1/ingest/file", json={"path": path, "collection": collection}
            ),
        )

    async def ingest_directory(self, directory: str, collection: str) -> dict[str, Any]:
        """POST /api/v1/ingest/dir — recursive directory ingest."""
        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/api/v1/ingest/dir",
                json={"directory": directory, "collection": collection},
            ),
        )

    async def reindex(self, path: str, collection: str) -> dict[str, Any]:
        """POST /api/v1/ingest/reindex — atomic reindex of one source."""
        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/api/v1/ingest/reindex",
                json={"path": path, "collection": collection},
            ),
        )

    async def delete_source(self, path: str, collection: str) -> dict[str, Any]:
        """POST /api/v1/ingest/delete — cascade-delete one source."""
        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/api/v1/ingest/delete",
                json={"path": path, "collection": collection},
            ),
        )

    async def sync(self, directory: str, collection: str) -> dict[str, Any]:
        """POST /api/v1/ingest/sync — incremental new/modified/deleted."""
        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/api/v1/ingest/sync",
                json={"directory": directory, "collection": collection},
            ),
        )

    # ------------------------------------------------------------------ #
    # collections + chunks surface
    # ------------------------------------------------------------------ #

    async def create_collection(
        self, collection: str, dimension: int | None = None
    ) -> dict[str, Any]:
        """POST /api/v1/collections — create at the resolved dimension
        (None = the server's provider dimension)."""
        body: dict[str, object] = {"collection": collection}
        if dimension is not None:
            body["dimension"] = dimension
        return cast(dict[str, Any], await self._request("POST", "/api/v1/collections", json=body))

    async def list_collections(self) -> dict[str, Any]:
        """GET /api/v1/collections."""
        return cast(dict[str, Any], await self._request("GET", "/api/v1/collections"))

    async def collection_stats(self, collection: str) -> dict[str, Any]:
        """GET /api/v1/collections/{collection} — CollectionStats shape."""
        path = "/api/v1/collections/" + urllib.parse.quote(collection, safe="")
        return cast(dict[str, Any], await self._request("GET", path))

    async def list_chunks(
        self,
        collection: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /api/v1/chunks — page of stored chunks in id order."""
        return cast(
            dict[str, Any],
            await self._request(
                "GET",
                "/api/v1/chunks",
                params={"collection": collection, "limit": limit, "offset": offset},
            ),
        )

    async def search_chunks(
        self,
        collection: str,
        query: str,
        *,
        top_k: int = 10,
        raw: bool = False,
        min_score: float | None = None,
        filters: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/chunks/search — FTS-only search within a collection."""
        body: dict[str, object] = {
            "collection": collection,
            "query": query,
            "top_k": top_k,
            "raw": raw,
        }
        if min_score is not None:
            body["min_score"] = min_score
        if filters:
            body["filters"] = filters
        return cast(dict[str, Any], await self._request("POST", "/api/v1/chunks/search", json=body))


# --------------------------------------------------------------------------- #
# request/response helpers
# --------------------------------------------------------------------------- #


def _search_body(
    *,
    query: str,
    collection: str,
    top_k: int,
    mode: str,
    weights: list[float] | None,
    retrieve_k: int,
    min_score: float | None,
    raw: bool,
    filters: dict[str, object] | None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "query": query,
        "collection": collection,
        "top_k": top_k,
        "mode": mode,
        "retrieve_k": retrieve_k,
        "raw": raw,
    }
    if weights is not None:
        body["weights"] = weights
    if min_score is not None:
        body["min_score"] = min_score
    if filters:
        body["filters"] = filters
    return body


def _handle(response: Any, *, path: str) -> Any:
    """Parse a response; raise ClientError on non-2xx. The server's
    structured error shape (`{"error": {"type", "message", "detail"}}`) is
    carried on the exception for callers that want it."""
    payload = None
    try:
        payload = response.json()
    except Exception:
        payload = None
    if response.status_code >= 400:
        message = _extract_message(payload, response.text)
        raise ClientError(message, status_code=response.status_code, detail=payload)
    return payload


def _extract_message(payload: Any, text: str) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
            return str(error)
        return str(payload)[:200]
    return text[:200]
