"""Reranker implementations — local CrossEncoder + remote TEI/Jina endpoint.

CONCEPT §5.3 / plan §3: the reranker is an optional post-fusion stage. The
remote rerank endpoint uses the TEI/Jina shape (`POST {path}` with
`{"query": ..., "texts": [...], "top_n": N}`), NOT a non-existent OpenAI
`/v1/rerank` standard (CONCEPT §7).

Both implementations conform to `RerankerProvider` (protocol/rerank.py):
caller resolves role -> instruction and passes the resolved string; local
CrossEncoders ignore it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from embeddy.errors import HTTPProviderError, RerankError
from embeddy.protocol.rerank import RerankHit


class _CrossEncoder(Protocol):
    """The slice of sentence-transformers' CrossEncoder the local reranker
    consumes (duck-typed, never imported at runtime — the docling pattern)."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray: ...


if TYPE_CHECKING:  # pragma: no cover - import-time type info only
    pass

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 2
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_DEFAULT_RERANK_PATH = "/rerank"


class CrossEncoderReranker:
    """Local CrossEncoder reranker (embeddy[local] extra, LAZY import).

    Scores every (query, document) pair and returns hits ranked by
    descending score. `instruction` is accepted for protocol compatibility
    and ignored — CrossEncoders are not prompt-configured.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: _CrossEncoder | None = cast(_CrossEncoder | None, model)

    def load(self) -> _CrossEncoder:
        """Lazy-load the CrossEncoder. Raises RerankError when the `local`
        extra is not installed."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # embeddy[local] not installed
                raise RerankError(
                    "CrossEncoderReranker requires the `embeddy[local]` extra "
                    "(sentence-transformers)"
                ) from exc
            self._model = cast(_CrossEncoder, CrossEncoder(self.model_name, device=self._device))
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[RerankHit]:
        del instruction  # CrossEncoders are not prompt-configured
        if not documents:
            return []
        model = self.load()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs, batch_size=self._batch_size, convert_to_numpy=True)
        hits = [RerankHit(index=i, score=float(score)) for i, score in enumerate(scores)]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        if top_k is not None:
            hits = hits[:top_k]
        return hits


class HTTPReranker:
    """Remote reranker over the TEI/Jina wire shape (embeddy[client] extra,
    LAZY import of httpx). `{base_url}{rerank_path}` defaults to `/rerank`
    (TEI native); the request body is `{"query", "texts", "top_n"}` and the
    response is parsed from `results[].score` (TEI) or
    `results[].relevance_score` (Jina)."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str,
        api_key: str | None = None,
        rerank_path: str = _DEFAULT_RERANK_PATH,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        http_client: Any = None,
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url.rstrip("/") + rerank_path
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._http_client = http_client

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[RerankHit]:
        del instruction  # remote rerank endpoints do not take instructions
        if not documents:
            return []
        import httpx

        client: Any = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            owns_client = True
        else:
            owns_client = False

        body: dict[str, object] = {"query": query, "texts": documents}
        if top_k is not None:
            body["top_n"] = top_k
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = httpx.Request("POST", self._base_url, json=body, headers=headers)
        try:
            response = await self._send_with_retries(client, request)
        except BaseException:
            if owns_client:
                await client.aclose()
            raise
        return self._parse_response(response, expected=len(documents))

    async def _send_with_retries(self, client: Any, request: Any) -> Any:
        import httpx

        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.send(request)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise RerankError(
                    f"rerank request failed after {self._max_retries + 1} attempts: {exc}"
                ) from exc
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                last_error = HTTPProviderError(
                    f"upstream returned status {response.status_code}; retrying"
                )
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if response.status_code >= 400:
                raise RerankError(
                    f"rerank request failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return response
        assert last_error is not None  # loop always returns or raises
        raise RerankError(f"rerank request failed: {last_error}")

    def _parse_response(self, response: Any, *, expected: int) -> list[RerankHit]:
        try:
            payload = response.json()
        except Exception as exc:
            raise RerankError(f"non-JSON rerank response: {exc}") from exc
        if not isinstance(payload, dict):
            raise RerankError(f"unexpected rerank response shape: {type(payload).__name__}")
        results = payload.get("results")
        if not isinstance(results, list):
            raise RerankError(f"rerank response has no `results` list: {payload!r:.120}")
        hits: list[RerankHit] = []
        for item in results:
            if not isinstance(item, dict):
                raise RerankError(f"rerank result is not an object: {item!r:.120}")
            index = item.get("index")
            score = item.get("score", item.get("relevance_score"))
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                raise RerankError(f"malformed rerank result: {item!r:.120}")
            if index < 0 or index >= expected:
                raise RerankError(
                    f"rerank result index {index} out of range for {expected} documents"
                )
            hits.append(RerankHit(index=index, score=float(score)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits
