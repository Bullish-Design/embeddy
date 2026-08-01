"""HTTPProvider — OpenAI-compatible `/v1/embeddings` adapter (embeddy[client]
extra, LAZY import of httpx). CONCEPT §7 / plan §3:

  * Text-dense only in v1. Multimodal (Qwen3-VL) and learned-sparse (bge-m3)
    are local-provider only — an ImageInput raises ProviderInputError.
  * The request body is the OpenAI shape `{"input": [...], "model": ...}`,
    plus `{"instruction": <resolved string>}` via extra_body when the caller
    passes an instruction. A generic upstream that ignores `extra_body` will
    silently drop it — the provider DECLARES whether it honors instructions
    (`honors_instructions`); callers must not rely on prompt behavior for an
    upstream that does not.
  * Response validation: the returned embedding dimension MUST equal the
    resolved dimension (wrong-dimension response raises WrongDimensionError —
    the C2 class of silent no-op is impossible at the wire).
  * Retries: transport errors, timeouts and 5xx are retried with exponential
    backoff up to `max_retries`; 4xx raise immediately.
  * The provider L2-normalizes upstream vectors (rank-preserving) so the
    protocol invariant (unit-norm Vector) holds regardless of upstream
    conventions (OpenAI does not normalize; TEI/vLLM/Ollama do).

`build_embeddings_request` is the ONE request builder — the Phase-6 client
and server reuse it (one wire protocol, no parallel incompatible request
shapes — the C5 class of bug is structurally impossible).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from embeddy.errors import HTTPProviderError, ProviderInputError, WrongDimensionError
from embeddy.protocol.types import EmbedInput, ImageInput, Vector, normalize_l2
from embeddy.registry import ModelSpec, get_model, resolve_dimension

if TYPE_CHECKING:  # pragma: no cover - import-time type info only
    pass

_EMBEDDINGS_PATH = "/v1/embeddings"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_BATCH_SIZE = 64

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def build_embeddings_request(
    *,
    base_url: str,
    model: str,
    inputs: list[str],
    instruction: str | None = None,
    api_key: str | None = None,
) -> Any:  # httpx.Request; typed Any to keep httpx lazy at import time
    """The ONE OpenAI-compatible embeddings request shape (CONCEPT §7).

    `inputs` are already text (the provider rejects ImageInput before this);
    `instruction` is a resolved string appended as an extra body field.
    """
    import httpx

    body: dict[str, object] = {"input": inputs, "model": model}
    if instruction:
        body["instruction"] = instruction
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + _EMBEDDINGS_PATH
    return httpx.Request("POST", url, json=body, headers=headers)


class HTTPProvider:
    """OpenAI-compatible embeddings adapter. Conforms to the
    EmbeddingProvider protocol (dimension / context_length / model_name /
    async encode)."""

    def __init__(
        self,
        model_id: str,
        dimension: int | None = None,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        honors_instructions: bool = False,
        http_client: Any = None,
        spec: ModelSpec | None = None,
    ) -> None:
        """`http_client` injects an httpx.AsyncClient (tests: MockTransport);
        `spec` injects model facts directly (tests). Production path:
        model_id -> registry -> ModelSpec."""
        self._spec = spec if spec is not None else get_model(model_id)
        self.model_name = self._spec.id
        self.dimension = resolve_dimension(self._spec, dimension)
        self.context_length = self._spec.context_length
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._http_client = http_client

        # The provider DECLARES whether the upstream honors instructions
        # (CONCEPT §7: generic upstreams silently drop extra_body).
        self.honors_instructions = honors_instructions

    @property
    def supports_instructions(self) -> bool:
        """Alias for `honors_instructions` (the protocol-level vocabulary)."""
        return self.honors_instructions

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #

    async def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[Vector]:
        if not inputs:
            return []
        texts = [self._as_text(item) for item in inputs]
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._encode_batch(batch, instruction))
        return vectors

    @staticmethod
    def _as_text(item: EmbedInput) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, ImageInput):
            raise ProviderInputError(
                "HTTPProvider is text-dense only in v1 (CONCEPT §7): "
                "multimodal inputs are local-provider only"
            )
        raise ProviderInputError(f"unsupported input type: {type(item).__name__!r}")

    async def _encode_batch(self, texts: list[str], instruction: str | None) -> list[Vector]:
        import httpx

        client: Any = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            owns_client = True
        else:
            owns_client = False

        request = build_embeddings_request(
            base_url=self._base_url,
            model=self.model_name,
            inputs=texts,
            instruction=instruction,
            api_key=self._api_key,
        )
        try:
            response = await self._send_with_retries(client, request)
        except BaseException:
            if owns_client:
                await client.aclose()
            raise
        return self._parse_embeddings(response, expected=len(texts))

    async def _send_with_retries(self, client: Any, request: Any) -> Any:
        """Retry transport errors / timeouts / 5xx with exponential backoff.
        A 4xx raises immediately; the caller sees HTTPProviderError after
        retries are exhausted."""
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
                raise HTTPProviderError(
                    f"HTTP embeddings request failed after {self._max_retries + 1} attempts: {exc}"
                ) from exc
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                last_error = HTTPProviderError(
                    f"upstream returned status {response.status_code}; retrying"
                )
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if response.status_code >= 400:
                detail = self._error_detail(response)
                raise HTTPProviderError(
                    f"embeddings request failed with status {response.status_code}: {detail}"
                )
            return response
        assert last_error is not None  # loop always returns or raises
        raise HTTPProviderError(f"embeddings request failed: {last_error}")

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:
            return response.text[:200]
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message", error))
            return str(payload)[:200]
        return str(payload)[:200]

    def _parse_embeddings(self, response: Any, *, expected: int) -> list[Vector]:
        try:
            payload = response.json()
        except Exception as exc:
            raise HTTPProviderError(f"non-JSON embeddings response: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPProviderError(
                f"unexpected embeddings response shape: {type(payload).__name__}"
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise HTTPProviderError(f"embeddings response has no `data` list: {payload!r:.120}")
        if len(data) != expected:
            raise HTTPProviderError(
                f"embeddings response has {len(data)} items for {expected} inputs"
            )
        vectors: list[Vector] = []
        for item in data:
            if not isinstance(item, dict):
                raise HTTPProviderError(f"embedding item is not an object: {item!r:.120}")
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise HTTPProviderError(f"embedding item has no list `embedding`: {item!r:.120}")
            arr = np.asarray(embedding, dtype=np.float32)
            if arr.shape != (self.dimension,):
                raise WrongDimensionError(
                    f"upstream returned dimension {arr.shape[0]} for resolved dimension "
                    f"{self.dimension} (model {self.model_name}); a wrong-dimension "
                    "response is a hard error, not a silent no-op"
                )
            vectors.append(normalize_l2(arr))
        return vectors
