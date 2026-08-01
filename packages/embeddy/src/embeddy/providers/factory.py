"""Provider factory — the config-time construction seam (plan §3).

Consumes the registry + resolved dimension: an unknown model or a
non-MRL-model-with-wrong-dimension raises HERE, at config time, before any
request or model load happens (the C2 class of silent no-op is impossible).
The factory is the single place where `backend` (local/http) maps to a
provider class; the Phase-6 server and CLI build providers through it.
"""

from __future__ import annotations

from typing import Any

from embeddy.errors import ProviderError
from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.registry import get_model, resolve_dimension

_LOCAL_BACKENDS = ("local", "sentence-transformers", "st")
_HTTP_BACKENDS = ("http", "openai", "remote")


def build_provider(
    model_id: str,
    dimension: int | None = None,
    *,
    backend: str = "local",
    **kwargs: Any,
) -> EmbeddingProvider:
    """Build a provider for a registry model at its RESOLVED dimension.

    `backend` is one of "local" (sentence-transformers) or "http"
    (OpenAI-compatible /v1/embeddings — `base_url` is required via kwargs).
    Unknown models and invalid dimension requests raise at config time.
    """
    spec = get_model(model_id)
    resolved = resolve_dimension(spec, dimension)

    if backend in _LOCAL_BACKENDS:
        from embeddy.providers.local import LocalProvider

        return LocalProvider(model_id, resolved, spec=spec, **kwargs)

    if backend in _HTTP_BACKENDS:
        from embeddy.providers.http import HTTPProvider

        return HTTPProvider(model_id, resolved, spec=spec, **kwargs)

    raise ProviderError(f"unknown provider backend {backend!r}; expected one of local/http")
