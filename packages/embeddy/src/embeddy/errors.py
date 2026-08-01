"""Embeddy error hierarchy — provider and config-time failures.

Phase 3 scope: provider construction and runtime errors are typed and
testable (wrong-dimension responses, retry exhaustion, multimodal over the
HTTP path). The server error map (ValidationError -> 400, other
EmbeddyError -> 500) consumes this in Phase 6 (plan §8). RegistryError stays
a ValueError subclass in registry.py (the resolve_dimension contract is
`raises ValueError` per CONCEPT §5.2); provider errors are RuntimeErrors so
they can never be confused with validation errors by a caller.
"""

from __future__ import annotations


class EmbeddyError(RuntimeError):
    """Base class for all embeddy runtime errors."""


class ProviderError(EmbeddyError):
    """A model provider failed to build or encode."""


class ProviderInputError(ProviderError):
    """An input cannot be handled by this provider (e.g. an ImageInput sent
    to the text-dense HTTP provider — multimodal is local-only in v1,
    CONCEPT §7)."""


class HTTPProviderError(ProviderError):
    """The HTTP provider failed after retries: non-2xx status that is not
    retryable, transport errors, or malformed responses."""


class WrongDimensionError(HTTPProviderError):
    """The upstream returned embeddings whose dimension differs from the
    resolved dimension. Fixes the C2 class of bug at the wire: a
    wrong-dimension response is a hard error, never a silent no-op."""


class RerankError(ProviderError):
    """A reranker (local or remote) failed."""


class ModelNotLoadedError(ProviderError):
    """A lazy provider method was called and the model/extra is not
    installed (e.g. `embeddy[local]` missing while using LocalProvider)."""
