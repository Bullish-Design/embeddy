"""Providers package. Importable with zero extras (heavy providers lazy).

LocalProvider / CrossEncoderReranker import sentence-transformers only
inside `load()`; HTTPProvider / HTTPReranker import httpx only inside
`encode()`/`rerank()`. `build_provider` is the config-time construction
seam (unknown model / wrong dimension raise there).
"""

from embeddy.providers.factory import build_provider
from embeddy.providers.fake import FakeProvider
from embeddy.providers.http import HTTPProvider, build_embeddings_request
from embeddy.providers.local import LocalProvider
from embeddy.providers.rerank import CrossEncoderReranker, HTTPReranker

__all__ = [
    "CrossEncoderReranker",
    "FakeProvider",
    "HTTPProvider",
    "HTTPReranker",
    "LocalProvider",
    "build_embeddings_request",
    "build_provider",
]
