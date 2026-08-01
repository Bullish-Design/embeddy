"""Shared fixtures for the Phase-6 contract layer (tests/contract/) — plan
§8/§11: ASGI tests with INJECTED mocks (FakeProvider / in-memory store)
prove the dependency-injection seam, not the real model. The lifespan runs
via `app.router.lifespan_context` (httpx ASGITransport does NOT run it —
verified 2026-08-01); `raise_app_exceptions=False` preserves the server's
handler-produced error responses (starlette re-raises after sending so the
ASGI server can log — verified 2026-08-01).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest

from embeddy.config import EmbedderSettings, PipelineSettings, ServerSettings
from embeddy.errors import ProviderError
from embeddy.index.sqlite import SqliteStore
from embeddy.pipeline import IngestPipeline
from embeddy.protocol.rerank import RerankHit
from embeddy.protocol.types import CollectionStats
from embeddy.providers.fake import FakeProvider
from embeddy.server import create_app

# The registry model name both test providers pretend to be: it is
# registry-backed (role resolution works) AND its query vs document
# instructions differ (the H2 regression test needs that).
REGISTRY_MODEL = "Qwen/Qwen3-Embedding-0.6B"

CORPUS = """\
The acme authentication service issues short-lived JWTs for API access.
Tokens expire after 15 minutes and are refreshed via a rotating key.

The acme billing API records usage in 5-minute buckets and invoices monthly.
Refunds are processed within three business days of the request.
"""


class RecordingProvider(FakeProvider):
    """FakeProvider (deterministic dim-8 vectors) with a REGISTRY model name
    so role resolution works, that records every encode call's instruction
    string — the H2 role-discipline regression probe."""

    model_name = REGISTRY_MODEL
    dimension = 8

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[list[str], str | None]] = []

    async def encode(self, inputs: list[Any], instruction: str | None = None) -> list[Any]:
        self.calls.append((list(cast(list[str], inputs)), instruction))
        return await super().encode(inputs, instruction)


class FailingProvider(FakeProvider):
    """Injected state where the provider cannot LOAD — the honest
    not-ready path (provider LOADED and store open => ready)."""

    def load(self) -> None:
        raise ProviderError("cannot load test model")


class RaisingProvider(RecordingProvider):
    """A registry-backed provider whose encode ALWAYS raises ProviderError
    — the EmbeddyError -> 500 path."""

    async def encode(self, inputs: list[Any], instruction: str | None = None) -> list[Any]:
        del inputs, instruction
        raise ProviderError("encode exploded")


class FakeReranker:
    """Deterministic reranker: score = index, so hits sort DESCENDING by
    index — a rerank stage observably REVERSES the fused order (proves the
    stage ran, not a no-op)."""

    model_name = "fake/reranker"

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[RerankHit]:
        del query, instruction
        hits = [RerankHit(index=i, score=float(i)) for i in range(len(documents))]
        hits.sort(key=lambda h: h.score, reverse=True)
        if top_k is not None:
            hits = hits[:top_k]
        return hits


class StatsOnlyStore:
    """A Searchable-shaped stub with ONLY stats() — the 501 path for store
    extras (create_collection / get_chunk / list_chunks / list_collections)."""

    async def stats(self, collection: str) -> CollectionStats:
        return CollectionStats(
            collection_id=collection,
            chunk_count=0,
            source_count=0,
            vector_dimension=8,
        )


@pytest.fixture
async def store() -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(":memory:")
    yield s
    await s.close()


@pytest.fixture
def settings() -> ServerSettings:
    return ServerSettings(store_path=":memory:")


@pytest.fixture
def embedder() -> EmbedderSettings:
    return EmbedderSettings(prompt_role="query")


@pytest.fixture
def pipeline_settings() -> PipelineSettings:
    return PipelineSettings(concurrency=2)


@pytest.fixture
def provider() -> RecordingProvider:
    return RecordingProvider()


@pytest.fixture
async def app(
    store: SqliteStore,
    provider: RecordingProvider,
    settings: ServerSettings,
    embedder: EmbedderSettings,
    pipeline_settings: PipelineSettings,
) -> Any:
    application = create_app(
        store=store,
        provider=provider,
        settings=settings,
        embedder=embedder,
        pipeline=pipeline_settings,
    )
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def seed_corpus(
    store: SqliteStore, provider: RecordingProvider, collection: str = "acme"
) -> None:
    """Create the collection and ingest CORPUS through the REAL pipeline
    (document role resolution included) — the store the tests then query."""
    await store.create_collection(collection, provider.dimension)
    pipeline = IngestPipeline(
        store=store, provider=provider, concurrency=2
    )  # no instruction: resolves the DOCUMENT role itself
    stats = await pipeline.ingest_text(
        CORPUS, collection=collection, path="docs/guide.md", content_type="markdown"
    )
    assert stats.files_indexed == 1
    assert stats.errors == ()
