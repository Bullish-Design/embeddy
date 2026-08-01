"""embeddy/server.py — FastAPI factory + bare app (plan §8 / CONCEPT §5.7).

The server is a THIN ADAPTER that owns lifecycle (CONCEPT §3.7):

  * `create_app(store=..., provider=..., reranker=..., settings=...)` is the
    dependency-injection seam — ASGI tests inject mocks; the lifespan opens
    the store (SqliteStore.open), builds + loads the provider from settings,
    and closes everything on shutdown.
  * The module-level `app = create_app()` is built from settings so
    `uvicorn embeddy.server:app` works BARE. The bare app opens
    `settings.store_path` and builds a LOCAL provider from the embedder
    settings; when the provider cannot load (e.g. `embeddy[local]` missing)
    the process stays up and reports NOT-ready — that is the honest health
    contract.
  * Health: `GET /health/live` = process up (always 200 while the process
    serves); `GET /health/ready` = provider LOADED and store OPEN (503 +
    `ready: false` otherwise). A startup failure is recorded, not re-raised:
    the process keeps serving and reports the reason.
  * Role resolution (H2 discipline, CONCEPT §5.1): the server is a CALLER.
    `/v1/embeddings` and `/api/v1/search` resolve the QUERY role via
    `resolve_instruction(provider.model_name, embedder.prompt_role)` and
    pass the RESOLVED string to encode/search_hybrid. The DOCUMENT role is
    used only by the ingest routes (IngestPipeline resolves it). Providers
    never see a role.
  * One wire protocol: `/v1/embeddings` is OpenAI-compatible; `/v1/rerank`
    is the TEI/Jina shape (`{"query","texts","top_n"}`) — there is no
    OpenAI /v1/rerank standard (CONCEPT §7).

ERROR MAP (every mapping used here, documented per plan §8):
  * FastAPI RequestValidationError (bad body/query)  -> 422 structured shape
  * pydantic ValidationError raised inside a handler -> 400 structured
  * HTTPException (raised by routes)                  -> its status, structured
    - unknown / invalid collection                    -> 404
    - unknown chunk id in /api/v1/similar             -> 404
    - unknown model on /v1/embeddings (model mismatch)-> 400
    - weights / retrieve_k / min_score violations     -> 400
    - body > max_body_bytes (middleware)              -> 413
    - inputs > max_embed_inputs (/v1/embeddings)      -> 413
    - top_k / retrieve_k / top_n / page-limit
      > max_top_k                                     -> 413
    - backend missing a store extra (create_collection
      / get_chunk / list_chunks / list_collections)   -> 501
    - not ready, or no reranker configured            -> 503
  * EmbeddyError                                      -> 500 "embeddy_error"
  * anything else (RegistryError, StoreError, ...)    -> 500 "internal_error"

The size limits are OOM/crash-prevention ONLY, never security (CONCEPT
§9.7: auth/SSRF/path-traversal are out of scope on the trusted tailnet).

LAZY-IMPORT NOTE: fastapi/starlette are `embeddy[server]` extras and ARE
imported at module level here — this module IS the server extra's entry
point. `embeddy/__init__.py` never imports this module, so zero-extras
`import embeddy` stays clean.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.cors import CORSMiddleware

from embeddy.config import (
    EmbedderSettings,
    PipelineSettings,
    ServerSettings,
    load_pipeline_settings,
    load_server_settings,
)
from embeddy.errors import EmbeddyError
from embeddy.index.base import Searchable, SearchFilters
from embeddy.index.sqlite import CollectionInfo, SqliteStore, StoreError
from embeddy.pipeline import IngestPipeline, IngestStats, SourceError
from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.protocol.rerank import RerankerProvider
from embeddy.protocol.types import (
    EmbedInput,
    ScoredDocument,
    SearchResult,
    StoredChunk,
    Vector,
)
from embeddy.registry import resolve_instruction
from embeddy.search import search_hybrid

__all__ = ["app", "create_app"]


# --------------------------------------------------------------------------- #
# request/response models (module-level: FastAPI 0.141.1 treats locally
# defined pydantic models as query params — verified 2026-08-01)
# --------------------------------------------------------------------------- #


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible `/v1/embeddings` request (CONCEPT §7).

    `input` is a single string or a list of strings (OpenAI shape).
    `model` must match the server's provider when given (mismatch -> 400).
    `instruction` is an optional ALREADY-RESOLVED string that overrides the
    server's query-role resolution for callers that resolved it themselves.
    """

    input: str | list[str]
    model: str | None = None
    instruction: str | None = None


class RerankRequest(BaseModel):
    """TEI/Jina rerank shape (CONCEPT §7): `{"query","texts","top_n"}`."""

    query: str
    texts: list[str]
    top_n: int | None = Field(default=None, ge=1)


class SearchFiltersModel(BaseModel):
    """Wire shape of `SearchFilters` (index/base.py) — compiled to SQL
    pre-filters by the store, never a post-filter over-fetch."""

    content_types: tuple[str, ...] = ()
    source_path_prefix: str | None = None
    chunk_types: tuple[str, ...] = ()
    metadata_match: dict[str, str] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """The hybrid-search request (`/api/v1/search`). `mode` is rrf (default)
    or weighted; `weights` are the vector/FTS blend (len must be 2);
    `retrieve_k` is the per-leg candidate fetch (must be >= top_k)."""

    query: str
    collection: str
    top_k: int = Field(default=10, ge=1)
    mode: Literal["rrf", "weighted"] = "rrf"
    weights: list[float] | None = None
    retrieve_k: int = Field(default=50, ge=1)
    min_score: float | None = None
    raw: bool = False
    filters: SearchFiltersModel = Field(default_factory=SearchFiltersModel)


class SearchRerankRequest(SearchRequest):
    """The search-then-rerank request (`/api/v1/rerank`): every SearchRequest
    field plus `rerank_top_k` for the rerank stage (defaults to top_k)."""

    rerank_top_k: int | None = Field(default=None, ge=1)


class SimilarRequest(BaseModel):
    """The find-similar request (`/api/v1/similar`): re-embed the chunk at
    `chunk_id` (query role) and return its nearest neighbours."""

    chunk_id: str
    collection: str
    top_k: int = Field(default=10, ge=1)
    filters: SearchFiltersModel = Field(default_factory=SearchFiltersModel)


class IngestTextRequest(BaseModel):
    """Raw-text ingest (no file I/O). `content_type` None = auto-detect."""

    text: str
    collection: str
    path: str = "<memory>"
    content_type: str | None = None
    mtime: datetime | None = None


class IngestFileRequest(BaseModel):
    """One file ingest (server-side filesystem path — the trusted-tailnet
    server does no path-traversal protection, CONCEPT §9.7)."""

    path: str
    collection: str


class IngestDirectoryRequest(BaseModel):
    """Recursive directory ingest via the bounded pipeline pool."""

    directory: str
    collection: str


class IngestReindexRequest(BaseModel):
    """Atomic reindex of one existing source (never delete-then-reingest)."""

    path: str
    collection: str


class IngestDeleteRequest(BaseModel):
    """Delete one source (cascade: chunks, vectors, FTS rows)."""

    path: str
    collection: str


class IngestSyncRequest(BaseModel):
    """Incremental sync: new / modified / deleted diff against the store."""

    directory: str
    collection: str


class CreateCollectionRequest(BaseModel):
    """Create a collection at its RESOLVED vector dimension (None = the
    provider's dimension)."""

    collection: str
    dimension: int | None = Field(default=None, ge=1)


class ChunkSearchRequest(BaseModel):
    """FTS-only search within a collection (the cheap text path; the hybrid
    path is /api/v1/search)."""

    collection: str
    query: str
    top_k: int = Field(default=10, ge=1)
    raw: bool = False
    min_score: float | None = None
    filters: SearchFiltersModel = Field(default_factory=SearchFiltersModel)


# --------------------------------------------------------------------------- #
# app state
# --------------------------------------------------------------------------- #


class _ServerState:
    """The app's dependency + readiness state.

    Populated by create_app (injected store/provider/reranker) and the
    lifespan (built-from-settings store/provider). `ready` is True only
    after the lifespan completed startup successfully — the honest health
    contract (provider loaded AND store open).
    """

    def __init__(
        self,
        settings: ServerSettings,
        embedder: EmbedderSettings,
        pipeline: PipelineSettings,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.pipeline = pipeline
        self.store: Searchable | None = None
        self.provider: EmbeddingProvider | None = None
        self.reranker: RerankerProvider | None = None
        self.ready: bool = False
        self.ready_reason: str | None = None


def _state_from(request: Request) -> _ServerState:
    return cast(_ServerState, request.app.state.server)


# --------------------------------------------------------------------------- #
# lifecycle + provider building
# --------------------------------------------------------------------------- #


def _build_provider(embedder: EmbedderSettings) -> EmbeddingProvider:
    """Build the provider from the embedder settings (config-time errors:
    unknown model / non-MRL wrong dimension raise HERE, before the lifespan
    loads anything — the C2 class of silent no-op is impossible). The bare
    server builds the LOCAL backend; HTTP deployments construct the app
    programmatically via `create_app(provider=build_provider(..., backend=
    "http", base_url=...))`."""
    from embeddy.providers.factory import build_provider

    return build_provider(embedder.model, embedder.embedding_dimension, backend="local")


def _load_provider(provider: EmbeddingProvider) -> None:
    """Load the provider when it has a `load()` (LocalProvider). Providers
    without one (FakeProvider, HTTPProvider) are ready by construction."""
    load = getattr(provider, "load", None)
    if callable(load):
        load()


async def _aclose(obj: object) -> None:
    """Close a provider/reranker that has a close() (sync or async)."""
    close = getattr(obj, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


# --------------------------------------------------------------------------- #
# the factory
# --------------------------------------------------------------------------- #


def create_app(
    *,
    store: Searchable | None = None,
    provider: EmbeddingProvider | None = None,
    reranker: RerankerProvider | None = None,
    settings: ServerSettings | None = None,
    embedder: EmbedderSettings | None = None,
    pipeline: PipelineSettings | None = None,
) -> FastAPI:
    """Build the API app. The DI seam: inject store/provider/reranker for
    tests (the lifespan then only records readiness and never closes the
    injected deps); leave them None for the settings-driven bare server.

    `settings`/`embedder`/`pipeline` default to the loaded settings
    (CLI > file > env > defaults — config.py).
    """
    server_settings = settings if settings is not None else load_server_settings()
    embedder_settings = embedder if embedder is not None else EmbedderSettings()
    pipeline_settings = pipeline if pipeline is not None else load_pipeline_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state: _ServerState = app.state.server
        state.ready = False
        state.ready_reason = None
        opened_store: SqliteStore | None = None
        try:
            if state.store is None:
                opened_store = await SqliteStore.open(server_settings.store_path)
                state.store = opened_store
            provider = state.provider
            if provider is None:
                provider = _build_provider(embedder_settings)
                state.provider = provider
            _load_provider(provider)
            state.ready = True
        except Exception as exc:
            # honest readiness: keep serving, report why we are not ready.
            state.ready_reason = f"{type(exc).__name__}: {exc}"
            if opened_store is not None:
                await opened_store.close()
                state.store = None
        try:
            yield
        finally:
            if opened_store is not None:
                await opened_store.close()
                state.store = None
            await _aclose(state.provider)
            await _aclose(state.reranker)

    app = FastAPI(
        title="embeddy",
        description="Typed, metric-honest embedding, storage, search, and serving (plan §8).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.server = _ServerState(server_settings, embedder_settings, pipeline_settings)
    app.state.server.store = store
    app.state.server.provider = provider
    app.state.server.reranker = reranker

    _register_middleware_and_handlers(app, server_settings)
    _register_routes(app)
    return app


def _register_middleware_and_handlers(app: FastAPI, settings: ServerSettings) -> None:
    """Body-size middleware registered FIRST, CORS LAST — `add_middleware`
    PREPENDS and the first user middleware is OUTERMOST (verified 2026-08-01
    on starlette 1.3.1), so CORS sits outside the body-limit middleware and
    adds its headers to error responses too. Exception handlers map every
    failure to the structured error shape (module docstring table)."""

    @app.middleware("http")
    async def _limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        # OOM/crash-prevention ONLY (CONCEPT §9.7). Header-based: a body
        # without Content-Length (chunked) is not size-limited in v1.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = 0
            if size > settings.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content=_error_payload(
                        "payload_too_large",
                        f"request body of {size} bytes exceeds max_body_bytes "
                        f"({settings.max_body_bytes})",
                    ),
                )
        return await call_next(request)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(RequestValidationError)
    async def _on_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload("validation_error", "request validation failed", exc.errors()),
        )

    @app.exception_handler(ValidationError)
    async def _on_pydantic_validation(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400, content=_error_payload("validation_error", str(exc), exc.errors())
        )

    @app.exception_handler(HTTPException)
    async def _on_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(_error_type_for(exc.status_code), str(exc.detail)),
        )

    @app.exception_handler(EmbeddyError)
    async def _on_embeddy_error(request: Request, exc: EmbeddyError) -> JSONResponse:
        return JSONResponse(status_code=500, content=_error_payload("embeddy_error", str(exc)))

    @app.exception_handler(Exception)
    async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Everything else -> structured 500 (RegistryError, StoreError, ...).
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", f"{type(exc).__name__}: {exc}"),
        )


def _register_routes(app: FastAPI) -> None:
    """All routes. Handlers read the DI/readiness state via
    `_state_from(request)` and the guards below."""

    # ------------------------------------------------------------------ #
    # health — honest readiness
    # ------------------------------------------------------------------ #

    @app.get("/health/live")
    async def health_live(request: Request) -> dict[str, object]:
        return {"status": "live", "ready": _state_from(request).ready}

    @app.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        state = _state_from(request)
        if state.ready:
            return JSONResponse({"status": "ready", "ready": True})
        return JSONResponse(
            {"status": "not_ready", "ready": False, "reason": state.ready_reason},
            status_code=503,
        )

    # ------------------------------------------------------------------ #
    # OpenAI-compatible surface
    # ------------------------------------------------------------------ #

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingsRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        provider = _provider(state)
        if req.model is not None and req.model != provider.model_name:
            raise HTTPException(
                status_code=400,
                detail=f"unknown model {req.model!r}; this server serves {provider.model_name!r}",
            )
        inputs = [req.input] if isinstance(req.input, str) else list(req.input)
        if len(inputs) > state.settings.max_embed_inputs:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"too many inputs: {len(inputs)} > max_embed_inputs "
                    f"({state.settings.max_embed_inputs})"
                ),
            )
        instruction = req.instruction if req.instruction is not None else _query_instruction(state)
        vectors = await _encode_texts(provider, inputs, instruction)
        # OpenAI-compatible response; `usage` is deliberately omitted (the
        # server does not count tokens — that is chonkai's business).
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": v.tolist()}
                for i, v in enumerate(vectors)
            ],
            "model": provider.model_name,
        }

    @app.post("/v1/rerank")
    async def rerank(req: RerankRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        reranker = _reranker(state)
        if req.top_n is not None:
            _check_top_k(state, req.top_n)
        hits = await reranker.rerank(req.query, req.texts, top_k=req.top_n)
        return {"results": [{"index": h.index, "score": h.score} for h in hits]}

    # ------------------------------------------------------------------ #
    # search surface
    # ------------------------------------------------------------------ #

    @app.post("/api/v1/search")
    async def search(req: SearchRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        _validate_search_request(state, req.top_k, req.retrieve_k, req.weights, req.min_score)
        provider = _provider(state)
        instruction = _query_instruction(state)
        vectors = await _encode_texts(provider, [req.query], instruction)
        if not vectors:
            raise HTTPException(status_code=500, detail="provider returned no vector for the query")
        result = await search_hybrid(
            _store(state),
            collection=req.collection,
            query_text=req.query,
            query_vector=vectors[0],
            filters=_to_filters(req.filters),
            top_k=req.top_k,
            mode=req.mode,
            weights=req.weights,
            retrieve_k=req.retrieve_k,
            min_score=req.min_score,
            raw=req.raw,
            instruction=instruction,
        )
        return _search_result_payload(result)

    @app.post("/api/v1/similar")
    async def similar(req: SimilarRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        _check_top_k(state, req.top_k)
        store = _store(state)
        get_chunk = _store_extra(store, "get_chunk", "get_chunk (server /api/v1/similar)")
        chunk = await cast(Callable[..., Awaitable[StoredChunk | None]], get_chunk)(
            req.collection, req.chunk_id
        )
        if chunk is None:
            raise HTTPException(
                status_code=404,
                detail=f"chunk {req.chunk_id!r} not found in collection {req.collection!r}",
            )
        provider = _provider(state)
        vectors = await _encode_texts(provider, [chunk.content], _query_instruction(state))
        if not vectors:
            raise HTTPException(status_code=500, detail="provider returned no vector for the chunk")
        hits = await store.search_vector(
            req.collection, vectors[0], _to_filters(req.filters), req.top_k
        )
        # "similar" = OTHER chunks: exclude the query chunk itself.
        hits = [h for h in hits if h.chunk_id != req.chunk_id]
        return {
            "results": [_scored_payload(h) for h in hits],
            "collection": req.collection,
            "chunk_id": req.chunk_id,
        }

    @app.post("/api/v1/rerank")
    async def search_rerank(req: SearchRerankRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        _reranker(state)  # 503 when no reranker is configured
        await _require_collection(state, req.collection)
        _validate_search_request(state, req.top_k, req.retrieve_k, req.weights, req.min_score)
        if req.rerank_top_k is not None:
            _check_top_k(state, req.rerank_top_k)
        provider = _provider(state)
        instruction = _query_instruction(state)
        vectors = await _encode_texts(provider, [req.query], instruction)
        if not vectors:
            raise HTTPException(status_code=500, detail="provider returned no vector for the query")
        result = await search_hybrid(
            _store(state),
            collection=req.collection,
            query_text=req.query,
            query_vector=vectors[0],
            filters=_to_filters(req.filters),
            top_k=req.top_k,
            mode=req.mode,
            weights=req.weights,
            retrieve_k=req.retrieve_k,
            min_score=req.min_score,
            raw=req.raw,
            reranker=_reranker(state),
            rerank_top_k=req.rerank_top_k,
            instruction=instruction,
        )
        return _search_result_payload(result)

    # ------------------------------------------------------------------ #
    # ingest surface (the pipeline owns the DOCUMENT role resolution)
    # ------------------------------------------------------------------ #

    @app.post("/api/v1/ingest/text")
    async def ingest_text(req: IngestTextRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        stats = await _pipeline(state).ingest_text(
            req.text,
            collection=req.collection,
            path=req.path,
            content_type=req.content_type,
            mtime=req.mtime,
        )
        return _stats_payload(stats)

    @app.post("/api/v1/ingest/file")
    async def ingest_file(req: IngestFileRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        stats = await _pipeline(state).ingest_file(req.path, collection=req.collection)
        return _stats_payload(stats)

    @app.post("/api/v1/ingest/dir")
    async def ingest_directory(req: IngestDirectoryRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        stats = await _pipeline(state).ingest_directory(req.directory, collection=req.collection)
        return _stats_payload(stats)

    @app.post("/api/v1/ingest/reindex")
    async def ingest_reindex(req: IngestReindexRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        stats = await _pipeline(state).reindex(req.path, collection=req.collection)
        return _stats_payload(stats)

    @app.post("/api/v1/ingest/delete")
    async def ingest_delete(req: IngestDeleteRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        deleted = await _pipeline(state).delete_source(req.collection, req.path)
        return {"deleted": deleted}

    @app.post("/api/v1/ingest/sync")
    async def ingest_sync(req: IngestSyncRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        stats = await _pipeline(state).sync(req.directory, collection=req.collection)
        return _stats_payload(stats)

    # ------------------------------------------------------------------ #
    # collections surface (the server owns collection lifecycle)
    # ------------------------------------------------------------------ #

    @app.post("/api/v1/collections")
    async def create_collection(
        req: CreateCollectionRequest, request: Request
    ) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        create = _store_extra(_store(state), "create_collection", "create_collection")
        dimension = req.dimension if req.dimension is not None else _provider(state).dimension
        try:
            await cast(Callable[..., Awaitable[None]], create)(req.collection, dimension)
        except StoreError as exc:
            # caller errors only (bad id / bad dimension / exists with a
            # different dimension) — mapped to 400, documented.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"collection": req.collection, "dimension": dimension}

    @app.get("/api/v1/collections")
    async def list_collections(request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        list_all = _store_extra(_store(state), "list_collections", "list_collections")
        infos = await cast(Callable[..., Awaitable[list[CollectionInfo]]], list_all)()
        return {
            "collections": [
                {"collection": info.collection_id, "dimension": info.vector_dimension}
                for info in infos
            ]
        }

    @app.get("/api/v1/collections/{collection}")
    async def collection_stats(collection: str, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, collection)
        stats = await _store(state).stats(collection)
        return {
            "collection_id": stats.collection_id,
            "chunk_count": stats.chunk_count,
            "source_count": stats.source_count,
            "vector_dimension": stats.vector_dimension,
            "size_bytes": stats.size_bytes,
        }

    # ------------------------------------------------------------------ #
    # chunks surface
    # ------------------------------------------------------------------ #

    @app.get("/api/v1/chunks")
    async def list_chunks(
        request: Request,
        collection: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, collection)
        if limit < 1:
            raise HTTPException(status_code=400, detail=f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise HTTPException(status_code=400, detail=f"offset must be >= 0, got {offset}")
        _check_top_k(state, limit)  # the page bound is a size guard too
        list_chunks_extra = _store_extra(_store(state), "list_chunks", "list_chunks")
        chunks = await cast(Callable[..., Awaitable[list[StoredChunk]]], list_chunks_extra)(
            collection, limit=limit, offset=offset
        )
        return {
            "collection": collection,
            "chunks": [_chunk_payload(c) for c in chunks],
            "limit": limit,
            "offset": offset,
        }

    @app.post("/api/v1/chunks/search")
    async def search_chunks(req: ChunkSearchRequest, request: Request) -> dict[str, object]:
        state = _state_from(request)
        _require_ready(state)
        await _require_collection(state, req.collection)
        _check_top_k(state, req.top_k)
        if req.min_score is not None and not math.isfinite(req.min_score):
            raise HTTPException(status_code=400, detail="min_score must be finite")
        hits = await _store(state).search_fts(
            req.collection,
            req.query,
            _to_filters(req.filters),
            req.top_k,
            min_score=req.min_score,
            raw=req.raw,
        )
        return {
            "collection": req.collection,
            "query": req.query,
            "results": [_scored_payload(h) for h in hits],
        }


# --------------------------------------------------------------------------- #
# guards + helpers
# --------------------------------------------------------------------------- #


def _require_ready(state: _ServerState) -> None:
    """Every non-health route requires the completed lifespan startup."""
    if not state.ready:
        reason = state.ready_reason or "server is starting"
        raise HTTPException(status_code=503, detail=f"server not ready: {reason}")


def _provider(state: _ServerState) -> EmbeddingProvider:
    provider = state.provider
    if provider is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return provider


def _store(state: _ServerState) -> Searchable:
    store = state.store
    if store is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return store


def _reranker(state: _ServerState) -> RerankerProvider:
    reranker = state.reranker
    if reranker is None:
        raise HTTPException(status_code=503, detail="no reranker configured on this server")
    return reranker


def _store_extra(store: Searchable, name: str, description: str) -> Callable[..., object]:
    """Resolve a SqliteStore extra (create_collection / get_chunk /
    list_chunks / list_collections) off a Searchable backend. The extras are
    sqlite-only in v1 (like count_fts); a backend without one gets 501."""
    extra = getattr(store, name, None)
    if not callable(extra):
        raise HTTPException(status_code=501, detail=f"store backend does not support {description}")
    return extra


async def _require_collection(state: _ServerState, collection: str) -> None:
    """Unknown / invalid collection -> 404 (the server owns collection
    lifecycle; every collection-scoped route checks existence first)."""
    try:
        await _store(state).stats(collection)
    except StoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _check_top_k(state: _ServerState, value: int) -> None:
    """The retained operational guard (CONCEPT §9.7): OOM/crash-prevention,
    not security. Enforced on top_k / retrieve_k / top_n / page limits."""
    limit = state.settings.max_top_k
    if value > limit:
        raise HTTPException(status_code=413, detail=f"top_k {value} exceeds max_top_k {limit}")


def _validate_search_request(
    state: _ServerState,
    top_k: int,
    retrieve_k: int,
    weights: list[float] | None,
    min_score: float | None,
) -> None:
    """Caller-error validation shared by /api/v1/search and /api/v1/rerank
    (the search_hybrid ValueError class is a caller bug -> 400, not 500)."""
    _check_top_k(state, top_k)
    _check_top_k(state, retrieve_k)
    if retrieve_k < top_k:
        raise HTTPException(
            status_code=400, detail=f"retrieve_k ({retrieve_k}) must be >= top_k ({top_k})"
        )
    if weights is not None and len(weights) != 2:
        raise HTTPException(
            status_code=400, detail="weights must have exactly 2 entries (vector, fts)"
        )
    if min_score is not None and not math.isfinite(min_score):
        raise HTTPException(status_code=400, detail="min_score must be finite")


def _query_instruction(state: _ServerState) -> str:
    """The QUERY-role instruction (H2 discipline): the server is a CALLER
    and resolves role -> instruction via the registry — providers never see
    a role. `embedder.prompt_role` (default "query") selects the role."""
    provider = _provider(state)
    return resolve_instruction(provider.model_name, state.embedder.prompt_role or "query")


async def _encode_texts(
    provider: EmbeddingProvider, texts: list[str], instruction: str
) -> list[Vector]:
    """Encode TEXT inputs (EmbedInput = str | ImageInput; the server is
    text-only in v1 — multimodal is local-provider only, CONCEPT §7). The
    cast is the protocol's own text path (the pipeline does the same)."""
    return await provider.encode(cast(list[EmbedInput], texts), instruction=instruction)


def _pipeline(state: _ServerState) -> IngestPipeline:
    """One pipeline per ingest request (cheap — no model load; the provider
    is shared). The pipeline resolves the DOCUMENT role itself (its
    constructor), so ingest never touches the query role."""
    return IngestPipeline(
        store=_store(state),
        provider=_provider(state),
        concurrency=state.pipeline.concurrency,
    )


def _to_filters(m: SearchFiltersModel) -> SearchFilters:
    return SearchFilters(
        content_types=tuple(m.content_types),
        source_path_prefix=m.source_path_prefix,
        chunk_types=tuple(m.chunk_types),
        metadata_match=tuple(sorted(m.metadata_match.items())),
    )


def _error_type_for(status: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        422: "validation_error",
        501: "not_implemented",
        503: "service_unavailable",
    }.get(status, "http_error")


def _error_payload(error_type: str, message: str, detail: object = None) -> dict[str, object]:
    return {"error": {"type": error_type, "message": message, "detail": detail}}


def _search_result_payload(result: SearchResult) -> dict[str, object]:
    return {
        "results": [_scored_payload(h) for h in result.results],
        "total_results": result.total_results,
        "metric": result.metric.value,
        "mode": result.mode,
        "query": result.query,
        "collection": result.collection,
    }


def _scored_payload(hit: ScoredDocument) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "collection_id": hit.collection_id,
        "source_id": hit.source_id,
        "source_path": hit.source_path,
        "content": hit.content,
        "score": hit.score,
        "metric": hit.metric.value,
    }


def _chunk_payload(chunk: StoredChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.id,
        "collection_id": chunk.collection_id,
        "source_id": chunk.source_id,
        "content": chunk.content,
        "chunk_type": chunk.chunk_type,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "parent": chunk.parent,
        "granularity": chunk.granularity,
        "token_count": chunk.token_count,
    }


def _stats_payload(stats: IngestStats) -> dict[str, object]:
    return {
        "files_attempted": stats.files_attempted,
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "files_deleted": stats.files_deleted,
        "chunks_indexed": stats.chunks_indexed,
        "errors": [_source_error_payload(e) for e in stats.errors],
    }


def _source_error_payload(error: SourceError) -> dict[str, object]:
    return {
        "path": error.path,
        "phase": error.phase.value,
        "message": error.message,
        "error_type": error.error_type,
    }


# --------------------------------------------------------------------------- #
# the bare app (uvicorn embeddy.server:app)
# --------------------------------------------------------------------------- #

app = create_app()
