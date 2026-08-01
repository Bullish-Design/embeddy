"""Embeddy settings — pydantic-settings, one source of truth (CONCEPT §3.8,
§5.9). "Config = implementation": no field exists unless a code path reads
it. Phase 5 adds the `pipeline` section (the worker-pool bound); Phase 6 adds
`server` (the M5 product surface: size limits + CORS + the store the server
opens + the CLI's server base URL). `store` / `chunk` land with their
consuming code paths in Phases 6-7.

Precedence is CLI > file > env > defaults (plan §6 / CONCEPT §3.8):
`settings_customise_sources` puts the dotenv file BEFORE the environment
source (stock pydantic-settings is init > env > dotenv — verified 2026-08-01
on pydantic-settings 2.14.2); explicit init kwargs (CLI values) stay first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _PrecedenceSettings(BaseSettings):
    """Shared source order: CLI(init) > file(dotenv) > env > secrets >
    defaults. Verified 2026-08-01 on pydantic-settings 2.14.2 (stock order
    is init > env > dotenv, so the dotenv file must be moved BEFORE env).

    Every section class sets `dotenv_filtering="match_prefix"` so a SHARED
    dotenv file with multiple sections (EMBEDDY_EMBEDDER_*, EMBEDDY_PIPELINE_*,
    EMBEDDY_SERVER_*) loads cleanly: each class only pulls its own prefixed
    keys and never trips `extra="forbid"` on another section's keys
    (verified 2026-08-01 — without match_prefix the dotenv source passes ALL
    file keys into validation, and forbid rejects foreign-section keys).
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        del settings_cls
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)


class EmbedderSettings(_PrecedenceSettings):
    """The `embedder` section: model selection + MRL resolution + the
    semantic role used for instruction resolution (registry)."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDY_EMBEDDER_", extra="forbid", dotenv_filtering="match_prefix"
    )

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int | None = None
    prompt_role: str = "query"


DEFAULT_PIPELINE_CONCURRENCY = 4
"""Documented sane default for the pipeline worker pool (plan §7 / CONCEPT
§5.6: "the `concurrency` config finally works"). Small by design: the pool
bounds concurrently-in-flight FILES (including chunking, embedding and
store writes), so a modest bound overlaps the read/chunk/embed/write phases
without saturating a local model provider or accumulating corpus-sized
batches in memory. Read by a real code path — IngestPipeline's constructor
default imports this constant, so the config field is never dead."""


class PipelineSettings(_PrecedenceSettings):
    """The `pipeline` section (CONCEPT §5.9): the worker-pool bound.

    Config = implementation: every field here is read by a code path.
    `concurrency` is the bounded-worker-pool bound consumed by
    `embeddy.pipeline.IngestPipeline` (its constructor default imports
    `DEFAULT_PIPELINE_CONCURRENCY`; CLI/file/env overrides flow through
    `load_pipeline_settings` into the pipeline at construction time).
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDY_PIPELINE_", extra="forbid", dotenv_filtering="match_prefix"
    )

    concurrency: int = DEFAULT_PIPELINE_CONCURRENCY

    @field_validator("concurrency")
    @classmethod
    def _concurrency_must_be_positive(cls, value: int) -> int:
        """A worker pool bound < 1 can never make progress (fixes the H6
        class: a zero-bound pool would deadlock instead of draining)."""
        if value < 1:
            raise ValueError(f"concurrency must be >= 1, got {value}")
        return value


def load_settings(env_file: str | Path | None = None, **overrides: Any) -> EmbedderSettings:
    """Load settings with CLI > file > env > defaults precedence.

    `env_file` points at a dotenv file (KEY=VALUE, KEYs use the
    EMBEDDY_EMBEDDER_ prefix); `overrides` are CLI/programmatic values and
    win over everything. Reads a code path that exists: the factory and
    instruction resolution consume the resulting settings (config.py ->
    factory.build_provider -> registry.resolve_dimension / resolve_instruction).
    """
    return EmbedderSettings(_env_file=str(env_file) if env_file is not None else None, **overrides)


def load_pipeline_settings(
    env_file: str | Path | None = None, **overrides: Any
) -> PipelineSettings:
    """Load PIPELINE settings with CLI > file > env > defaults precedence
    (the same source order as `load_settings`, EMBEDDY_PIPELINE_ prefix).

    Reads a code path that exists: `concurrency` is the bounded-worker-pool
    bound handed to `IngestPipeline` (the Phase-6 server/CLI construct the
    pipeline from these settings; the constructor default mirrors the config
    default via `DEFAULT_PIPELINE_CONCURRENCY`).
    """
    return PipelineSettings(_env_file=str(env_file) if env_file is not None else None, **overrides)


# ---------------------------------------------------------------------------
# The `server` section (Phase 6 — plan §8 / CONCEPT §5.7, §9.7)
# ---------------------------------------------------------------------------

DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024
"""Request-body ceiling (10 MiB). The retained operational guard — an
OOM/crash-prevention control, NOT a security control (CONCEPT §9.7: auth /
SSRF / path-traversal are out of scope on the trusted tailnet). Read by the
server's body-limit middleware (embeddy/server.py)."""

DEFAULT_MAX_EMBED_INPUTS = 1024
"""Max inputs in one /v1/embeddings call. OOM/crash-prevention (a batch of
inputs is embedded at once). Read by the /v1/embeddings route."""

DEFAULT_MAX_TOP_K = 100
"""Max top_k / top_n / retrieve_k across the search/rerank routes. The one
retained operational guard (CONCEPT §9.7) — bounds KNN scan depth and the
response size. Read by the search/rerank/chunks routes."""

DEFAULT_SERVER_STORE_PATH = "embeddy.db"
"""Default on-disk store for the bare server (`uvicorn embeddy.server:app`
opens this relative to the working directory). Read by the server lifespan
(embeddy/server.py: SqliteStore.open(settings.store_path))."""

DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:8000"
"""Default server base URL the CLI's client commands talk to. Read by
embeddy/cli.py (`--base-url` CLI option > EMBEDDY_SERVER_BASE_URL file/env
> this default)."""


class ServerSettings(_PrecedenceSettings):
    """The `server` section (CONCEPT §5.9): CORS + the retained operational
    guards + the store the server opens + the CLI's server base URL.

    Config = implementation: every field is read by a code path (the
    CORS middleware / body-limit middleware / the embed+search routes / the
    server lifespan / the CLI). The size limits are OOM/crash-prevention
    ONLY (CONCEPT §9.7) — the trusted-tailnet server does no auth.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDY_SERVER_", extra="forbid", dotenv_filtering="match_prefix"
    )

    cors_origins: tuple[str, ...] = ()
    """CORS allow-list (no CORS headers when empty). Read by the server's
    CORSMiddleware. Tuple fields parse from env as JSON (verified 2026-08-01
    on pydantic-settings 2.14.2: `EMBEDDY_SERVER_CORS_ORIGINS='["http://a"]'`)."""
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    """Request-body ceiling (413 when exceeded) — OOM prevention."""
    max_embed_inputs: int = DEFAULT_MAX_EMBED_INPUTS
    """Max inputs per /v1/embeddings call (413 when exceeded) — OOM prevention."""
    max_top_k: int = DEFAULT_MAX_TOP_K
    """Max top_k/top_n/retrieve_k on the search+rerank routes (413 when
    exceeded) — the retained operational guard (CONCEPT §9.7)."""
    store_path: str = DEFAULT_SERVER_STORE_PATH
    """The sqlite store the bare server opens at startup (lifespan)."""
    base_url: str = DEFAULT_SERVER_BASE_URL
    """The server's own base URL — read by the CLI (embeddy/cli.py) as the
    default target for its client commands (EMBEDDY_SERVER_BASE_URL)."""

    @field_validator("max_body_bytes", "max_embed_inputs", "max_top_k")
    @classmethod
    def _limits_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"server limit must be >= 1, got {value}")
        return value

    @field_validator("store_path", "base_url")
    @classmethod
    def _strings_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


def load_server_settings(env_file: str | Path | None = None, **overrides: Any) -> ServerSettings:
    """Load SERVER settings with CLI > file > env > defaults precedence
    (the same source order as `load_settings`, EMBEDDY_SERVER_ prefix).

    Reads code paths that exist: `cors_origins` / `max_body_bytes` / the
    size limits feed the server factory (embeddy/server.py); `store_path` is
    opened by the server lifespan; `base_url` is the CLI's default client
    target (embeddy/cli.py).
    """
    return ServerSettings(_env_file=str(env_file) if env_file is not None else None, **overrides)
