"""Embeddy settings — pydantic-settings, one source of truth (CONCEPT §3.8,
§5.9). "Config = implementation": no field exists unless a code path reads
it. Phase 5 adds the `pipeline` section (the worker-pool bound); `store` /
`chunk` / `server` land with their consuming code paths in Phases 4-6.

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
    is init > env > dotenv, so the dotenv file must be moved BEFORE env)."""

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

    model_config = SettingsConfigDict(env_prefix="EMBEDDY_EMBEDDER_", extra="forbid")

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

    model_config = SettingsConfigDict(env_prefix="EMBEDDY_PIPELINE_", extra="forbid")

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
