"""Embeddy settings — pydantic-settings, one source of truth (CONCEPT §3.8,
§5.9). "Config = implementation": no field exists unless a code path reads
it. Phase 3 adds only the `embedder` section; `store` / `chunk` / `pipeline`
/ `server` land with their consuming code paths in Phases 4-6.

Precedence is CLI > file > env > defaults (plan §6 / CONCEPT §3.8):
`settings_customise_sources` puts the dotenv file BEFORE the environment
source (stock pydantic-settings is init > env > dotenv — verified 2026-08-01
on pydantic-settings 2.14.2); explicit init kwargs (CLI values) stay first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderSettings(BaseSettings):
    """The `embedder` section: model selection + MRL resolution + the
    semantic role used for instruction resolution (registry)."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDY_EMBEDDER_", extra="forbid")

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int | None = None
    prompt_role: str = "query"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """CLI(init) > file(dotenv) > env > secrets > defaults."""
        del settings_cls
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)


def load_settings(env_file: str | Path | None = None, **overrides: Any) -> EmbedderSettings:
    """Load settings with CLI > file > env > defaults precedence.

    `env_file` points at a dotenv file (KEY=VALUE, KEYs use the
    EMBEDDY_EMBEDDER_ prefix); `overrides` are CLI/programmatic values and
    win over everything. Reads a code path that exists: the factory and
    instruction resolution consume the resulting settings (config.py ->
    factory.build_provider -> registry.resolve_dimension / resolve_instruction).
    """
    return EmbedderSettings(_env_file=str(env_file) if env_file is not None else None, **overrides)
