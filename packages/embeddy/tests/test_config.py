"""Config tests — pydantic-settings precedence (plan §6 / CONCEPT §3.8):
CLI > file > env > defaults. Verified 2026-08-01 on pydantic-settings
2.14.2: stock order is init > env > dotenv, so the custom source order in
config.py puts the dotenv FILE before ENV. Non-MRL model + wrong dimension
raises at config time (via the provider factory).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embeddy import EmbedderSettings, load_settings
from embeddy.errors import ProviderError
from embeddy.providers.factory import build_provider

_ENV_KEYS = (
    "EMBEDDY_EMBEDDER_MODEL",
    "EMBEDDY_EMBEDDER_EMBEDDING_DIMENSION",
    "EMBEDDY_EMBEDDER_PROMPT_ROLE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults() -> None:
    s = EmbedderSettings()
    assert s.model == "Qwen/Qwen3-Embedding-0.6B"
    assert s.embedding_dimension is None  # None = native (MRL policy)
    assert s.prompt_role == "query"


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDY_EMBEDDER_MODEL", "microsoft/harrier-oss-v1-0.6b")
    s = EmbedderSettings()
    assert s.model == "microsoft/harrier-oss-v1-0.6b"


def test_file_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_EMBEDDER_MODEL=from_file\n")
    monkeypatch.setenv("EMBEDDY_EMBEDDER_MODEL", "from_env")
    s = load_settings(env_file=env_file)
    assert s.model == "from_file", "file must beat env (CLI > file > env > defaults)"


def test_cli_overrides_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_EMBEDDER_MODEL=from_file\n")
    s = load_settings(env_file=env_file, model="from_cli")
    assert s.model == "from_cli"


def test_file_parses_dimension(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_EMBEDDER_EMBEDDING_DIMENSION=256\n")
    s = load_settings(env_file=env_file)
    assert s.embedding_dimension == 256


def test_unknown_init_field_rejected() -> None:
    """extra='forbid': a typo'd programmatic/CLI key is a hard error, not a
    silent no-op (config = implementation, CONCEPT §3.8). Note: pydantic-
    settings 2.14.2 silently ignores unknown ENV vars even with
    extra='forbid' (verified 2026-08-01) — the forbid applies to init
    kwargs, which is where CLI values enter."""
    with pytest.raises(Exception, match="extra"):
        # model_validate with an extra key: forbid applies to validation
        # (pydantic-settings 2.14.2 silently ignores unknown ENV vars, so
        # the forbid is enforced here — where CLI values enter).
        EmbedderSettings.model_validate({"modle": "typo"})


def test_prompt_role_default_reads_instruction() -> None:
    """The prompt_role default is a code path: resolve the instruction for
    the configured role + model (the search-side instruction the pipeline
    will use)."""
    s = EmbedderSettings()
    from embeddy import resolve_instruction

    instruction = resolve_instruction(s.model, s.prompt_role)
    assert "Instruct: Given a web search query" in instruction


# --- config-time validation (the C2 bug class is impossible) -------------------


def test_config_time_non_mrl_wrong_dimension_raises() -> None:
    """Config says harrier (non-MRL, native 1024) at dim 512 -> raises at
    config time, before any model load or request."""
    with pytest.raises(ValueError, match="not MRL-capable"):
        build_provider("microsoft/harrier-oss-v1-0.6b", 512, backend="local")


def test_config_time_unknown_model_raises() -> None:
    with pytest.raises(Exception, match="unknown model"):
        build_provider("no/such-model", backend="local")


def test_config_time_mrl_in_range_ok() -> None:
    provider = build_provider("Qwen/Qwen3-Embedding-0.6B", 256, backend="local")
    assert provider.dimension == 256
    assert provider.context_length == 32768


def test_config_time_unknown_backend_raises() -> None:
    with pytest.raises(ProviderError, match="backend"):
        build_provider("Qwen/Qwen3-Embedding-0.6B", backend="qdrant")


def test_config_to_provider_roundtrip() -> None:
    """The settings -> factory path: embedder.model + embedding_dimension
    drive provider construction (config = implementation)."""
    settings = load_settings(embedding_dimension=256)
    provider = build_provider(settings.model, settings.embedding_dimension, backend="local")
    assert provider.dimension == 256
    assert provider.model_name == settings.model
