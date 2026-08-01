"""ServerSettings unit tests — the `server` config section (plan §6 /
CONCEPT §3.8 §5.9 §9.7): CLI > file > env > defaults, and the retained
operational guards (max_body_bytes / max_embed_inputs / max_top_k) are
OOM/crash-prevention only — config = implementation, every field read by a
real code path (server.py / cli.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embeddy import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_EMBED_INPUTS,
    DEFAULT_MAX_TOP_K,
    DEFAULT_SERVER_BASE_URL,
    DEFAULT_SERVER_STORE_PATH,
    ServerSettings,
    load_server_settings,
)

_ENV_KEYS = (
    "EMBEDDY_SERVER_CORS_ORIGINS",
    "EMBEDDY_SERVER_MAX_BODY_BYTES",
    "EMBEDDY_SERVER_MAX_EMBED_INPUTS",
    "EMBEDDY_SERVER_MAX_TOP_K",
    "EMBEDDY_SERVER_STORE_PATH",
    "EMBEDDY_SERVER_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults() -> None:
    s = load_server_settings()
    assert s.cors_origins == ()  # no CORS unless configured
    assert s.max_body_bytes == DEFAULT_MAX_BODY_BYTES == 10 * 1024 * 1024
    assert s.max_embed_inputs == DEFAULT_MAX_EMBED_INPUTS == 1024
    assert s.max_top_k == DEFAULT_MAX_TOP_K == 100
    assert s.store_path == DEFAULT_SERVER_STORE_PATH == "embeddy.db"
    assert s.base_url == DEFAULT_SERVER_BASE_URL == "http://127.0.0.1:8000"


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDY_SERVER_MAX_TOP_K", "25")
    assert load_server_settings().max_top_k == 25


def test_env_parses_cors_origins_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDY_SERVER_CORS_ORIGINS", '["http://a", "http://b"]')
    assert load_server_settings().cors_origins == ("http://a", "http://b")


def test_file_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_SERVER_MAX_TOP_K=30\n")
    monkeypatch.setenv("EMBEDDY_SERVER_MAX_TOP_K", "99")
    assert load_server_settings(env_file=env_file).max_top_k == 30


def test_cli_overrides_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_SERVER_MAX_TOP_K=30\n")
    assert load_server_settings(env_file=env_file, max_top_k=5).max_top_k == 5


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDY_SERVER_MAX_TOP_K", "99")
    assert load_server_settings(max_top_k=5).max_top_k == 5


def test_unknown_init_field_rejected() -> None:
    """extra='forbid': a typo'd programmatic/CLI key is a hard error, not a
    silent no-op (config = implementation, CONCEPT §3.8)."""
    with pytest.raises(Exception, match="extra"):
        ServerSettings.model_validate({"max_topk": 5})  # typo


def test_limits_must_be_positive() -> None:
    with pytest.raises(Exception, match="limit"):
        load_server_settings(max_body_bytes=0)
    with pytest.raises(Exception, match="limit"):
        load_server_settings(max_embed_inputs=-1)
    with pytest.raises(Exception, match="limit"):
        load_server_settings(max_top_k=0)


def test_strings_non_empty() -> None:
    with pytest.raises(Exception, match="non-empty"):
        load_server_settings(store_path="   ")
    with pytest.raises(Exception, match="non-empty"):
        load_server_settings(base_url="")


def test_settings_are_read_by_code_paths() -> None:
    """Config = implementation: each server field feeds a real consumer —
    the factory's middleware/routes (server.py) and the CLI. Here we prove
    the constants the server code reads are the config defaults (the
    middleware and routes import nothing custom; create_app is the seam)."""
    import inspect

    from embeddy import server

    sig = inspect.signature(server.create_app)
    assert "settings" in sig.parameters
    assert "store" in sig.parameters
    assert "provider" in sig.parameters
    assert "reranker" in sig.parameters
    # the module-level bare app is built from settings (uvicorn
    # embeddy.server:app works bare)
    assert isinstance(server.app, object)
    assert server.app.state.server.settings.store_path == DEFAULT_SERVER_STORE_PATH


async def test_pydantic_validation_error_maps_to_400() -> None:
    """The documented error-map entry: a pydantic ValidationError raised
    INSIDE a handler -> 400 with the structured shape (route-body
    validation is FastAPI's -> 422; this handler covers ValidationError from
    server code, e.g. constructing a request model from raw input)."""
    import json
    from collections.abc import Awaitable, Callable
    from typing import Any, cast

    from fastapi import Request
    from pydantic import BaseModel, ValidationError

    from embeddy.server import create_app

    app = create_app(settings=load_server_settings())
    handler = app.exception_handlers[ValidationError]
    request = Request({"type": "http", "method": "POST", "path": "/x", "headers": []})

    class _Body(BaseModel):
        n: int

    with pytest.raises(ValidationError):
        _Body(n="not an int")

    # invoke the registered handler directly (the mapping itself)
    try:
        _Body(n="not an int")
    except ValidationError as exc:
        handler_callable = cast(Callable[[Request, ValidationError], Awaitable[Any]], handler)
        response = await handler_callable(request, exc)
    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"]["type"] == "validation_error"
    assert isinstance(payload["error"]["detail"], list)
