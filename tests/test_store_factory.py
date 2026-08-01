"""Phase-8 unit tests — store DSN parsing, the build_store factory, and the
`store` config section (plan §10 / CONCEPT §5.4 "one config line").
Mock-free: parse_store_url is pure; build_store opens real in-memory stores.
"""

from __future__ import annotations

import pytest

from embeddy.config import DEFAULT_STORE_URL, StoreSettings, load_store_settings
from embeddy.index.base import StoreError
from embeddy.index.factory import StoreSpec, build_store, parse_store_url
from embeddy.index.qdrant import QdrantStore
from embeddy.index.sqlite import SqliteStore

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# parse_store_url — the DSN grammar
# --------------------------------------------------------------------------- #


def test_sqlite_dsn_variants() -> None:
    assert parse_store_url("sqlite://embeddy.db") == StoreSpec(
        backend="sqlite", url="sqlite://embeddy.db", sqlite_path="embeddy.db"
    )
    assert parse_store_url("sqlite://:memory:") == StoreSpec(
        backend="sqlite", url="sqlite://:memory:", sqlite_path=":memory:"
    )
    assert parse_store_url("sqlite:/abs/path.db") == StoreSpec(
        backend="sqlite", url="sqlite:/abs/path.db", sqlite_path="/abs/path.db"
    )
    assert parse_store_url("sqlite:rel/path.db") == StoreSpec(
        backend="sqlite", url="sqlite:rel/path.db", sqlite_path="rel/path.db"
    )
    assert parse_store_url("sqlite://") == StoreSpec(
        backend="sqlite", url="sqlite://", sqlite_path=":memory:"
    )


def test_bare_path_is_sqlite() -> None:
    assert parse_store_url("embeddy.db").backend == "sqlite"
    assert parse_store_url("embeddy.db").sqlite_path == "embeddy.db"
    assert parse_store_url(":memory:").sqlite_path == ":memory:"


def test_qdrant_dsn_forms() -> None:
    spec = parse_store_url("qdrant://localhost:6333")
    assert spec.backend == "qdrant"
    assert spec.host == "localhost"
    assert spec.port == 6333
    assert spec.https is False
    assert spec.memory is False
    # default port when omitted
    assert parse_store_url("qdrant://localhost").port == 6333
    # https transport
    spec = parse_store_url("qdrant+https://db.example:6334")
    assert spec.host == "db.example"
    assert spec.port == 6334
    assert spec.https is True


def test_qdrant_in_memory_dsn() -> None:
    spec = parse_store_url("qdrant://:memory:")
    assert spec.backend == "qdrant"
    assert spec.memory is True
    assert spec.host is None


def test_qdrant_quantization_query_param() -> None:
    assert parse_store_url("qdrant://h:6333?quantization=int8").quantization == "int8"
    assert parse_store_url("qdrant://h:6333?quantization=binary").quantization == "binary"
    assert parse_store_url("qdrant://h:6333?quantization=none").quantization is None
    assert parse_store_url("qdrant://h:6333").quantization is None


def test_invalid_dsns_raise() -> None:
    with pytest.raises(StoreError):
        parse_store_url("")
    with pytest.raises(StoreError):
        parse_store_url("   ")
    with pytest.raises(StoreError):
        parse_store_url("qdrant://")  # no host
    with pytest.raises(StoreError):
        parse_store_url("qdrant://?x=1")  # no host
    with pytest.raises(StoreError):
        parse_store_url("qdrant://:6333")  # netloc with no hostname
    with pytest.raises(StoreError):
        parse_store_url("qdrant://h?quantization=float16")  # unknown quantization


# --------------------------------------------------------------------------- #
# build_store — the factory (mirrors build_provider)
# --------------------------------------------------------------------------- #


async def test_build_store_sqlite_memory() -> None:
    store = await build_store("sqlite://:memory:")
    try:
        assert isinstance(store, SqliteStore)
        await store.create_collection("c", 4)
        stats = await store.stats("c")
        assert stats.chunk_count == 0
    finally:
        await store.close()


async def test_build_store_qdrant_memory() -> None:
    store = await build_store("qdrant://:memory:")
    try:
        assert isinstance(store, QdrantStore)
        await store.create_collection("c", 4)
        stats = await store.stats("c")
        assert stats.chunk_count == 0
    finally:
        await store.close()


async def test_build_store_qdrant_unreachable_raises() -> None:
    """The honest-health contract: an unreachable qdrant is a config-time
    StoreError (the lifespan records not-ready), never a half-open store.
    localhost:1 refuses instantly — offline, no external network."""
    with pytest.raises(StoreError, match="cannot reach qdrant"):
        await build_store("qdrant://127.0.0.1:1")


# --------------------------------------------------------------------------- #
# the `store` config section (CONCEPT §5.9 / §3.8)
# --------------------------------------------------------------------------- #


def test_store_defaults() -> None:
    settings = StoreSettings()
    assert settings.url is DEFAULT_STORE_URL  # None -> server.store_path sqlite
    assert settings.url is None


def test_store_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDY_STORE_URL", "qdrant://localhost:6333")
    assert StoreSettings().url == "qdrant://localhost:6333"


def test_store_file_overrides_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_STORE_URL=qdrant://:memory:\n")
    monkeypatch.setenv("EMBEDDY_STORE_URL", "from_env")
    settings = load_store_settings(env_file=str(env_file))
    assert settings.url == "qdrant://:memory:"


def test_store_cli_overrides_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_STORE_URL=qdrant://:memory:\n")
    settings = load_store_settings(env_file=str(env_file), url="sqlite://other.db")
    assert settings.url == "sqlite://other.db"


def test_store_unknown_init_field_rejected() -> None:
    with pytest.raises(Exception, match="extra"):
        StoreSettings.model_validate({"url": "x", "not_a_field": "y"})


def test_store_settings_are_read_by_code_path() -> None:
    """Config = implementation (CONCEPT §3.8): the field feeds build_store."""
    settings = load_store_settings(url="qdrant://:memory:")
    spec = parse_store_url(settings.url or "")
    assert spec.backend == "qdrant"
    assert spec.memory is True
