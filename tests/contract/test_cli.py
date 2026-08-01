"""CLI tests — typer CliRunner (plan §8/§11). The CLI is a thin CLIENT of
the API server; the client commands are exercised against an httpx
MockTransport (the client<->server wire parity is covered by
test_client.py's contract walk), `serve` is monkeypatched at uvicorn.run,
and config precedence (CLI > file > env > defaults) is pinned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from embeddy.cli import app as cli_app
from embeddy.client import EmbeddyClient

pytestmark = pytest.mark.integration

runner = CliRunner()


def _mock_client(handler: Any) -> EmbeddyClient:
    """A real EmbeddyClient over MockTransport (used inside the CLI's own
    asyncio.run — the loop is created there, verified 2026-08-01)."""
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return EmbeddyClient("http://test", http_client=http_client)


def _stats_response() -> dict[str, object]:
    return {
        "files_attempted": 2,
        "files_indexed": 2,
        "files_skipped": 0,
        "files_deleted": 0,
        "chunks_indexed": 3,
        "errors": [],
    }


def _search_response() -> dict[str, object]:
    return {
        "results": [
            {
                "chunk_id": "src1:0",
                "collection_id": "acme",
                "source_id": "src1",
                "source_path": "a.txt",
                "content": "token expiry and rotation policy",
                "score": 0.0164,
                "metric": "rrf",
            }
        ],
        "total_results": 3,
        "metric": "rrf",
        "mode": "rrf",
        "query": "token policy",
        "collection": "acme",
    }


def test_help_exits_zero() -> None:
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout
    assert "ingest" in result.stdout
    assert "search" in result.stdout
    assert "info" in result.stdout


def test_serve_runs_uvicorn_on_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[Any, str, int]] = []

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured.append((app, kwargs["host"], kwargs["port"]))

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert result.exit_code == 0
    assert "serving embeddy API on http://0.0.0.0:9000" in result.stdout
    (app, host, port) = captured[0]
    assert host == "0.0.0.0"
    assert port == 9000
    assert app is not None  # the FastAPI app (created from settings)


def test_info_prints_effective_config() -> None:
    result = runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0
    assert "embedder.model: Qwen/Qwen3-Embedding-0.6B" in result.stdout
    assert "embedder.embedding_dimension: 1024" in result.stdout
    assert "pipeline.concurrency: 4" in result.stdout
    assert "server.store_path: embeddy.db" in result.stdout


def test_info_file_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_EMBEDDER_MODEL=microsoft/harrier-oss-v1-0.6b\n")
    monkeypatch.setenv("EMBEDDY_EMBEDDER_MODEL", "from_env")
    result = runner.invoke(cli_app, ["info", "--env-file", str(env_file)])
    assert result.exit_code == 0
    assert "embedder.model: microsoft/harrier-oss-v1-0.6b" in result.stdout


def test_ingest_text_requires_collection() -> None:
    result = runner.invoke(cli_app, ["ingest", "text", "hello"])
    assert result.exit_code == 2  # usage error: --collection missing
    assert "--collection" in result.stderr


def test_search_prints_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body["query"]))
        return httpx.Response(200, json=_search_response())

    monkeypatch.setattr(
        "embeddy.cli._make_client", lambda base_url, env_file: _mock_client(handler)
    )
    result = runner.invoke(
        cli_app, ["search", "token policy", "--collection", "acme", "--top-k", "3"]
    )
    assert result.exit_code == 0
    assert calls == [("/api/v1/search", "token policy")]
    assert "src1:0" in result.stdout
    assert "metric rrf" in result.stdout
    assert "0.0164" in result.stdout


def test_search_client_error_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": {"type": "not_found", "message": "unknown collection"}}
        )

    monkeypatch.setattr(
        "embeddy.cli._make_client", lambda base_url, env_file: _mock_client(handler)
    )
    result = runner.invoke(cli_app, ["search", "q", "--collection", "nope"])
    assert result.exit_code == 1
    assert "unknown collection" in result.stderr


def test_ingest_text_prints_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_stats_response())

    monkeypatch.setattr(
        "embeddy.cli._make_client", lambda base_url, env_file: _mock_client(handler)
    )
    result = runner.invoke(
        cli_app,
        [
            "ingest",
            "text",
            "hello world",
            "--collection",
            "acme",
            "--path",
            "note.md",
            "--content-type",
            "markdown",
        ],
    )
    assert result.exit_code == 0
    assert "indexed=2" in result.stdout
    assert seen[0]["path"] == "note.md"
    assert seen[0]["content_type"] == "markdown"


def test_ingest_dir_prints_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_stats_response())

    monkeypatch.setattr(
        "embeddy.cli._make_client", lambda base_url, env_file: _mock_client(handler)
    )
    result = runner.invoke(cli_app, ["ingest", "dir", "/data", "--collection", "acme"])
    assert result.exit_code == 0
    assert "chunks=3" in result.stdout


def test_info_shared_multisection_env_file(tmp_path: Path) -> None:
    """One dotenv file with ALL sections must load cleanly (the
    dotenv_filtering="match_prefix" contract — each section pulls only its
    own prefixed keys; extra="forbid" must not reject foreign keys)."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EMBEDDY_EMBEDDER_MODEL=microsoft/harrier-oss-v1-0.6b\n"
        "EMBEDDY_PIPELINE_CONCURRENCY=2\n"
        "EMBEDDY_SERVER_MAX_TOP_K=7\n"
    )
    result = runner.invoke(cli_app, ["info", "--env-file", str(env_file)])
    assert result.exit_code == 0
    assert "embedder.model: microsoft/harrier-oss-v1-0.6b" in result.stdout
    assert "pipeline.concurrency: 2" in result.stdout
    assert "max_top_k=7" in result.stdout


def test_base_url_cli_option_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_response())

    def fake_make_client(base_url: str | None, env_file: str | None) -> EmbeddyClient:
        captured.append(base_url or load_env_base(env_file))
        return _mock_client(handler)

    def load_env_base(env_file: str | None) -> str:
        from embeddy.config import load_server_settings

        return load_server_settings(env_file=env_file).base_url

    monkeypatch.setattr("embeddy.cli._make_client", fake_make_client)
    result = runner.invoke(
        cli_app,
        ["search", "q", "--collection", "acme", "--base-url", "http://other:9000"],
    )
    assert result.exit_code == 0
    assert captured == ["http://other:9000"]


def test_base_url_from_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_SERVER_BASE_URL=http://from-file:7000\n")
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_response())

    def fake_make_client(base_url: str | None, env_file: str | None) -> EmbeddyClient:
        from embeddy.config import load_server_settings

        captured.append(
            base_url if base_url is not None else load_server_settings(env_file=env_file).base_url
        )
        return _mock_client(handler)

    monkeypatch.setattr("embeddy.cli._make_client", fake_make_client)
    result = runner.invoke(
        cli_app, ["search", "q", "--collection", "acme", "--env-file", str(env_file)]
    )
    assert result.exit_code == 0
    assert captured == ["http://from-file:7000"]


# --------------------------------------------------------------------------- #
# unit-level coverage of the CLI plumbing (the real client/uvicorn paths)
# --------------------------------------------------------------------------- #


def test_resolve_base_url_precedence(tmp_path: Path) -> None:
    from embeddy.cli import _resolve_base_url

    # CLI option wins
    assert _resolve_base_url("http://cli:1", None) == "http://cli:1"
    # no option -> env-file -> default
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDY_SERVER_BASE_URL=http://file:2\n")
    assert _resolve_base_url(None, str(env_file)) == "http://file:2"
    assert _resolve_base_url(None, None) == "http://127.0.0.1:8000"


def test_make_client_constructs_real_client() -> None:
    """The real _make_client (not monkeypatched) builds an EmbeddyClient
    with the resolved base URL and an OWNED httpx client (the server-command
    path). No request is made here."""
    from embeddy.cli import _make_client

    client = _make_client("http://real:9000", None)
    assert client.base_url == "http://real:9000"
    assert client._owns_client is None  # lazy: created on first request


def test_print_stats_with_errors(capsys: pytest.CaptureFixture[str]) -> None:
    from embeddy.cli import _print_stats

    _print_stats(
        {
            "files_attempted": 2,
            "files_indexed": 1,
            "files_skipped": 0,
            "files_deleted": 0,
            "chunks_indexed": 1,
            "errors": [{"path": "bad.md", "phase": "chunk", "message": "boom"}],
        }
    )
    out = capsys.readouterr()
    assert "attempted=2 indexed=1" in out.out
    assert "error [chunk] bad.md: boom" in out.err


def test_ingest_file_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_stats_response())

    monkeypatch.setattr(
        "embeddy.cli._make_client", lambda base_url, env_file: _mock_client(handler)
    )
    result = runner.invoke(cli_app, ["ingest", "file", "/tmp/a.txt", "--collection", "acme"])
    assert result.exit_code == 0
    assert "indexed=2" in result.stdout


def test_main_entrypoint_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m embeddy.cli` runs main() -> app(); --help exits 0."""
    import embeddy.cli as cli_module

    monkeypatch.setattr("sys.argv", ["embeddy", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()
    assert excinfo.value.code == 0
