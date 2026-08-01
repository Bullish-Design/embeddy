"""embeddy/cli.py — the Typer CLI (plan §8: `serve`, `ingest text|file|dir`,
`search`, `info`). Config precedence CLI > file > env > defaults (config.py:
the `--env-file` dotenv file sits between CLI options and env vars).

The CLI is a CLIENT of the API server: `ingest`/`search` talk to a running
server via `EmbeddyClient` (embeddy/client.py); `serve` starts that server
(uvicorn on `create_app`); `info` prints the effective settings. `--base-url`
(CLI) > `EMBEDDY_SERVER_BASE_URL` (file/env) > the documented default.

typer is an `embeddy[server]` extra and is imported at module level here —
this module IS the CLI extra's entry point. `embeddy/__init__.py` never
imports it, so zero-extras `import embeddy` stays clean. Run via the
`embeddy` console script or `python -m embeddy.cli`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import typer

from embeddy.client import EmbeddyClient
from embeddy.config import (
    load_pipeline_settings,
    load_server_settings,
    load_settings,
)
from embeddy.errors import ClientError
from embeddy.registry import get_model, resolve_dimension

__all__ = ["app", "main"]

app = typer.Typer(
    name="embeddy",
    help="Typed, metric-honest embedding, storage, search, and serving (plan §8).",
)
ingest = typer.Typer(help="Ingest content into a collection on the running server.")
app.add_typer(ingest, name="ingest")


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #


def _resolve_base_url(base_url: str | None, env_file: str | None) -> str:
    """--base-url (CLI) > EMBEDDY_SERVER_BASE_URL (file/env) > default."""
    if base_url is not None:
        return base_url
    return load_server_settings(env_file=env_file).base_url


def _make_client(base_url: str | None, env_file: str | None) -> EmbeddyClient:
    return EmbeddyClient(_resolve_base_url(base_url, env_file))


async def _invoke(client: EmbeddyClient, coro: Awaitable[Any]) -> Any:
    async with client:
        return await coro


def _run(client: EmbeddyClient, coro: Awaitable[Any]) -> Any:
    """Run one async client call; a ClientError becomes a clean CLI error."""
    try:
        return asyncio.run(_invoke(client, coro))
    except ClientError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _print_stats(stats: dict[str, Any]) -> None:
    typer.echo(
        f"attempted={stats['files_attempted']} indexed={stats['files_indexed']} "
        f"skipped={stats['files_skipped']} deleted={stats['files_deleted']} "
        f"chunks={stats['chunks_indexed']} errors={len(stats['errors'])}"
    )
    for error in stats.get("errors", []):
        typer.echo(
            f"  error [{error.get('phase')}] {error.get('path')}: {error.get('message')}",
            err=True,
        )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    env_file: str | None = typer.Option(
        None, help="Path to a dotenv file (EMBEDDY_* keys), between CLI and env"
    ),
) -> None:
    """Run the embeddy API server (uvicorn on the FastAPI app)."""
    import uvicorn

    from embeddy.server import create_app

    settings = load_server_settings(env_file=env_file)
    typer.echo(
        f"serving embeddy API on http://{host}:{port} "
        f"(store={settings.store_path}, max_top_k={settings.max_top_k})"
    )
    uvicorn.run(create_app(settings=settings), host=host, port=port)


@app.command("info")
def info(
    env_file: str | None = typer.Option(
        None, help="Path to a dotenv file (EMBEDDY_* keys), between CLI and env"
    ),
) -> None:
    """Show the effective configuration (CLI > file > env > defaults)."""
    embedder = load_settings(env_file=env_file)
    pipeline = load_pipeline_settings(env_file=env_file)
    server = load_server_settings(env_file=env_file)
    spec = get_model(embedder.model)
    resolved = resolve_dimension(spec, embedder.embedding_dimension)
    typer.echo(f"embedder.model: {embedder.model}")
    typer.echo(
        f"embedder.embedding_dimension: {resolved} "
        f"(native {spec.native_dimension}, mrl {spec.mrl_range})"
    )
    typer.echo(f"embedder.prompt_role: {embedder.prompt_role}")
    typer.echo(f"pipeline.concurrency: {pipeline.concurrency}")
    typer.echo(f"server.base_url: {server.base_url}")
    typer.echo(f"server.store_path: {server.store_path}")
    typer.echo(
        f"server limits: max_body_bytes={server.max_body_bytes} "
        f"max_embed_inputs={server.max_embed_inputs} max_top_k={server.max_top_k}"
    )
    typer.echo(f"server.cors_origins: {list(server.cors_origins)}")


@ingest.command("text")
def ingest_text(
    text: str = typer.Argument(..., help="The text to ingest"),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to ingest into"),
    path: str = typer.Option("<memory>", "--path", help="Canonical source path"),
    content_type: str | None = typer.Option(
        None, "--content-type", help="Content type (markdown/python/...); None = auto-detect"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Server base URL"),
    env_file: str | None = typer.Option(
        None, help="Path to a dotenv file (EMBEDDY_* keys), between CLI and env"
    ),
) -> None:
    """Ingest raw text as one source on the running server."""
    client = _make_client(base_url, env_file)
    stats = _run(
        client,
        client.ingest_text(text, collection=collection, path=path, content_type=content_type),
    )
    _print_stats(stats)


@ingest.command("file")
def ingest_file(
    path: str = typer.Argument(..., help="Path to the file to ingest"),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to ingest into"),
    base_url: str | None = typer.Option(None, "--base-url", help="Server base URL"),
    env_file: str | None = typer.Option(
        None, help="Path to a dotenv file (EMBEDDY_* keys), between CLI and env"
    ),
) -> None:
    """Ingest one file on the running server (server-side filesystem path)."""
    client = _make_client(base_url, env_file)
    _print_stats(_run(client, client.ingest_file(path, collection=collection)))


@ingest.command("dir")
def ingest_dir(
    directory: str = typer.Argument(..., help="Directory to ingest recursively"),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to ingest into"),
    base_url: str | None = typer.Option(None, "--base-url", help="Server base URL"),
    env_file: str | None = typer.Option(
        None, help="Path to a dotenv file (EMBEDDY_* keys), between CLI and env"
    ),
) -> None:
    """Ingest every file under a directory on the running server."""
    client = _make_client(base_url, env_file)
    _print_stats(_run(client, client.ingest_directory(directory, collection=collection)))


@app.command("search")
def search(
    query: str = typer.Argument(..., help="The search query"),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to search"),
    top_k: int = typer.Option(10, "--top-k", help="Number of results to return"),
    mode: str = typer.Option("rrf", "--mode", help="Fusion mode (rrf|weighted)"),
    base_url: str | None = typer.Option(None, "--base-url", help="Server base URL"),
    env_file: str | None = typer.Option(
        None, help="Path to a dotenv file (EMBEDDY_* keys), between CLI and env"
    ),
) -> None:
    """Hybrid search on the running server (the server embeds the query)."""
    client = _make_client(base_url, env_file)
    result = _run(client, client.search(query, collection=collection, top_k=top_k, mode=mode))
    typer.echo(
        f"query: {query!r} (collection {collection!r}, mode {result.get('mode')}, "
        f"metric {result.get('metric')}, total {result.get('total_results')})"
    )
    for rank, hit in enumerate(result.get("results", []), start=1):
        score = float(hit.get("score", 0.0))
        snippet = str(hit.get("content", ""))[:80].replace("\n", " ")
        typer.echo(f"  {rank}. [{score:.4f}] {hit.get('chunk_id')}  {snippet}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover - module entry point (not reachable under pytest)
    main()
