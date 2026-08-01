# embeddy CLI reference (M5)

The CLI is a **client** of the API server: `ingest`/`search` talk to a
running server via `EmbeddyClient`; `serve` starts that server; `info`
prints the effective configuration. Every command honors the config
precedence **CLI > file > env > defaults** (`--env-file` sits between CLI
options and env vars; `--base-url` beats `EMBEDDY_SERVER_BASE_URL`, which
beats `http://127.0.0.1:8000`).

Install: `pip install "embeddy[server]"` provides the `embeddy` console
script (typer is a `server` extra). Without extras you can still run
`python -m embeddy.cli`.

## Commands

### `embeddy serve [--host HOST] [--port PORT] [--env-file FILE]`

Run the API server (uvicorn on `create_app`). The app opens
`server.store_path` (default `embeddy.db`) and builds the provider from the
`embedder` settings.

```console
$ embeddy serve --host 0.0.0.0 --port 9000 --env-file .env
```

### `embeddy ingest text TEXT --collection NAME [--path PATH] [--content-type TYPE] [--base-url URL] [--env-file FILE]`

Ingest raw text as one source (no file I/O). `content_type` None = auto
routing by the chunker registry.

```console
$ embeddy ingest text "token expiry policy" --collection acme --path docs/note.md --content-type markdown
```

### `embeddy ingest file PATH --collection NAME [--base-url URL] [--env-file FILE]`

Ingest one file (a server-side filesystem path — the trusted-tailnet server
does no path-traversal protection). Unchanged content at the same path is
skipped by the dedup policy; changed content is atomically reindexed.

### `embeddy ingest dir DIRECTORY --collection NAME [--base-url URL] [--env-file FILE]`

Ingest every file under a directory recursively through the bounded worker
pool (`pipeline.concurrency`). Store paths are collection-relative.

### `embeddy search QUERY --collection NAME [--top-k N] [--mode rrf|weighted] [--base-url URL] [--env-file FILE]`

Hybrid search on the running server (the server embeds the query with the
query-role instruction). Prints the ranked hits with score and a content
snippet.

### `embeddy info [--env-file FILE]`

Print the effective configuration — every section that exists
(`embedder`, `pipeline`, `server`) — as resolved by the precedence chain.

```console
$ embeddy info
embedder.model: Qwen/Qwen3-Embedding-0.6B
embedder.embedding_dimension: 1024 (native 1024, mrl range(32, 1025))
embedder.prompt_role: query
pipeline.concurrency: 4
server.base_url: http://127.0.0.1:8000
server.store_path: embeddy.db
server limits: max_body_bytes=10485760 max_embed_inputs=1024 max_top_k=100
server.cors_origins: []
```

## Exit codes

- `0` — success (including `--help`).
- `1` — a `ClientError` (the server answered non-2xx, or was unreachable).
- `2` — usage error (missing required option/argument).

## Programmatic use

The CLI is a thin wrapper over `EmbeddyClient`; use the client directly for
scripting (see `docs/config.md` for the wire surface):

```python
from embeddy.client import EmbeddyClient

async with EmbeddyClient("http://127.0.0.1:8000") as client:
    stats = await client.ingest_text("hello", "acme")
    result = await client.search("hello", "acme")
```
