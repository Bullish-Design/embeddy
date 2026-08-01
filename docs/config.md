# embeddy config reference (M7)

Config = implementation (CONCEPT §3.8): **no field exists unless a code path
reads it.** Precedence is **CLI > file > env > defaults** — a dotenv file
(`--env-file FILE`) sits between CLI options and env vars; each section
pulls only its own prefixed keys from a shared file
(`dotenv_filtering="match_prefix"`), so one `.env` can hold all sections.
Unknown init/CLI keys are rejected (`extra="forbid"`); unknown env vars are
ignored by pydantic-settings.

Env vars are the section prefix + uppercased field name:
`EMBEDDY_EMBEDDER_MODEL`, `EMBEDDY_PIPELINE_CONCURRENCY`,
`EMBEDDY_SERVER_MAX_TOP_K`, `EMBEDDY_STORE_URL`, ... Tuples parse from env
as JSON (`EMBEDDY_SERVER_CORS_ORIGINS='["http://a"]'`).

## `embedder` — model selection + MRL + role

| Field | Default | Read by |
|-------|---------|---------|
| `model` | `Qwen/Qwen3-Embedding-0.6B` | provider factory (`build_provider`), server/CLI |
| `embedding_dimension` | `None` (native) | `resolve_dimension` (MRL policy) |
| `prompt_role` | `"query"` | the server's query-role resolution (`resolve_instruction(model, prompt_role)`) and the pipeline |

The provider is built at the server/CLI seam: `build_provider(model,
embedding_dimension, backend="local")`. Non-MRL model + wrong dimension
raises at config time (the C2 class of silent no-op is impossible).

## `pipeline` — the worker-pool bound

| Field | Default | Read by |
|-------|---------|---------|
| `concurrency` | `4` | `IngestPipeline` (bounded worker pool) |

## `server` — CORS + the retained operational guards (CONCEPT §9.7)

The size limits are **OOM/crash-prevention ONLY, never security** (auth /
SSRF / path-traversal are out of scope on the trusted tailnet).

| Field | Default | Read by |
|-------|---------|---------|
| `cors_origins` | `()` (no CORS) | the server's CORSMiddleware |
| `max_body_bytes` | `10485760` (10 MiB) | the body-limit middleware (413) |
| `max_embed_inputs` | `1024` | `/v1/embeddings` (413) |
| `max_top_k` | `100` | search/rerank/chunks routes (413) |
| `store_path` | `"embeddy.db"` | the server lifespan (the sqlite fallback when `store.url` is unset) |
| `base_url` | `http://127.0.0.1:8000` | the CLI's default client target |

## `store` — the Searchable backend (Phase 8, one config line)

CONCEPT §5.4: *"Scale: Qdrant adapter. Dense + sparse vectors, payload
filters, quantization. One config line (`store: qdrant://...`)."* The
`store` section selects the backend behind the frozen `Searchable` protocol
(`embeddy/index/factory.py` — `build_store` mirrors `build_provider`).

| Field | Default | Read by |
|-------|---------|---------|
| `url` | `None` | the server lifespan (`build_store(dsn)`) and the CLI `serve`/`info` commands |

`url` is a store DSN. `None` (default) = the server's `store_path` sqlite
DSN — the bare `app = create_app()` behavior is unchanged. Supported forms
(`parse_store_url`, unit-tested):

- **sqlite**: `sqlite://<path>` | `sqlite://:memory:` | `sqlite:<path>` | a
  bare path (`embeddy.db`) | `:memory:`
- **qdrant**: `qdrant://:memory:` (offline, hermetic — the test mode) |
  `qdrant://host` | `qdrant://host:port` | `qdrant+https://host:port`, plus
  `?quantization=int8|binary|none` for collection-level quantization
  (applied at `create_collection` time — `docs/decisions/0004`).

Qdrant caveats (documented in `docs/decisions/0004` and `USER_GUIDE`): no
BM25 — `search_fts` is a pure-Python BM25 scan over payloads; no
transactions — reindex is upsert-first with the old chunks intact on
failure; the sync client blocks the event loop on a remote call. See also
the INTEGRATION guide's storage-backend section.

## Example `.env`

A shared dotenv file with all sections (the store section selects the
backend):

```dotenv
EMBEDDY_EMBEDDER_MODEL=microsoft/harrier-oss-v1-0.6b
EMBEDDY_EMBEDDER_EMBEDDING_DIMENSION=512
EMBEDDY_PIPELINE_CONCURRENCY=2
EMBEDDY_SERVER_MAX_TOP_K=25
EMBEDDY_SERVER_MAX_BODY_BYTES=5242880
EMBEDDY_SERVER_CORS_ORIGINS=["http://localhost:5173"]
EMBEDDY_STORE_URL=qdrant://localhost:6333?quantization=int8
```

## API surface summary (what the server exposes)

- `GET /health/live` — process up; `GET /health/ready` — provider loaded and
  store open (503 + reason otherwise).
- `POST /v1/embeddings` — OpenAI-compatible (`{"input", "model", "instruction"?}`);
  `POST /v1/rerank` — TEI/Jina shape (`{"query", "texts", "top_n"?}`).
- `POST /api/v1/search` (+ `/similar`, `/rerank`), `/api/v1/ingest/*`
  (text/file/dir + reindex/delete/sync), `/api/v1/collections`
  (create/list/stats), `/api/v1/chunks` (list/search).
- Structured errors: `{"error": {"type", "message", "detail"}}`; maps are
  documented in `embeddy/server.py`'s module docstring (400/404/413/422/500/
  501/503).

Client: `embeddy/client.py` (`EmbeddyClient`) mirrors every route and shares
the request builder with `HTTPProvider` (one wire protocol).
