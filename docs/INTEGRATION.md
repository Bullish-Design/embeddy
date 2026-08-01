# Integration guide — the wire protocol and upstreams

embeddy speaks **one wire protocol** in two roles (CONCEPT §7): the
`HTTPProvider` (embeddy as a **client** of any serving layer) and the embeddy
server + `EmbeddyClient` (embeddy as a **server**). Everything below is the
implemented surface at M6.

## The protocol

### `POST /v1/embeddings` — OpenAI-compatible

Request:

```json
{"input": ["text a", "text b"], "model": "optional/provider-check", "instruction": "optional already-resolved instruction"}
```

Response (OpenAI shape; `usage` is deliberately omitted — the server does
not count tokens):

```json
{"object": "list", "data": [{"object": "embedding", "embedding": [0.1, ...], "index": 0}], "model": "Qwen/Qwen3-Embedding-0.6B"}
```

Scope (be honest — OpenAI-compat is not universal):

- Carries **text, dense, single-vector** embeddings. That interoperable core
  is what the `HTTPProvider` uses.
- **Instructions** are sent via `extra_body` — an upstream that ignores
  `extra_body` silently drops them. Providers declare whether they honor
  instructions.
- **Multimodal** (image inputs) and **learned-sparse** (bge-m3) are NOT
  expressible over this endpoint. In v1 they are local-provider only.
- The provider validates the response's dimension against the resolved
  dimension; a mismatch raises.

### `POST /v1/rerank` — TEI/Jina shape

There is **no** OpenAI rerank standard; embeddy uses the TEI/Jina/Cohere
shape:

```json
{"query": "q", "texts": ["a", "b"], "top_n": 5}
```

### `GET /health/live` / `GET /health/ready`

- `live` — process up (200).
- `ready` — provider loaded AND store open (200); otherwise 503 with
  `{"ready": false, "reason": ...}`. The client's `health_ready()` treats a
  503 as an **answer**, not an error.

### embeddy extensions — `/api/v1/*`

`/api/v1/search`, `/similar`, `/rerank`, `/api/v1/ingest/{text,file,dir,
reindex,delete,sync}`, `/api/v1/collections`, `/api/v1/chunks` — full route
list and request/response shapes are in `docs/USER_GUIDE.md` §3 and the
server module docstring.

## embeddy as a client of your serving layer

The `HTTPProvider` is a drop-in client for any OpenAI-compatible embeddings
endpoint — **TEI, Infinity, vLLM, Ollama, OpenAI** are all valid upstreams:

```python
from embeddy import build_provider

provider = build_provider(
    "Qwen/Qwen3-Embedding-0.6B",
    backend="http",
    base_url="http://localhost:8080",      # your serving layer
    api_key=None,                          # OpenAI-style auth, if your layer has it
)
vecs = await provider.encode(["hello"], instruction="query")
```

The shared request builder (`build_embeddings_request`) is the SAME code the
`EmbeddyClient` uses — one wire shape, structurally no C5-class drift.

The reranker has the same shape split: `HTTPReranker` targets a TEI/Jina
rerank endpoint; `CrossEncoderReranker` (`embeddy[local]`) is in-process.

## embeddy as a server (for your app / agents / other tools)

```bash
pip install "embeddy[server]"
embeddy serve --host 0.0.0.0 --port 9000 --env-file .env
# or programmatically:
#   uvicorn embeddy.server:app   (bare app; opens settings.store_path)
#   from embeddy.server import create_app; create_app(store=..., provider=...)
```

The bare app builds a **local** provider from the embedder settings; when the
provider cannot load (missing `embeddy[local]`), the process stays up and
reports not-ready — the honest health contract.

Point an OpenAI-compatible client (or the embeddy `EmbeddyClient`) at
`/v1/embeddings`. `curl` smoke test:

```bash
curl -s localhost:9000/health/ready
curl -s localhost:9000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input": ["token expiry policy"], "model": "Qwen/Qwen3-Embedding-0.6B"}'
curl -s localhost:9000/v1/rerank \
  -H 'content-type: application/json' \
  -d '{"query": "token expiry", "texts": ["tokens expire after 30 days", "rate limits"], "top_n": 2}'
```

## Error contract

All errors are structured: `{"error": {"type", "message", "detail"}}`.
Mappings (documented in `embeddy/server.py`):

| Status | Meaning |
|--------|---------|
| 400 | validation inside a handler (bad model, weights/min_score violations) |
| 404 | unknown collection / unknown chunk id |
| 413 | body > `max_body_bytes`, inputs > `max_embed_inputs`, `top_k` > `max_top_k` |
| 422 | FastAPI request validation (bad body/query shape) |
| 500 | `EmbeddyError` / unexpected internal error |
| 501 | backend missing a store extension (`create_collection`/`get_chunk`/`list_chunks`/`list_collections`) |
| 503 | not ready, or no reranker configured |

The size limits are **OOM/crash-prevention only, never security** — auth,
SSRF, and path-traversal protection are explicitly out of scope (personal
project on a trusted tailnet). Do not put this server on an untrusted network
without adding your own auth layer.

## CORS

`server.cors_origins` (tuple; parsed from env as JSON):

```dotenv
EMBEDDY_SERVER_CORS_ORIGINS=["http://localhost:5173"]
```

No CORS headers are emitted when the tuple is empty (default).

## Config precedence

CLI > file (`--env-file`) > env > defaults. Sections pull only their own
prefixed keys from a shared dotenv file (`dotenv_filtering="match_prefix"`),
so one `.env` can hold all sections. See `docs/config.md`.

## Lifecycle contract for embedders

- The server lifespan **opens the store, builds + loads the provider, closes
  everything on shutdown** — the DI seam (`create_app`) is the integration
  point for embedding embeddy's API inside another FastAPI app or test
  harness.
- `EmbeddyClient` is an async context manager; it owns an `httpx.AsyncClient`
  unless one is injected (tests inject `MockTransport`/`ASGITransport`).
