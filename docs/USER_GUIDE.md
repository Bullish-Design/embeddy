# User guide — chonkai + embeddy

Two libraries in one repository. **chonkai** ingests, parses, and chunks
documents with guaranteed invariants. **embeddy** embeds, stores, searches,
and serves them — typed, metric-honest retrieval. embeddy consumes chonkai
(one-way dependency); chonkai never imports embeddy.

This guide covers what is **implemented today** (M6). Nothing here is
aspirational — every feature and config field listed has a code path that
reads it.

## Install

```bash
pip install chonkai embeddy            # zero extras: core only
pip install "embeddy[server]"          # + FastAPI server + CLI (typer)
pip install "embeddy[client]"          # + httpx client
pip install "embeddy[local]"           # + sentence-transformers (local models)
pip install "chonkai[docling]"         # + Docling (PDF/DOCX/HTML parsing)
pip install "chonkai[tokenizers]"      # + arbitrary-HF tokenizers
```

Both packages import cleanly with **zero extras** (`import chonkai`,
`import embeddy`). Heavy dependencies are lazy: `docling`, `tokenizers`,
`sentence-transformers` (torch), `fastapi`/`starlette`, `httpx`, `typer` are
only imported where their extra is installed. The two module-level entry
points — `embeddy.server` (needs `[server]`) and `embeddy.cli` (needs
`[server]`) — raise `ImportError` without their extra by design.

Extras in the `server` group provide the `embeddy` console script.

## 1. Document processing (chonkai)

```python
from chonkai import Ingestor, get_chunker, ChunkBudget, ValidatedChunker

ingest = Ingestor().ingest_file("src/foo.py")                 # sync, IngestResult
ingest = Ingestor().ingest_text("raw text", content_type="markdown")  # sync, no file I/O

chunker = ValidatedChunker(                                  # invariant wrapper
    get_chunker(ingest.source.content_type, config={"granularity": "function"})
)
chunks = chunker.chunk(ingest, budget=ChunkBudget(max_tokens=512))
```

`ValidatedChunker` guarantees, on every chunk: non-empty content, `start_line`
/ `end_line` present (1-based, inclusive), `chunk_type` in the known
vocabulary, and `token_count <= budget.max_tokens` (oversized chunks are
post-split on token windows, never silently truncated).

Chunker routing (`strategy="auto"` by content type):

| Content type | Chunker |
|--------------|---------|
| `pdf`/`docx`/`html`/`img`/`tex` (docling-routed) | `DoclingChunker` (`chonkai[docling]`) |
| 10 code types (`python`, `javascript`, `typescript`, `rust`, `go`, `c`, `cpp`, `java`, `ruby`, `bash`) + `markdown` | `TreesitterChunker` (bundled grammars, offline-safe) |
| `rst` | `MarkdownChunker` |
| generic text | `SemchunkChunker` (token-accurate) |

Explicit strategies `paragraph`, `markdown`, `semchunk`, `treesitter`,
`docling` bypass routing. Tree-sitter granularity (`function`/`class`/
`module`) selects which definitions become chunks; decorators are recovered
from the raw parse tree and prepended.

## 2. The retrieval workflow (embeddy)

### Model registry and MRL

Model facts (dimension, context, MRL range) come from the **registry**, never
from config:

```python
from embeddy import get_model, resolve_dimension

spec = get_model("Qwen/Qwen3-Embedding-0.6B")
dim = resolve_dimension(spec, None)     # native 1024
dim = resolve_dimension(spec, 512)      # MRL-truncated (32..1024)
# resolve_dimension(spec_non_mrl, wrong) raises at config time — never silent
```

Registered models (implemented): `Qwen/Qwen3-Embedding-0.6B` (default, MRL
32–1024), `Qwen/Qwen3-VL-Embedding-2B` (multimodal, local-only in v1),
`microsoft/harrier-oss-v1-0.6b` / `-270m` / `-27b` (non-MRL high-quality
options). MRL truncation is always followed by L2 re-normalization (cosine
assumes unit vectors).

### Providers

```python
from embeddy import build_provider

provider = build_provider("Qwen/Qwen3-Embedding-0.6B", dimension=None, backend="local")
provider = build_provider("Qwen/Qwen3-Embedding-0.6B", dimension=512, backend="http", base_url="http://localhost:8080")
```

- **local** (`embeddy[local]`): sentence-transformers adapter. Loads a model
  by registry id, resolves instruction prompts from the model card, supports
  MRL truncation, batching, device options.
- **http**: OpenAI-compatible `POST /v1/embeddings` — TEI, Infinity, vLLM,
  Ollama, and OpenAI are drop-in upstreams. Text-dense only in v1;
  multimodal/learned-sparse are local-only (the OpenAI protocol cannot carry
  them). Instructions go via `extra_body` (an upstream that ignores it will
  silently drop the instruction — providers declare whether they honor it).

Providers expose `dimension` (the **resolved** dimension), `context_length`
(drives the chunk budget), and `model_name` — never config knobs.

### Ingest

The pipeline is a bounded, overlapping, source-aware worker pool. Sources are
first-class: dedup, reindex (atomic swap), delete (cascade), and incremental
sync are all **source** operations.

```python
import asyncio
from embeddy import IngestPipeline, build_provider
from embeddy.index.sqlite import SqliteStore

async def main():
    store = await SqliteStore.open("embeddy.db")
    await store.create_collection("acme", dimension=1024)   # SqliteStore extension
    pipeline = IngestPipeline(
        store=store,
        provider=build_provider("Qwen/Qwen3-Embedding-0.6B", backend="local"),
        concurrency=4,
    )
    stats = await pipeline.ingest_directory("docs/", collection="acme")
    print(stats)  # files_attempted/indexed/skipped/deleted, chunks_indexed, errors
    await store.close()

asyncio.run(main())
```

`IngestStats` is typed: `files_attempted`, `files_indexed`, `files_skipped`,
`files_deleted`, `chunks_indexed`, `errors` (per-source, phase-tagged).
Chunking/embedding/store errors are **collected**, never raised out of the
pool.

The document role (`prompt_role`) is resolved by the pipeline — document
content is never embedded with the query instruction.

### Search

```python
from embeddy import resolve_instruction, search_hybrid
from embeddy.index.base import SearchFilters

query_vec = (await provider.encode(
    ["how do tokens expire?"],
    instruction=resolve_instruction(provider.model_name, "query"),
))[0]
result = await search_hybrid(
    store,
    collection="acme",
    query_text="token expiry",
    query_vector=query_vec,
    top_k=10,
    mode="rrf",                 # or "weighted"
    filters=SearchFilters(content_types=("markdown",)),
    reranker=reranker,          # optional CrossEncoderReranker / HTTPReranker
)
for hit in result.results:
    print(hit.score, hit.metric, hit.chunk_id, hit.content[:80])
```

- Vector leg: cosine (metric-correct; vec tables declare `distance_metric=cosine`).
- FTS leg: FTS5 BM25 (porter + unicode61); queries are escaped/quote-wrapped
  by default, `raw=True` opts into verbatim syntax.
- Fusion: RRF (k=60) default, weighted (min-max normalized) alternative.
  Scores carry their metric semantics (`result.metric`).
- `total_results` = unique candidates across both legs **before** fusion
  truncation.
- Filters compile to SQL **pre-filters** (never post-filter over-fetch).
- Optional rerank stage: retrieve ~50, rerank to top_k.

## 3. Serving

```bash
pip install "embeddy[server]"
embeddy serve --host 0.0.0.0 --port 9000      # or: uvicorn embeddy.server:app
```

Or programmatically:

```python
from embeddy.server import create_app

app = create_app()   # DI seam: store=, provider=, reranker=, settings=, embedder=, pipeline=
```

Health is honest: `GET /health/live` = process up; `GET /health/ready` =
provider loaded AND store open (503 + reason otherwise). A failed provider
load does not crash the process — it serves and reports not-ready.

Routes:

- `POST /v1/embeddings` — OpenAI-compatible (`{"input", "model"?, "instruction"?}`)
- `POST /v1/rerank` — TEI/Jina shape (`{"query", "texts", "top_n"?}`)
- `POST /api/v1/search` | `/similar` | `/rerank`
- `POST /api/v1/ingest/text` | `/file` | `/dir` | `/reindex` | `/delete` | `/sync`
- `POST /api/v1/collections`, `GET /api/v1/collections`, `GET /api/v1/collections/{collection}`
- `GET /api/v1/chunks`, `POST /api/v1/chunks/search`

Errors are structured: `{"error": {"type", "message", "detail"}}` with the
mapping `400/404/413/422/500/501/503` documented in `embeddy/server.py`'s
module docstring.

## 4. The client

```python
from embeddy.client import EmbeddyClient

async with EmbeddyClient("http://127.0.0.1:9000") as client:
    stats = await client.ingest_text("hello", "acme")
    result = await client.search("hello", "acme")
```

`EmbeddyClient` mirrors every server route and shares the request builder
with `HTTPProvider` — one wire protocol, so the client and server cannot
drift. Non-2xx responses raise `ClientError` carrying the status and the
server's structured detail.

## 5. Configuration

One precedence: **CLI > file > env > defaults** (via pydantic-settings; the
dotenv file sits between CLI options and env vars). Sections are `embedder`,
`pipeline`, `server` — every field is read by a code path (CONCEPT §3.8,
"config = implementation"). See `docs/config.md` for the full reference.

```bash
EMBEDDY_EMBEDDER_MODEL=microsoft/harrier-oss-v1-0.6b
EMBEDDY_EMBEDDER_EMBEDDING_DIMENSION=512
EMBEDDY_PIPELINE_CONCURRENCY=2
EMBEDDY_SERVER_MAX_TOP_K=25
embeddy info --env-file .env
```

## 6. Quality gate

A deterministic, offline retrieval eval runs in the default test suite and is
the documented pre-release gate (thresholds mean nDCG@10 ≥ 0.50, recall@10 ≥
0.80 on the fixed 20-doc corpus). See `docs/retrieval-eval.md`.

## 7. Status and roadmap

- M1–M5 complete: keystone e2e, chonkai v1, real providers + MRL, storage &
  search core + pipeline, server/client/CLI.
- M6 (this phase): packaging/docs/benchmarks — see `CHANGELOG.md`.
- Phase 8 (post-v1, **not implemented**): Qdrant adapter (the `qdrant`
  extra installs `qdrant-client` but no adapter ships yet), integration
  adapters (Haystack/LlamaIndex).
