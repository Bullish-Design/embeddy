# Architecture — chonkai + embeddy

The greenfield rewrite architecture (CONCEPT.md / IMPLEMENTATION_PLAN.md §1–
§12). This is a **map of what exists** at M7, not a design wishlist.

## 1. Layering — two distributions, one-way dependency

```
chonkai   document processing (ingest + parse + chunk + invariants)
   ↑  never imports embeddy (enforced: CI grep fails on `from embeddy` in chonkai/src)
embeddy   embed + store + search + serve; consumes the chonkai public API
```

Both packages live in one uv workspace (`packages/chonkai`,
`packages/embeddy`), ship as separate distributions, and are released
together initially (release train, `CHANGELOG.md`).

```
repo/
├── pyproject.toml            workspace root: members + shared ruff/ty/pytest config
├── packages/chonkai/         chonkai distribution (src/chonkai)
├── packages/embeddy/         embeddy distribution (src/embeddy)
├── tests/                    shared integration/contract/e2e tests
├── eval/                     retrieval-quality eval harness (M4 gate)
├── benchmarks/               Phase-7 harness + Phase-4 spike (NOT in testpaths)
├── docs/                     USER_GUIDE / INTEGRATION / ARCHITECTURE / config / cli
├── scripts/                  install-matrix + lazy-import CI checks
└── .github/workflows/ci.yml  lint / type / test / install-matrix / benchmarks
```

## 2. The frozen protocols (M4 freeze — changes require a decision record)

**`EmbeddingProvider`** (`embeddy/protocol/embedding.py`):

```python
class EmbeddingProvider(Protocol):
    dimension: int              # RESOLVED dimension (native or MRL) — a fact, never a knob
    context_length: int         # drives the chunk budget
    model_name: str
    async def encode(self, inputs: list[EmbedInput], instruction: str | None = None) -> list[Vector]
```

Callers resolve `role → instruction` via the registry and pass a **resolved
string** — providers never see a role (the H2 bug class is impossible by
construction). `Vector` is an L2-normalized float32 ndarray; `EmbedInput` is
`str | ImageInput` (multimodal is local-only in v1 — the OpenAI wire cannot
carry images).

**`RerankerProvider`** (`embeddy/protocol/rerank.py`): `CrossEncoderReranker`
(local) and `HTTPReranker` (TEI/Jina endpoint) implement it.

**`Searchable`** (`embeddy/index/base.py`) — including the **source
operations**, because the future Qdrant adapter must implement the same
contract (Phase 8):

```python
class Searchable(Protocol):
    async def add(self, collection, chunks, vectors): ...
    async def delete(self, collection, ids): ...
    async def search_vector(self, collection, query, filters, top_k) -> list[ScoredDocument]: ...
    async def search_fts(self, collection, query, filters, top_k) -> list[ScoredDocument]: ...
    async def stats(self, collection) -> CollectionStats: ...
    # source ops — every backend implements them:
    async def upsert_source(self, collection, source) -> SourceId: ...
    async def get_source(self, collection, path) -> SourceMetadata | None: ...
    async def reindex_source(self, collection, source, chunks, vectors): ...  # atomic swap
    async def delete_source(self, collection, source_id): ...                 # cascade
    async def list_sources(self, collection) -> list[SourceMetadata]: ...
```

Typed records everywhere — no `dict[str, Any]` crosses any protocol
(`StoredChunk`, `ScoredDocument` with `.metric`, `CollectionStats`,
`SourceMetadata`, `SourceId`).

The **chonkai public API** (chunkers, `ValidatedChunker`, `ChunkBudget`,
`Ingestor`, `get_chunker`, models) is also frozen at M4.

## 3. Storage — sqlite-vec + FTS5 (the default `Searchable`)

`SqliteStore` (`embeddy/index/sqlite.py`) — aiosqlite (never `to_thread`):

- **vec0** tables declare `distance_metric=cosine`; the collection records
  the resolved dimension and vectors can never disagree.
- **FTS5** (porter + unicode61); queries are escaped/quote-wrapped by default
  with an explicit `raw` opt-in.
- **Sources** table with `UNIQUE(collection_id, path)`; `chunks.source_id`
  FK `ON DELETE CASCADE`. Dedup is source-level (two identical files at
  different paths = two sources); reindex is an **atomic swap** in one
  transaction (never delete-then-reingest); sync is a diff on `sources`.
- **Pre-filters**: `SearchFilters` (`content_types`, `source_path_prefix`,
  `chunk_types`, `metadata_match`) compiles to SQL WHERE clauses joined
  before KNN/FTS — never post-filter over-fetch. vec0 auxiliary columns
  carry the filterable fields for in-scan pre-filtering.
- **`PRAGMA user_version`** stamp — guards dev-time schema breakage (there is
  deliberately no v0.3.x migration; policy is re-ingest from source).
- `create_collection` / `get_chunk` / `list_chunks` / `list_collections` /
  `count_fts` are **beyond-protocol extras** (the server's 501 is for stores
  that do not implement them).

### 3.1 The scale path — Qdrant (`embeddy/index/qdrant.py`, Phase 8 / M7)

`QdrantStore` implements the SAME frozen `Searchable` — dense + sparse
named vectors, payload filters, collection-level quantization, and the full
source-op set (sources are first-class on every backend). It is selected by
one config line: `store.url` (`EMBEDDY_STORE_URL`) → `build_store`
(`embeddy/index/factory.py`, mirrors `build_provider`) → the server
lifespan opens it. `None` keeps the sqlite `store_path` default — the bare
`app = create_app()` path is unchanged.

- **Mapping**: chunk/source ids hash to stable uuid5 point ids (qdrant
  rejects non-UUID string ids); originals live in the payload and are
  restored on every read. Sources live in a per-collection `__sources`
  companion collection (the sqlite `sources`-table analogue).
- **Filters**: `SearchFilters` compile to payload `Filter`s; the
  source_path_prefix uses per-point `path_prefixes` arrays (exact prefix —
  1.18 has no string-prefix condition). Pre-filters, so restrictive filters
  return full top_k (the M3 recall contract).
- **FTS**: no BM25 engine — pure-Python BM25 scan over payloads, FTS5
  negative-rank convention, `raw` a no-op (decision 0004).
- **Reindex**: no transactions — upsert-first then stale-delete; old chunks
  stay intact and queryable on failure (weaker than sqlite's one-txn swap;
  decision 0004).
- **Sparse**: plumbing ships (`add(..., sparse_vectors=)` +
  `search_sparse`, metric `SPARSE_DOT`); a real learned-sparse encoder is
  out of scope (decision 0004).
- **The store factory**: `parse_store_url` (pure, unit-tested) + async
  `build_store`; qdrant reachability is verified at open time so
  `/health/ready` reports not-ready honestly.

Decisions: `docs/decisions/0003` (LanceDB revisit — not adopted; Qdrant is
 the scale path) and `0004` (sparse status, reindex atomicity, the FTS
 path, quantization, the sync-client note).

## 4. Search — typed, metric-honest fusion

`search.py` (`search_hybrid` + `fuse_rrf`/`fuse_weighted`):

- Two legs in parallel: `search_vector` (cosine) + `search_fts` (BM25), each
  bounded by `retrieve_k` (default 50).
- Fusion: RRF (k=60) default; weighted (min-max normalized) alternative —
  pure functions over typed ranked results.
- Optional rerank stage after fusion (retrieve ~50, rerank to top_k).
- `min_score` is interpreted in each leg's metric semantics, never
  cross-metric.
- `total_results` = unique candidates before fusion truncation (cheap count).

## 5. Pipeline — bounded, overlapping, source-aware

`IngestPipeline` (`embeddy/pipeline.py`) is a pure consumer of the frozen
protocols:

- **Bounded worker pool** (`asyncio.Semaphore` + one task per file);
  `concurrency` bounds concurrently-in-flight files; read/chunk/embed/write
  phases overlap across files; memory stays flat (per-file, released after
  each store write).
- **Errors are collected** into typed `IngestStats` (phase-tagged
  `SourceError`s), never raised out of the pool — chunking errors included
  (fixes H6).
- **Source ops**: `ingest_text`, `ingest_file` (dedup), `ingest_directory`,
  `reindex` (atomic swap), `delete_source`, `sync` (incremental diff).
- **Chunk budget** comes from the caller: `budget.chunk_budget(provider
  .context_length)` — embeddy owns policy, chonkai owns mechanics.
- SourceId = hash of (collection, path) — content-independent, stable across
  re-ingests; chunk ids `f"{source_id}:{seq}"`.

## 6. Server — a thin adapter that owns lifecycle

`embeddy/server.py` is the **server-extra entry point** (fastapi/starlette at
module level by design). `create_app(store=, provider=, reranker=, settings=,
embedder=, pipeline=)` is the DI seam; the module-level `app = create_app()`
makes `uvicorn embeddy.server:app` work bare.

- **Lifespan** opens the store, builds + loads the provider, closes on
  shutdown. A provider load failure is recorded, not re-raised: the process
  keeps serving and reports not-ready (honest health).
- `/health/live` = process; `/health/ready` = provider loaded AND store open.
- One wire protocol (`/v1/embeddings` OpenAI-compat, `/v1/rerank` TEI/Jina) +
  the `/api/v1/*` extensions; shared request builders with the client.
- Size limits (`max_body_bytes`, `max_embed_inputs`, `max_top_k`) are OOM
  prevention only (auth/SSRF/path-traversal out of scope).
- CORS from config; structured error map (400/404/413/422/500/501/503).

## 7. Packaging — architecture as a contract

- **Extras split**:
  - chonkai core: pydantic, numpy, tree-sitter, tree-sitter-language-pack,
    semchunk, tiktoken; extras `docling`, `tokenizers`.
  - embeddy core: chonkai, pydantic, numpy, sqlite-vec, aiosqlite,
    pydantic-settings; extras `local` (sentence-transformers), `server`
    (fastapi/uvicorn/typer), `client` (httpx), `qdrant` (qdrant-client — the
    adapter lands in Phase 8).
- **Lazy imports**: heavy deps are imported only inside functions, or at
  module level in exactly two KNOWN entry-point modules (`server.py`,
  `cli.py`), or under `TYPE_CHECKING`. `scripts/check_lazy_imports.py`
  enforces this in CI (AST-based).
- **Zero-extras import**: `embeddy/__init__.py` never imports
  server/client/cli — `import embeddy` stays clean without extras
  (`scripts/check_install_matrix.py` verifies every extras row in a fresh
  venv).
- Wheels ship only `src/` packages (hatchling `packages = ["src/…"]`);
  verified contents contain no accidental heavy deps in core.

## 8. Config — config equals implementation

`embeddy/config.py` — pydantic-settings with a custom source order
**CLI(init) > file(dotenv) > env > secrets > defaults**. Every section
(`embedder`, `pipeline`, `server`) sets `extra="forbid"` and
`dotenv_filtering="match_prefix"` so a shared `.env` holds all sections. No
field exists unless a code path reads it (CONCEPT §3.8). `store`/`chunk`
sections land only when their consuming code paths exist.

## 9. Errors

`embeddy/errors.py` — `EmbeddyError` root + typed subclasses
(`ProviderError`, `HTTPProviderError`, `WrongDimensionError`, `RerankError`,
`ClientError`, `RegistryError`, `ModelNotLoadedError`…). The server maps
them: `ValidationError` → 400, other `EmbeddyError` → 500, FastAPI 422 →
structured shape, plus 404/413/501/503 for the specific cases (server module
docstring is the canonical map).

## 10. Quality gates (CI, all in the default suite)

- 463 tests: unit (per-package), integration (`tests/`, real sqlite/FTS5),
  contract (`tests/contract/`, client↔server), e2e keystone, and the **eval
  gate** (`eval/`, deterministic, offline).
- Coverage: core ≥90% (server 95%, client 97%, cli 100% at M5; total 96%).
- ty strict; ruff clean; one-way-dep grep; lazy-import audit; install matrix.
- Benchmarks (`benchmarks/`) are outside testpaths — never the default suite
  (docs/decisions/0002).

## 11. Roadmap (NOT implemented — do not document as features)

- Out of scope (not planned — `docs/decisions/0004`): a real learned-sparse
  encoder (bge-m3 + FlagEmbedding) behind the shipped sparse plumbing; a
  lexical index for the Qdrant FTS path (pure-Python BM25 today); an async
  qdrant client; integration adapters (Haystack/LlamaIndex — plan §10 lists
  them as optional post-v1).
- Server auth/SSRF/path-traversal: explicitly out of scope (trusted tailnet).
- MCP server, agents orchestration, distributed storage: non-goals for v1.

See also: `docs/decisions/0001` (default search backend), `0002` (benchmark
 tooling), `0003` (LanceDB revisit at M7), `0004` (Qdrant adapter
 decisions).
