# Blue Sky Rewrite Concept — chonkai + embeddy

**Date**: 2026-07-31
**Status**: Concept (researched, decisions locked, no code)
**Scope**: Complete greenfield rewrite of embeddy as two distributions in one repo

---

## 1. Executive summary

Rewrite embeddy as **two libraries in one repository**:

1. **chonkai** — document processing (ingest + parse + chunk). Pure document layer,
   zero knowledge of embeddings or search.
2. **embeddy** — embed + store + search + serve. The retrieval workflow library,
   consuming chonkai.

The rewrite is driven by a code review that found the current library's core
function (embedding) unimplemented, several advertised features inert, and its
packaging broken. The rewrite follows a keystone-protocol architecture built on
best-in-class open source: sentence-transformers for embedding, Docling for
parsing, semchunk for text chunking, tree-sitter for code chunking, and
sqlite-vec + FTS5 (with a Qdrant adapter) for storage.

Three decisions define the shape:

- **The embedding protocol is the keystone** — a thin, typed contract that model
  adapters implement. Everything else conforms to it.
- **MRL-capable, but not required** — dimension truncation is a per-model
  capability, exposed through a model registry. Non-MRL models are first-class.
- **The retrieval core stays owned** — chunking and parsing are delegated
  wholesale to best-in-class libraries; the retrieval core (dedup, scoring,
  fusion, typing) is written by us, because that is where the opinionation lives.

---

## 2. Why a rewrite (motivation from code review)

The current embeddy (v0.3.12, 520 passing tests) has a solid skeleton but
fatal gaps:

| # | Finding | Severity |
|---|---------|----------|
| C1 | LocalBackend is a stub — `_load_model_sync`/`_encode_sync` raise `NotImplementedError`. No transformers/torch/qwen-vl-utils imports exist. No code path ever calls `load()`. | Critical |
| C2 | MRL truncation never fires: both backends report `dimension == config.embedding_dimension`, so the truncation condition is always false. Tests mask it by mocking `backend.dimension = 2048`. | Critical |
| C3 | sqlite-vec vec0 tables use the default **L2** metric (verified empirically), yet scores are computed as `1.0 - distance` and documented as cosine. Score values and `min_score` semantics are wrong. | Critical |
| C4 | `import embeddy` fails without fastapi/httpx (optional deps), while torch/transformers/docling are mandatory core deps. The "thin client" claim is false. | Critical |
| C5 | RemoteBackend POSTs `/encode`; the server exposes `/api/v1/embed`. The two cannot interoperate. | Critical |
| H1 | `SearchFilters.metadata_match` is a silent no-op in both KNN and FTS. | High |
| H2 | `find_similar` embeds chunk content with the *query* instruction instead of the document instruction. | High |
| H3 | `python_granularity` is dead code (identical if/else branches); decorators are orphaned from functions; methods never extracted; `parent` never populated. | High |
| H4 | Markdown chunker treats `# comments` inside fenced code blocks as headings (verified). | High |
| H5 | `max_tokens` is only honored by TokenWindowChunker, which is never auto-selected. Oversized chunks silently truncate at embed time. | High |
| H6 | Chunking errors escape the pipeline's error-collection design (embed/store errors are collected, chunk errors propagate). | High |
| H7 | `reindex_file` deletes old chunks before re-embedding — data loss on failure. | High |
| H8 | `PipelineConfig.concurrency` is dead — `ingest_directory` is strictly sequential. | High |
| H9 | `load_config_file` documents env-var override but never applies it; `EmbedderConfig.from_env()` is never called. | High |
| H10 | Server: CORS config never applied; collection `metadata` dropped everywhere; `/embed/query` ignores its `instruction` field. | High |
| M1–M12 | Health endpoint lies; unsanitized FTS queries; post-filter over-fetch recall hole; per-collection (not per-source) dedup; `total_results` meaningless; struct format-string per vector; zero-vector handling; CLI lifecycle; 422 error shape; doc/spec drift (0.3.11 vs 0.3.12); UTF-8-only reads; dead `SimilarityScore` operators. | Med |

The skeleton worth keeping: per-collection virtual tables, dependency-injection
seams (`create_app`, `_build_deps`), chunker factory pattern, pure-function RRF
fusion (k=60), exception hierarchy + server error mapping, Pydantic config
validation, WAL + `to_thread` bridging, and the test discipline (520 tests, real
sqlite-vec in-memory coverage).

---

## 3. Architecture principles

1. **The embedding protocol is the keystone.** A typed contract that model
   adapters implement. The pipeline and index depend on the protocol, never on a
   concrete model.
2. **One wire protocol.** OpenAI-compatible `/v1/embeddings` (+ rerank). Every
   HTTP consumer speaks it; TEI, Infinity, vLLM, Ollama, OpenAI are drop-in
   upstreams. No parallel incompatible clients (the C5 class of bug is
   structurally impossible).
3. **Sources are first-class.** `sources(id, collection_id, path, content_hash,
   size, mtime)` rows with `UNIQUE(collection_id, path)` and
   `chunks.source_id` FK `ON DELETE CASCADE`. Dedup, reindex, delete, and
   incremental sync are all source operations. Reindex = atomic swap, never
   delete-then-reingest. Every store carries a schema `user_version` stamp
   (see §12) so schema evolution during development never silently corrupts.
   Source operations are part of the storage protocol (§5.4), not a
   sqlite-only side layer — every backend implements them.
4. **The index is typed and metric-honest.** Typed records end to end, never
   `dict[str, Any]`. Vec tables declare their metric. Scores carry their metric
   semantics. Filters compile to SQL (pre-filter, never post-filter over-fetch).
5. **Chunking invariants are enforced once.** A `ValidatedChunker` wrapper
   guarantees: non-empty content, `max_tokens` respected (token-accurate),
   line ranges present. Chunkers stay simple; the contract doesn't drift.
6. **Concurrency is real.** Bounded worker pool with backpressure for ingest;
   overlap of read/chunk/embed/write phases. Progress via async iterator /
   callback.
7. **The server is a thin adapter that owns lifecycle.** FastAPI lifespan opens
   store, loads provider, closes on shutdown. `/health/live` (process) and
   `/health/ready` (provider loaded, store open). CORS applied if configured.
8. **Config = implementation.** No config field exists unless a code path reads
   it. One precedence: CLI > file > env > defaults (via pydantic-settings).
9. **Fusion is pure and typed.** RRF (k=60) default; weighted as alternative.
   Pure functions over typed ranked results. Optional rerank stage after fusion.
10. **Packaging is architecture.** Lazy imports; extras split; core importable
    with zero extras.

---

## 4. chonkai — document processing library

### 4.1 Scope

```
Path / URL / raw text
  → Ingestor          (content-type detection, file reading, Docling routing)
  → Chunker registry  (content type / strategy → chunker)
  → ValidatedChunker  (invariants)
  → list[Chunk]       (typed records)
```

Owns: `Chunk`, `IngestResult`, `SourceMetadata` models; content-type detection;
file ingestion; Docling bridge; all chunkers; token counting; the chunker
factory; the `ValidatedChunker` invariant wrapper.

**Does not own**: embeddings, storage, search, pipeline orchestration, server,
client, CLI. Must never import embeddy.

### 4.2 Wholesale dependencies

| Concern | Library | Why |
|---------|---------|-----|
| Rich-doc parsing | **Docling** | Best-in-class PDF/DOCX/HTML. It *is* the parser; chonkai bridges to its `Document` model. |
| Text chunking | **semchunk** | Token-accurate, semantic boundaries, offsets, overlap. |
| Code chunking | **tree-sitter** + **tree-sitter-language-pack** | 306 precompiled grammars, MIT. `get_parser()` for every code content type. |
| Token counting | **tiktoken** / HF **tokenizers** | Token-accurate counting for chunk budgets. |

### 4.3 The chunker model and invariants

```python
class BaseChunker(ABC):
    def chunk(self, ingest: IngestResult, budget: ChunkBudget | None = None) -> list[Chunk]: ...

class ValidatedChunker(BaseChunker):   # wraps any chunker
    # guarantees:
    #   - content non-empty
    #   - len(tokens(content)) <= budget.max_tokens
    #   - start_line / end_line present
    #   - chunk_type in known vocabulary
    #   - parent populated where the strategy knows it
```

The `ChunkBudget` is supplied by the *caller* (embeddy passes
`model.context_length - headroom`). chonkai owns mechanics; embeddy owns policy.
This keeps the cross-package contract (chunk size vs model context) explicit and
versionable.

### 4.4 Tree-sitter code chunking (the differentiator)

Verified empirically (tree-sitter-language-pack 1.13.7, `process()` API). The
API is **not** what an early read of the docs suggests — verified by direct
probing on 2026-07-31:

- `process(source: str, config: ProcessConfig)`. **`ProcessConfig(language=...)`
  with default `chunk_max_size=None` returns ZERO chunks.** Chunking is
  **size-windowed**, not semantic-per-definition: with a large `chunk_max_size`
  a whole file collapses to one chunk; with a small one a single function is
  split across many chunks. There is no `granularity=function/class/module`
  selector on this API.
- A `CodeChunk` exposes `content`, `start_byte`/`end_byte`,
  `start_line`/`end_line`, and `metadata` (a `ChunkContext`). The fields we want
  — `symbols_defined`, `context_path`, `node_types`, `has_error_nodes`,
  `comments`, `docstrings` — live on `chunk.metadata`, **not** on the chunk, and
  are per-window (many symbols per chunk / one symbol across many chunks), not a
  clean one-definition-per-chunk mapping.
- For true per-definition chunking, use the separate top-level `structure`
  (`list[StructureItem]`) each carrying `name`, `kind`, `span`, `body_span`,
  `decorators`, `children`, `signature`, `visibility` — this is where decorators
  stay attached and where function/class/module granularity actually comes from.
  Enable it with `ProcessConfig(structure=True, symbols=True, docstrings=True)`.
- Syntax-error recovery: broken code still parses; `metadata.has_error_nodes`
  flags it — no paragraph fallback (fixes the old SyntaxError path).
- **Grammars are fetched at runtime** by a DownloadManager (probing hit
  `Download error: Language not available for download`). This means a
  cache-dir, offline/air-gapped behavior, and checksum verification are
  operational concerns (§9.3), despite "306 precompiled grammars."

**Decision (confirm in the Phase-0 spike):** build the code chunker on
`structure` (per-definition, decorators attached, granularity real →
implements the dead `python_granularity` config, fixes H3), using
`chunk_max_size` windowing only to split an oversized single definition (fixes
H5). Do **not** rely on `process()`'s default chunk stream.

Granularity (`function` / `class` / `module`) maps to `StructureItem.kind`
selection, not to a `process()` config flag.

Risk: language-pack is young (441 stars, repo now `xberg-io/…`) with a moving
high-level API. Mitigate with a thin adapter and a pinned version; fallback is
the stable raw tree-sitter API (`get_parser` + node walking).

### 4.5 Markdown and text chunking

- Markdown chunker must track fenced code blocks (fixes H4) and populate
  `parent` (heading hierarchy).
- semchunk provides token-accurate splitting with overlap for generic text.
- Paragraph merging retained (configurable `min_tokens`).

### 4.6 Packaging

- PyPI: `chonkai` (naming — see §9.4 for availability conflict).
- Extras: `chonkai[docling]`, `chonkai[treesitter]`, `chonkai[semchunk]` (or
  semchunk in core — it is tiny), `chonkai[tokenizers]`.
- Core importable with minimal deps; heavy parsers lazy-loaded.

### 4.7 Public API sketch

```python
from chonkai import Ingestor, get_chunker, ChunkBudget
ingest = await Ingestor().ingest_file("src/foo.py")      # IngestResult
chunker = get_chunker(ingest.content_type, config, budget=ChunkBudget(max_tokens=512))
chunks: list[Chunk] = chunker.chunk(ingest)
```

---

## 5. embeddy — embed + store + search + serve

### 5.1 Embedding protocol (the keystone)

```python
# Core types (defined once, used everywhere — no dict[str, Any]):
EmbedInput = str | ImageInput          # text or image (multimodal)
Vector     = np.ndarray                # float32, shape (dimension,), L2-normalized

class EmbeddingProvider(Protocol):
    dimension: int                 # RESOLVED dimension (native, or MRL-truncated) — a fact, never a free config knob
    context_length: int            # model context — drives chunk budget
    model_name: str
    async def encode(self, inputs: list[EmbedInput],
                     instruction: str | None = None) -> list[Vector]: ...
```

**Role → instruction ownership.** `encode` takes a *resolved* instruction
string, never a role. The caller (pipeline / search / server) resolves
`role → instruction` via the registry (`ModelSpec.instructions[role]`) and
passes the string. This keeps the provider dumb and makes the H2 class of bug
(document content embedded with the query instruction) a typed, testable
contract — add a regression test that `find_similar` resolves the *document*
role.

Implementations:

- **LocalProvider** — sentence-transformers adapter. One line per model; gives
  prompts (instruction-awareness), MRL via `truncate_dim`, multimodal, batching.
- **HTTPProvider** — OpenAI-compatible `/v1/embeddings` (+ rerank endpoint).
  Speaks the same wire protocol as the server's embed route and every serving
  layer (TEI, Infinity, vLLM, Ollama, OpenAI).

Model facts (`dimension`, `context_length`, `mrl_range`) come from a **model
registry** (§6), never from config.

### 5.2 MRL policy — capable, not required

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str
    native_dimension: int
    mrl_range: range | None   # None = not MRL-capable
    context_length: int
    instructions: dict[str, str]   # role -> prompt_name / prompt
    license: str

def resolve_dimension(spec: ModelSpec, requested: int | None) -> int:
    if requested is None:
        return spec.native_dimension
    if spec.mrl_range is None:
        if requested != spec.native_dimension:
            raise ValidationError(f"{spec.id} is not MRL-capable; "
                                  f"embedding_dimension must be {spec.native_dimension}")
        return requested
    if requested not in spec.mrl_range:
        raise ValidationError(f"{spec.id} supports MRL {spec.mrl_range.start}-{spec.mrl_range.stop-1}")
    return requested
```

- `embedding_dimension: int | None = None` — `None` means native.
- Non-MRL model + wrong dimension → explicit config-time error (the C2 class of
  silent no-op is impossible).
- One truncation point in the provider post-process, keyed on model facts.
  **Truncation is followed by L2 re-normalization** — a sliced MRL vector is not
  unit-norm, and cosine scoring assumes unit vectors.
- Collection records the *resolved* dimension; vec table dimension and vectors
  can never disagree.
- MRL benefits (why the capability): storage linear in dims, faster KNN
  (dims-bound), smooth quality/cost curve, coarse-to-fine retrieval, retroactive
  dimension tuning via slicing.

### 5.3 Reranking

Sibling protocol: `RerankerProvider` (local CrossEncoder / remote rerank
endpoint). Optional pipeline stage after fusion: retrieve ~50, rerank to top_k.
The single largest retrieval-quality lever.

### 5.4 Storage — the `Searchable` protocol

```python
class Searchable(Protocol):
    async def add(self, collection, chunks, vectors): ...
    async def delete(self, collection, ids): ...
    async def search_vector(self, collection, query, filters, top_k) -> list[ScoredDocument]: ...
    async def search_fts(self, collection, query, filters, top_k) -> list[ScoredDocument]: ...
    async def stats(self, collection) -> CollectionStats: ...
    # Source operations are part of the protocol — every backend implements them,
    # so the Qdrant adapter has a defined contract (not a sqlite-only side layer):
    async def upsert_source(self, collection, source: SourceMetadata) -> SourceId: ...
    async def get_source(self, collection, path) -> SourceMetadata | None: ...
    async def reindex_source(self, collection, source, chunks, vectors): ...  # atomic swap
    async def delete_source(self, collection, source_id): ...                 # cascade
    async def list_sources(self, collection) -> list[SourceMetadata]: ...     # for incremental sync diff
```

`StoredChunk`, `ScoredDocument` (carries `.metric`), `CollectionStats`,
`SourceMetadata`, `SourceId` are typed records — no `dict[str, Any]` crosses
this boundary.

Two backends:

1. **Default: sqlite-vec + FTS5** (zero-config). Metric-correct vec tables
   (`distance_metric=cosine` — verified syntax in sqlite-vec 0.1.9). Metadata
   filters compiled to SQL pre-filters (join against chunks table before KNN, or
   vec0 auxiliary columns). Sources table + atomic swap reindex. FTS5 for BM25
   with porter+unicode61. Async via **aiosqlite**.
2. **Scale: Qdrant adapter.** Dense + sparse vectors, payload filters,
   quantization. One config line (`store: qdrant://...`).

LanceDB is the documented spike candidate to replace sqlite-vec as default
(embedded vector+FTS+hybrid in one package) — prototype before committing; the
pyarrow footprint and Lance file-format commitment are the costs.

### 5.5 Search & fusion

- Vector: cosine (metric-correct), typed scores.
- Fulltext: FTS5 BM25.
- Hybrid: RRF (k=60) default; weighted (min-max normalized) alternative.
- Pure functions over typed ranked results — independently testable.
- Optional rerank stage (§5.3).
- `total_results` means something: report the count *before* truncation, or
  document it as "returned results."

### 5.6 Pipeline / orchestration

- Bounded worker pool (`asyncio.Semaphore` + queue) — the `concurrency` config
  finally works.
- Overlap read/chunk/embed/write phases; flat memory regardless of corpus size.
- Progress: async iterator or callback per file.
- Source-level ops: ingest (dedup by source), reindex (atomic swap), delete
  (cascade), incremental sync (new/modified/deleted diff on `sources`).
- Errors collected into typed `IngestStats` — including chunking errors.

### 5.7 Server

- FastAPI factory; lifespan owns store init + provider load + shutdown.
- `GET /health/live` and `GET /health/ready` (provider loaded, store open).
- Routes: embed (OpenAI-compatible shape), search (+ similar + rerank), ingest,
  collections, chunks.
- Error map: `ValidationError` → 400; other `EmbeddyError` → 500; FastAPI 422
  mapped to the structured error shape.
- CORS applied from config when set.

### 5.8 Client

- httpx-based; mirrors the server exactly (one protocol, §7).
- `EmbeddyClient` and the HTTP provider share request-building code.

### 5.9 Config

- **pydantic-settings**: one source of truth for CLI > file > env > defaults.
- Section per layer (`embedder`, `store`, `chunk`, `pipeline`, `server`),
  but only fields a code path reads.
- Model selection is a registry key, not a pile of model knobs.

---

## 6. Model registry (researched 2026-07-31)

Aggregated from the MTEB results dataset (mteb/results, 8.6M rows) + HF model
cards. Raw-mean aggregates; the official leaderboard remains canonical.

### 6.1 Default registry

| Role | Model | Native dim | MRL | Context | License | Notes |
|------|-------|-----------|-----|---------|---------|-------|
| **Text default** | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32–1024 | 32,768 | Apache-2.0 | **Default**: proven, ungated, ST+TEI native, MRL-capable (showcases the flagship MRL feature), same family as the multimodal default |
| High-quality text option | `microsoft/harrier-oss-v1-0.6b` | 1024 | — | 32,768 | MIT | Higher MTEB v2 per size (69.0) but newer/less-proven, non-MRL, decoder-only last-token pooling — **verify ST pooling config before adopting**. Not the default (see §9.5) |
| Multimodal default | `Qwen/Qwen3-VL-Embedding-2B` | 2048 | 64–2048 | 32,768 | Apache-2.0 | Only fully-open multimodal with ST support (local-only path) |
| High-quality text (large) | `microsoft/harrier-oss-v1-27b` | 5376 | — | 32,768 | MIT | retr ~0.874; watch storage (5376 dims) |
| Small/fast | `microsoft/harrier-oss-v1-270m` | 640 | — | 32,768 | MIT | Mobile-class |
| Sparse hybrid (Qdrant path) | `BAAI/bge-m3` | 1024 | — | 8,192 | MIT | Dense+sparse |
| API fallback | voyage-3 / gemini | via API | — | — | commercial | OpenAI-compatible protocol |

### 6.2 Research notes

- **harrier-oss-v1** (Microsoft): MIT, ungated, ST+TEI native, 32K ctx,
  decoder-only last-token pooling, L2-normalized, instruction-aware via
  `prompt_name`. The current open-source leader in the aggregate.
- **Retrieval specialists**: tencent/KaLM-Embedding-Gemma3-12B (license:other,
  retr ~0.879), Bytedance/Seed1.6 (gated), nvidia/Nemotron-3-Embed-1B/8B
  (NVIDIA license, retr ~0.844/0.853), zeroentropy/zembed-1 (Apache-2.0),
  perplexity-ai/pplx-embed-v1-4b (MIT).
- **EmbeddingGemma** (Google): 300M public metadata but downloads **gated**
  (`gated: manual`, `license:gemma` — requires Google license acceptance + HF
  token). 768-dim, MRL 128–768, **2K context** (mobile-first). Registry
  alternative, not default; the `context_length` guard makes its short context
  safe by construction. 1B/3B gated.
- **Qwen3-Embedding family**: Apache-2.0, ungated, 32K ctx, MRL, ST+TEI native.
  The MRL-capable default family.
- **Commercial APIs** (via the OpenAI-compatible protocol): voyage-3-large
  (retr ~0.863), gemini-embedding.

---

## 7. Wire protocol (one protocol, two roles)

```
OpenAI-compatible:
  POST /v1/embeddings        {"input": [...], "model": ...} -> {"data": [{"embedding": [...]}]}
  POST /v1/rerank             (or TEI/Infinity rerank shape)

embeddy server (extends):
  POST /api/v1/search         {"query", "collection", "top_k", "mode", "filters", ...}
  POST /api/v1/ingest/...     POST /api/v1/collections/...  POST /api/v1/chunks/...
```

- The HTTPProvider and EmbeddyClient share the client-side code.
- The server embeds via its configured provider (local or proxied remote).
- TEI / Infinity / vLLM / Ollama / OpenAI are all valid upstreams.

**Scope of "one protocol" (be honest — OpenAI-compat is not universal):**

- OpenAI `/v1/embeddings` cleanly carries **text, dense, single-vector**
  embeddings only. That is the interoperable core and is enough to kill the
  C5 class of bug.
- **Instructions**: OpenAI embeddings has no instruction field. The HTTPProvider
  sends it via `extra_body`; a generic upstream that ignores `extra_body` will
  silently drop it. Providers declare whether they honor instructions.
- **Multimodal** (image inputs, Qwen3-VL) and **learned-sparse** (bge-m3): not
  expressible over OpenAI `/v1/embeddings`. In v1 these are **local-provider
  only**; the remote path is text-dense.
- **Rerank is not an OpenAI endpoint.** The reranker uses the TEI/Jina/Cohere
  rerank shape, not `/v1/rerank` (there is no such OpenAI standard).
- Consequence for Phase 3: "local and remote produce interchangeable vectors" is
  **ranking-parity within tolerance**, not bitwise/numeric equality (ST vs a
  remote TEI will never match exactly).

---

## 8. Repo layout & packaging

Canonical layout is the **uv workspace** (matches the implementation plan §2 —
this supersedes the earlier flat `src/` sketch):

```
repo (uv workspace, two distributions)
├── pyproject.toml                    # workspace root: members + shared tool config
├── uv.lock
├── packages/chonkai/                 # document processing  (PyPI: chonkai)
│   ├── pyproject.toml
│   └── src/chonkai/
│       ├── ingest/                  #   content-type detection, file reading, docling bridge
│       ├── chunkers/                #   treesitter, markdown, semchunk, paragraph, docling
│       ├── validated.py             #   ValidatedChunker (invariants)
│       └── models.py                #   Chunk, IngestResult, SourceMetadata
├── packages/embeddy/                 # embed + store + search + serve  (PyPI: embeddy)
│   ├── pyproject.toml               #   depends on chonkai
│   └── src/embeddy/
│       ├── protocol/               #   EmbeddingProvider, Searchable, RerankerProvider
│       ├── providers/              #   local (sentence-transformers), http (OpenAI-compatible)
│       ├── registry.py             #   ModelSpec registry + resolve_dimension
│       ├── index/                  #   sqlite (default), qdrant (adapter), sources
│       ├── search.py               #   pure fusion + rerank stage
│       ├── pipeline.py             #   bounded concurrency, source ops
│       ├── server.py               #   FastAPI, lifespan, health
│       └── client.py               #   httpx client
├── benchmarks/  tests/  docs/
```

`requires-python = ">=3.11"` (not 3.13 — 3.13 is too new to force on a library;
relax unless a hard 3.13-only feature is used).

Rules:

- **One-way dependency, enforced**: chonkai never imports embeddy.
- Shared CI initially; versions may decouple once the split proves itself.
- extras (tree-sitter + semchunk live in **core** — both are light; only
  Docling and arbitrary-HF tokenizers are extras):
  - `chonkai[docling]`, `chonkai[tokenizers]`
  - `embeddy[local]` (sentence-transformers), `embeddy[server]` (fastapi,
    uvicorn), `embeddy[client]` (httpx), `embeddy[qdrant]`
- `import chonkai` and `import embeddy` both work with zero extras.
- All heavy imports lazy (torch, transformers, docling, tree-sitter, fastapi).

---

## 9. Open questions & risks

### 9.1 Naming — resolved: **chonkai**

Verified 2026-07-31: the original working name **`chonky` is taken** on PyPI by
an existing semantic text chunking library ("intelligently segments text into
meaningful semantic chunks using a fine-tuned transformer model") — a direct
competitor.

**Decision: the document-processing library is named `chonkai`** (verified free
on PyPI). Also verified free: `chonk`, `chonkaiio`, `chonkai-core`,
`chonkaitext`, `chonkai-docs` — available as fallbacks if needed. The earlier
shortlist (**margin**, **sliver**, **cleve**, **fissure**, **sectio**,
**docket**) remains available if the family naming ever changes.

Remaining checks before publishing: confirm the import name `chonkai` does not
collide with a transitive dependency, and check GitHub/npm for branding clashes.

### 9.2 Model strategy moves fast

harrier was not on the radar months ago; the leaderboard shifts. The registry +
protocol is the hedge: model choice is config, not code. Review the default
registry before each release.

### 9.3 Dependencies to watch

- `tree-sitter-language-pack`: young, fast-moving high-level API — pin and
  wrap (raw tree-sitter as fallback).
- `sqlite-vec`: pre-v1 with announced breaking changes — isolate behind
  `Searchable`.
- `LanceDB`: 0.x API churn — prototype before any commitment.
- sentence-transformers: stable; pin `transformers>=4.51` (Qwen3) /
  `>=4.57` (Qwen3-VL path).

### 9.5 Default text model — Qwen3-0.6B, not harrier

harrier-oss-v1-0.6b scores higher on MTEB v2 (69.0) and is real (MIT, ungated,
1024/32k — verified on HF 2026-07-31), but it is new, single-source-corroborated,
non-MRL, and uses decoder-only last-token pooling whose sentence-transformers
config we have not verified. The **default** is `Qwen/Qwen3-Embedding-0.6B`:
proven, Apache-2.0, MRL-capable (so the flagship MRL feature is *on* by default),
same family as the multimodal default. harrier stays a documented high-quality
option; promote it to default only after verifying its ST pooling and the score
in our own eval harness (§9.6).

### 9.6 Retrieval-quality eval harness (new — required for a "metric-honest" library)

A library whose thesis is honest, correct retrieval must prove retrieval quality
does not regress. Ship a small fixed-corpus harness (corpus + queries + qrels) that
reports **nDCG@k / recall@k** across model, chunker, and fusion changes, wired as a
CI gate (or at least a pre-release gate at M4/M6). This is separate from the
chunk-quality harness (§10.10) and from resource benchmarks.

### 9.7 M-findings disposition

The Critical (C1–C5) and High (H1–H10) findings are all fixed by design (see the
mapping through §3–§7). The Medium findings (M1–M12) must each be explicitly
dispositioned in the plan (fixed / deferred / won't-fix) so none is silently lost;
the plan carries that table.

**Explicitly out of scope (personal project on a trusted tailnet):** data
migration from v0.3.x (policy: no in-place migration — re-ingest from source; the
`user_version` stamp exists only to prevent self-inflicted schema breakage during
development) and server security hardening (auth / SSRF / path-traversal). The one
retained operational guard is **request/batch size limits** on the server — a
crash-prevention (OOM) control, not a security control.

### 9.8 Decisions deferred

- Coarse-to-fine multi-dimension retrieval (store one dim, slice others on
  demand) — future option, not in v1.
- MCP server for agents — future integration target.
- Integration adapters (chonkai chunks as Haystack components, embeddy behind a
  LlamaIndex vector-store interface) — the ecosystem escape hatch, built after
  the core is real.

---

## 10. Build order (vertical slice first)

1. **Keystone slice**: protocol + ModelSpec registry + one fake provider →
   metric-correct index → one chunker → one search. One end-to-end test:
   document in, result out, tiny fake model.
2. **chonkai v1**: ingest + content-type detection, ValidatedChunker, markdown +
   paragraph + semchunk chunkers; tree-sitter code chunker (all 10 types).
3. **Real providers**: sentence-transformers local adapter (harrier + Qwen3
   registry entries, MRL resolution); OpenAI-compatible HTTP provider.
4. **Storage**: sqlite-vec + FTS5 backend with SQL pre-filters and sources
   table; atomic reindex; aiosqlite.
5. **Search**: typed scores, RRF + weighted fusion, filters, rerank stage.
6. **Pipeline**: bounded concurrency, source ops, progress callbacks.
7. **Server + client**: lifespan, health live/ready, OpenAI-compatible embed
   route, search/ingest/collections/chunks routes; client.
8. **Config + packaging**: pydantic-settings, extras split, lazy imports.
9. **Scale path**: Qdrant adapter. **Spike**: LanceDB vs sqlite-vec at 1M docs.
10. **Polish**: docs, benchmarks (chunk-quality harness in chonkai), integration
    adapters.

---

## 11. Appendix: wholesale adoption analysis (2026-07-31)

**Adopt wholesale (the features):**

- sentence-transformers — embedding + reranking. Model-agnostic, prompts, MRL,
  multimodal, CrossEncoder rerankers.
- Docling — rich-document parsing (64k stars, extremely active).
- semchunk — token-accurate text chunking (used by Docling).
- tree-sitter + tree-sitter-language-pack — polyglot code chunking (306
  grammars, MIT).

**Own (the opinionation):**

- Retrieval core: metric-correct scoring, source-level dedup, MRL policy,
  typed records, fusion semantics.
- Orchestration: concurrency, source ops, lifecycle.

**Rejected as wholesale base:**

- txtai (9.12) — covers embeddy's entire scope incl. its own API server, but
  239 mandatory deps (torch/faiss/transformers), its own storage/scoring model,
  and our differentiators would fight its abstractions. Adopting it dissolves
  embeddy into a config file.
- Haystack (3.0) — cleanest framework, but imposes its Document/pipeline model;
  our chunkers and dedup become foreign objects. Viable base only if the product
  pivots to agents/LLM orchestration.
- LlamaIndex (0.14) — big surface, high churn, `nest-asyncio` in core; same
  abstraction-fight problem.
- LangChain — abstraction soup, highest churn.
- Chroma — competes with embeddy (brings its own server + onnxruntime).
- LanceDB — the strongest single-package *composable* candidate for
  storage+search (embedded vector+FTS+hybrid+filters). Spike before committing.
- duckdb-vss — experimental, watch only.

**Pattern**: build on primitives, then ship integration adapters to frameworks.
Framework users consume embeddy's value; embeddy is never hostage to a
framework's API churn.
