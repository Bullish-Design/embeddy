# Implementation Plan — chonkai + embeddy Greenfield Rewrite

**Date**: 2026-07-31
**Source**: `CONCEPT.md` (architecture, decisions, model registry)
**Status**: Plan for approval. All work items derive from decisions in the concept.

---

## 0. How to read this plan

- **Phases** are sequential; milestones mark review gates.
- Each phase has: scope, work items (checkboxes), acceptance criteria, test plan.
- Dependencies are explicit (a phase may not start before its listed inputs).
- Open decisions that block a task are listed inline as `DECISION:`.

---

## 1. Overview & goals

Build two libraries in one repository:

| Package | Purpose | Depends on |
|---------|---------|-----------|
| **chonkai** | Document processing: ingest + parse + chunk, with guaranteed invariants | — (never imports embeddy) |
| **embeddy** | Embed + store + search + serve; typed, metric-honest retrieval workflow | chonkai |

Non-goals (v1): agents/LLM orchestration, MCP server, distributed storage,
training/fine-tuning, integration adapters for Haystack/LlamaIndex (post-v1).

---

## 2. Repo & tooling setup (Phase 0.1 — do first)

### Layout refinement (refines CONCEPT §8)

Use a **uv workspace** with two package members — the cleanest multi-distribution
setup for uv/hatchling:

```
repo/
├── pyproject.toml                # workspace root: members, shared tool config
├── uv.lock
├── packages/chonkai/
│   ├── pyproject.toml            # chonkai distribution
│   └── src/chonkai/
├── packages/embeddy/
│   ├── pyproject.toml            # embeddy distribution (depends on chonkai)
│   └── src/embeddy/
├── tests/                        # shared integration/e2e tests live here
├── benchmarks/
├── docs/
└── .github/workflows/ci.yml
```

### Work items

- [ ] Root `pyproject.toml`: `[tool.uv.workspace]` members `packages/chonkai`,
      `packages/embeddy`; shared `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]`.
- [ ] `packages/chonkai/pyproject.toml`: name `chonkai`, MIT, `requires-python >=3.11`
      (NOT 3.13 — too new to force on a library), core deps only (pydantic, numpy,
      tree-sitter, tree-sitter-language-pack, semchunk, tiktoken). Extras:
      `docling`, `tokenizers`.
- [ ] `packages/embeddy/pyproject.toml`: name `embeddy`, `requires-python >=3.11`,
      deps on `chonkai` + pydantic,
      numpy, sqlite-vec, aiosqlite, pydantic-settings. Extras: `local`
      (sentence-transformers), `server` (fastapi, uvicorn), `client` (httpx),
      `qdrant` (qdrant-client).
- [ ] Dev tooling pinned: ruff (lint+format), mypy strict, pytest + pytest-asyncio,
      coverage. CI: GitHub Actions on Python 3.13 — lint, type, test, coverage gates.
- [ ] Enforce the one-way dependency: CI job greps for `from embeddy` inside
      `packages/chonkai/src` (fails if found).
- [ ] Import smoke test in CI: fresh venv, `pip install ./packages/chonkai` and
      `pip install ./packages/embeddy` (no extras) — both `import` clean.

**Acceptance**: `uv run pytest` passes on an empty suite; mypy strict is green;
both packages import with zero extras.

---

## 3. Phase 1 — Keystone vertical slice

**Goal**: prove the architecture holds with one end-to-end path before building
breadth. (Concept §10, item 1.)

**Inputs**: Phase 0.1.

> **Protocol freeze note:** Phase 1 defines the protocols but only the
> *keystone* shape (fake provider + minimal store) is exercised here. Do NOT
> freeze `EmbeddingProvider` / `Searchable` / `RerankerProvider` at M1 — the
> source ops, filters, rerank, and multimodal paths that reshape them don't
> exist until Phases 3–5. **Protocol freeze moves to M4** (see §12). Async: use
> **aiosqlite from the start** here — do not build a sync/`to_thread` store and
> rewrite it in Phase 4.

### Work items

- [ ] `embeddy/protocol/types.py` — core typed records **defined once**:
      `EmbedInput` (`str | ImageInput`), `Vector` (float32 ndarray, L2-normalized),
      `StoredChunk`, `ScoredDocument` (carries `.metric`), `CollectionStats`,
      `SourceMetadata`, `SourceId`. No `dict[str, Any]` crosses any protocol.
- [ ] `embeddy/protocol/embedding.py` — `EmbeddingProvider` Protocol
      (`dimension`, `context_length`, `model_name`, `encode`). Caller resolves
      `role → instruction` via the registry and passes a resolved string.
- [ ] `embeddy/registry.py` — `ModelSpec` dataclass (id, native_dimension,
      mrl_range, context_length, instructions, license), `resolve_dimension`
      (exact logic from CONCEPT §5.2), minimal registry dict.
- [ ] `embeddy/providers/fake.py` — test-only `FakeProvider` (dimension 8,
      deterministic vectors). Marked `[test]`/dev-only.
- [ ] `embeddy/index/base.py` — `Searchable` Protocol (CONCEPT §5.4), **including
      the source operations** (`upsert_source`, `get_source`, `reindex_source`,
      `delete_source`, `list_sources`) so the Qdrant adapter has a defined
      contract. Source ops are part of the protocol, not a sqlite-only side layer.
- [ ] `embeddy/index/sqlite.py` — first slice (**aiosqlite**, not to_thread):
      - collections + chunks + **sources** tables (`sources` has
        `UNIQUE(collection_id, path)`; `chunks.source_id` FK `ON DELETE CASCADE`),
      - schema `PRAGMA user_version` stamp (guards self-inflicted schema breakage
        during development — there is no v0.3.x migration; re-ingest from source),
      - vec0 table with `distance_metric=cosine` (verified syntax, sqlite-vec 0.1.9),
      - FTS5 table (porter + unicode61),
      - typed records (`StoredChunk`, `ScoredDocument` — no `dict[str, Any]`),
      - `add` / `search_vector` / `search_fts` / `stats` only.
- [ ] `embeddy/search.py` — `fuse_rrf` / `fuse_weighted` as pure functions over
      typed ranked results (k=60 default).
- [ ] `chonkai` minimal: `IngestResult`, `Chunk`, `SourceMetadata` models;
      paragraph chunker; `ValidatedChunker` (invariants: non-empty, token
      budget, line ranges).
- [ ] **E2E test** (`tests/e2e/test_keystone.py`): text → ingest → chunk →
      fake vectors → store → hybrid search → assert result. Tagged `[e2e]`.

**Acceptance**: e2e green; score semantics cosine-correct; no `dict[str, Any]`
leaks into public results; mypy strict clean; one-way dependency enforced.

---

## 4. Phase 2 — chonkai v1 (document processing)

**Goal**: chonkai is feature-complete and standalone-testable. embeddy's pipeline
consumes only its public API.

**Inputs**: Phase 1 (models + ValidatedChunker skeleton).

### Work items

- [ ] **Ingest** (`chonkai/ingest/`):
      - content-type detection: extension map for all 10 code types + markdown/
        rst/generic; docling-routed extensions (pdf/docx/html/img/tex...);
      - file reading with encoding fallback (utf-8 → latin-1 → errors collected);
      - Docling bridge (lazy import, `chonkai[docling]` extra, per-instance
        converter reuse);
      - `compute_content_hash`; `SourceMetadata` population (size, mtime, hash).
- [ ] **Token counting** (`chonkai/tokens.py`): tiktoken default +
      `tokenizers` extra for arbitrary HF models; `ChunkBudget` dataclass.
- [ ] **Text chunkers**: paragraph (short-merge), markdown (code-fence-aware —
      fixes CONCEPT H4; heading hierarchy → `parent`), semchunk-based
      token-accurate chunker (overlap, offsets).
- [ ] **Tree-sitter code chunker** (`chonkai/chunkers/treesitter.py`) — build on
      the **`structure` list**, NOT `process()`'s default chunk stream (verified
      2026-07-31: default `ProcessConfig` returns 0 chunks; `process()` chunking is
      size-windowed, and `symbols_defined`/`context_path`/`node_types` live on
      `chunk.metadata`, per-window, not one-per-definition — see CONCEPT §4.4):
      - grammar registry: ContentType → language (python, javascript,
        typescript, rust, go, c, cpp, java, ruby, bash) — all **bundled**
        (offline-safe), no cache dir needed for v1;
      - use `ProcessConfig(structure=True, symbols=True, docstrings=True)`; map
        `StructureItem` (`.name` → name, `.kind` → chunk_type + granularity,
        nesting via `children` → parent, `.span`/`.body_span` → byte/line
        ranges); no `context_path` on `StructureItem` (per-window only);
      - **decorators are NOT on `StructureItem`** (`.decorators`/`.visibility`/
        `.signature`/`.doc_comment` are inert in 1.13.7): recover them from the
        raw parse tree (Python `decorated_definition` wrapper; Rust
        `attribute_item` siblings) and prepend to chunk content;
      - granularity selection (function/class/module) = `StructureItem.kind`
        filter — implements the dead `python_granularity` config;
      - **`chunk_max_size` is a BYTE budget** — the `ValidatedChunker` token
        invariant remains the contract; oversized single definition → split its
        body with `chunk_max_size` windowing, carrying context;
      - API facts: `StructureKind` is a pyo3 enum (`str()` → `'Function'`, no
        `.value`); `ProcessResult` is attribute-access only (not a dict); all
        lines/spans are **0-based** (convert to 1-based at the public boundary);
        a structure `span` may exclude the trailing newline (slice
        `start_byte:end_byte`, not raw source length);
      - `metadata.has_error_nodes` handles broken syntax (no paragraph fallback);
      - **raw tree-sitter walker** (`get_parser` + node walk) is **load-bearing**
        (decorator recovery), not a churn contingency — keep it a thin seam;
        pin the version; cache-dir/download concerns apply only to the 296
        non-target languages (lazy `manifest_languages()`).
- [ ] **`ValidatedChunker`**: enforce invariants on every chunker output:
      non-empty, `tokens <= budget.max_tokens` (post-split if exceeded), line
      ranges present, chunk_type in vocabulary, parent policy per strategy.
- [ ] **Docling chunker bridge**: HybridChunker wrapper with heading metadata.

### Test plan

- [ ] Golden tests: fixtures per chunker (markdown with code fences, decorated
      Python, broken syntax, JS/TS/Rust/Go samples) — exact expected chunks.
- [ ] Property tests: for a corpus (incl. adversarial inputs), all invariants
      hold; token budget never exceeded; no empty chunks.
- [ ] Docling tests with mocked DocumentConverter; `ImportError` path tests.

**Acceptance**: chunk invariants hold over the corpus; tree-sitter handles all
10 types with decorators attached and `parent` populated; core import pulls no
heavy deps (docling/tree-sitter grammars lazy).

---

## 5. Phase 3 — Real embedding providers + MRL

**Goal**: real vectors flow through the pipeline; MRL policy verified.

**Inputs**: Phase 1 (protocol, registry), Phase 2 (chunk budget plumbing).

### Work items

- [ ] `embeddy/providers/local.py` — sentence-transformers adapter:
      - load by model id; expose `dimension` (native) and `context_length`
        from `ModelSpec` (or model metadata);
      - prompts map: model card → semantic roles (query/document/retrieval)
        using `prompt_name` (harrier) and `prompt` (Qwen3) conventions;
      - MRL: `truncate_dim` when `mrl_range` present and requested < native;
      - batching, dtype, device options.
- [ ] `embeddy/providers/http.py` — OpenAI-compatible adapter:
      - `POST /v1/embeddings` (+ rerank endpoint), timeouts, retries;
      - response validation: returned dimensions match resolved dimension;
      - shared request-building with the client (one protocol).
- [ ] `embeddy/providers/rerank.py` — `RerankerProvider` Protocol; local
      CrossEncoder + remote implementations.
- [ ] Registry entries (CONCEPT §6.1): **Qwen3-0.6B (text default, MRL)**,
      Qwen3-VL-2B (multimodal, local-only), harrier-0.6b/270m/27b (high-quality
      options — verify harrier ST pooling before adopting), bge-m3.
- [ ] MRL truncation followed by **L2 re-normalization** (a sliced vector is not
      unit-norm; cosine assumes unit vectors) — test that norms ≈ 1.0.
- [ ] Config: `embedder.model = "Qwen/Qwen3-Embedding-0.6B"`,
      `embedder.embedding_dimension: int | None = None`, `embedder.prompt_role`
      defaults; all via pydantic-settings.
- [ ] HTTP provider scope (CONCEPT §7): text-dense only; instructions via
      `extra_body` (upstreams may drop); multimodal/sparse are local-only in v1;
      rerank uses the TEI/Jina shape, not a non-existent OpenAI `/v1/rerank`.

### Test plan

- [ ] Unit: `resolve_dimension` matrix (CONCEPT §5.2 table) — exact.
- [ ] Integration (tag `[slow]`): tiny ST model end-to-end; MRL truncation
      produces correct dims and consistent cosine ranking.
- [ ] HTTP provider: httpx `MockTransport` — success, 4xx, 5xx, timeout,
      wrong-dimension response.
- [ ] Multimodal smoke (tag `[gpu]`, optional in CI): Qwen3-VL-2B text+image.

**Acceptance**: MRL matrix passes; truncated vectors are re-normalized (norm≈1);
non-MRL model + wrong dimension raises at config time; remote and local providers
produce **ranking-parity within tolerance** (NOT bitwise/numeric equality — ST vs
a remote TEI never match exactly) for the same model+instruction.

---

## 6. Phase 4 — Storage & search core

**Goal**: metric-honest, pre-filtered, source-aware retrieval.

**Inputs**: Phase 1 index slice; Phase 3 (query vectors real).

### Work items

- [ ] **Sources** (`embeddy/index/sources.py`):
      - `sources(id, collection_id, path, content_hash, size, mtime)`;
      - dedup = source-level compare (two identical files at different paths are
        two sources — fixes CONCEPT H-dedup);
      - reindex = atomic swap of a source's chunk set in one transaction
        (fixes H7);
      - delete source cascades; incremental sync = diff on `sources`.
- [ ] **SQL pre-filters** (`embeddy/index/filters.py`): compile
      `SearchFilters` (content_types, source_path_prefix, chunk_types,
      **metadata_match**) into SQL WHERE clauses joined before KNN/FTS.
      Eliminates post-filter over-fetch (fixes M3/H1).
- [ ] **FTS5**: porter + unicode61; query sanitization policy (escape FTS5
      metacharacters or reject; documented behavior).
- [ ] **Typed scores**: `ScoredDocument` carries `metric` (cosine/bm25);
      `min_score` compares against the metric's true semantics (fixes C3).
- [ ] **aiosqlite** migration of the index (replaces ad-hoc to_thread):
      - **never set `db.isolation_level` from the caller thread** — aiosqlite
        runs the connection on a worker thread and sqlite3's threading check
        rejects it (verified in the Phase-0 spike);
      - atomic reindex expressed with the **implicit transaction** +
        `commit()`/`rollback()` (spike proves both the success and the
        mid-swap-failure rollback paths);
      - vec0 tables with a **TEXT PRIMARY KEY** take no rowid —
        `INSERT INTO v(id, embedding) VALUES (...)` (verified).
- [ ] **Search**: `search_vector`, `search_fulltext`, `search_hybrid` (RRF +
      weighted), optional rerank stage (Phase 3 provider); `total_results` =
      pre-truncation count (documented semantics).
- [ ] **Benchmark spike** (`benchmarks/`): sqlite-vec vs LanceDB at 100k/1M
      vectors with filters → decision record for default backend.

### Test plan

- [ ] Pre-filter recall: restrictive filters return full `top_k` when matches
      exist beyond old over-fetch window.
- [ ] metadata_match now filters (regression for the silent no-op).
- [ ] Atomic reindex: inject failure mid-reingest → old index intact.
- [ ] Score correctness: known cosine pairs → exact expected scores.
- [ ] FTS sanitization: metacharacter queries documented behavior.

**Acceptance**: all filters work as pre-filters; scores metric-correct;
reindex failure-safe; spike decision recorded in `docs/decisions/`.

---

## 7. Phase 5 — Pipeline & orchestration

**Inputs**: Phase 4 (sources, store), Phase 3 (providers), Phase 2 (chonkai).

### Work items

- [ ] Bounded worker pool (`asyncio.Semaphore` + queue): `ingest_directory`
      honors `pipeline.concurrency`; read/chunk/embed/write phases overlap.
- [ ] Progress: async iterator / per-file callback (keeps the `on_file_indexed`
      concept).
- [ ] Source ops: `ingest_text`, `ingest_file` (dedup), `ingest_directory`,
      `reindex` (atomic swap), `delete_source`, `sync` (incremental).
- [ ] Error collection: chunking errors included in typed `IngestStats`
      (fixes H6); stats counters documented (chunks vs files semantics).
- [ ] `IngestStats` typed record with clear field semantics.

### Test plan

- [ ] Concurrency determinism: N files with slow embedder mock → max in-flight
      ≤ concurrency; flat memory.
- [ ] Dedup: identical content at two paths → both indexed (two sources).
- [ ] Reindex atomicity (repeat of Phase 4 test at pipeline level).
- [ ] Error paths: chunker failure recorded, not raised.

**Acceptance**: concurrency config is honored and tested; all source ops
covered; `IngestStats` semantics documented in code.

---

## 8. Phase 6 — Server, client, CLI

**Inputs**: Phases 3–5.

### Work items

- [ ] **Server** (`embeddy/server.py`):
      - FastAPI factory; **lifespan** opens store, loads provider, closes on
        shutdown — `uvicorn embeddy.server:app` works bare;
      - `GET /health/live`, `GET /health/ready` (provider loaded, store open);
      - OpenAI-compatible `POST /v1/embeddings`, `POST /v1/rerank`;
      - `POST /api/v1/search` (+ `/similar`, `/rerank`), `/api/v1/ingest/*`,
        `/api/v1/collections`, `/api/v1/chunks`;
      - error map: ValidationError → 400, other EmbeddyError → 500, FastAPI 422
        → structured shape; CORS from config;
      - **request/batch size limits** (max body, max inputs per embed, max
        top_k) — OOM/crash-prevention, not security (auth/SSRF/path-traversal are
        out of scope: personal project on a trusted tailnet, CONCEPT §9.7).
- [ ] **Client** (`embeddy/client.py`): httpx; mirrors server; shares
      request-building with HTTPProvider.
- [ ] **CLI**: `serve`, `ingest text|file|dir`, `search`, `info` (Typer);
      config precedence CLI > file > env > defaults.

### Test plan

- [ ] ASGI tests with injected mocks (keeps the dependency-injection seam).
- [ ] Contract tests: client ↔ live app for every endpoint; payload parity.
- [ ] Health: `ready=false` when provider not loaded.
- [ ] CLI tests (typer runner).

**Acceptance**: server starts bare and reports honest readiness; client and
server never drift (one protocol, shared builders); CLI documented.

---

## 9. Phase 7 — Packaging, docs, benchmarks

**Inputs**: Phases 2–6.

### Work items

- [ ] Extras verified: `pip install embeddy` (bare) imports; each extra
      installs cleanly; wheel contents inspected (no accidental heavy deps in
      core).
- [ ] Lazy imports audit: `grep -r "import torch\|import transformers\|import
      docling\|import fastapi"` → only in lazy-loading modules.
- [ ] Docs: rewrite USER_GUIDE, INTEGRATION, ARCHITECTURE for the new
      architecture; CLI reference; config reference (only implemented fields).
- [ ] Benchmarks: chonkai chunk-quality harness; embeddy search/ingest/
      resource benchmarks.
- [ ] **Retrieval-quality eval harness** (CONCEPT §9.6): fixed corpus + queries +
      qrels → nDCG@k / recall@k across model/chunker/fusion changes; pre-release
      gate. (First cut can land at M4 to catch retrieval regressions early.)
      **Scope: the fake-provider gate is MECHANICAL** — regression detection +
      full determinism, with 0.50 nDCG@10 / 0.80 recall@10 thresholds derived
      from the fixed 20-doc corpus (`eval/run_eval.py` recomputes them).
      Absolute retrieval-quality gates live in the Phase-3 `[slow]`
      sentence-transformers integration tests.
- [ ] Changelog + versioning policy (release train: chonkai + embeddy together
      initially).

**Acceptance**: fresh-env install matrix green; docs match implementation;
benchmarks runnable and wired to CI (optional job).

---

## 10. Phase 8 — Scale path (post-v1 gates)

- [ ] Qdrant adapter (implements `Searchable`) — dense + sparse, payload
      filters, quantization.
- [ ] LanceDB spike decision (from Phase 4) — adopt as second embedded backend
      only if it wins the benchmark.
- [ ] Integration adapters (Haystack components, LlamaIndex vector-store
      interface) — optional, post-v1.

---

## 11. Testing strategy (summary)

| Layer | Location | Tooling | Tags |
|-------|----------|---------|------|
| Unit (pure logic: chunkers, fusion, resolve_dimension, filters) | per-package `tests/` | pytest | `[unit]` (default) |
| Integration (real sqlite-vec/FTS5, mocks for models) | `tests/` | pytest-asyncio | `[integration]` |
| Golden (chunker fixtures) | `chonkai/tests/fixtures/` | pytest | `[unit]` |
| Property (invariants over corpus) | `chonkai/tests/` | hypothesis | `[unit]` |
| Contract (client↔server, HTTPProvider↔wire) | `tests/contract/` | pytest-asyncio + httpx MockTransport | `[integration]` |
| E2E (keystone slice; later full pipeline) | `tests/e2e/` | pytest-asyncio | `[e2e]` |
| Model-touching (real ST model, multimodal) | `tests/slow/` | pytest | `[slow]`, `[gpu]` (opt-in CI) |
| Benchmarks (search/ingest/chunk quality) | `benchmarks/` | pytest-benchmark, psutil | separate job |

Coverage gate: ≥90% on core paths (providers, index, search, pipeline);
chunker code ≥95%.

---

## 12. Milestones & review gates

| Milestone | Phases | Review gate |
|-----------|--------|-------------|
| M1 — Skeleton green | 0, 1 | E2E keystone test passes; protocols *drafted* (NOT frozen) |
| M2 — chonkai v1 | 2 | Invariants hold on corpus; standalone import clean |
| M3 — Real embeddings | 3 | MRL matrix green; local+remote interchangeable |
| M4 — Retrieval core | 4, 5 | Filters pre-filtered; reindex atomic; concurrency real; **protocols frozen here** (source ops, filters, rerank now proven); retrieval eval harness green |
| M5 — Product surface | 6 | Server bare-start + honest health; client parity |
| M6 — Release | 7 | Install matrix green; docs match; benchmarks run |
| M7 — Scale | 8 | Qdrant adapter; LanceDB decision recorded |

Freeze points: protocols (`EmbeddingProvider`, `Searchable`, `RerankerProvider`,
chonkai public API) are **drafted at M1 and frozen at M4** — freezing at M1 is
premature because source ops, filter compilation, rerank, and multimodal reshape
the protocols and don't exist until Phases 3–5. Changes after the M4 freeze
require a `docs/decisions/` record.

---

## 13. Risk register (from CONCEPT §9)

| Risk | Mitigation | When |
|------|-----------|------|
| tree-sitter-language-pack API churn | Pin; adapter; raw-walker fallback | M2 |
| sqlite-vec pre-v1 breaking changes | Isolate behind `Searchable` | M1+ |
| LanceDB API churn | Spike only; decision record before adoption | M4/M8 |
| Model landscape moves | Registry is config; re-review defaults per release | M3+ |
| chonkai import-name collision | Pre-publish check (CONCEPT §9.1) | M6 |
| Cross-package invariant drift (chunk size vs context) | `ChunkBudget` passed by caller; contract tests | M2–M3 |

---

## 14. Open decisions — RESOLVED (review 2026-07-31)

- [x] **Repo layout**: uv workspace (`packages/` dirs). CONCEPT §8 updated to match.
- [x] **semchunk packaging**: chonkai **core** (tiny, pure-python). tree-sitter
      also core; only Docling + arbitrary-HF tokenizers are extras.
- [x] **FTS query policy**: **escape/quote-wrap by default + documented**, plus an
      explicit opt-in `raw` mode. Prefer double-quote phrase-wrapping over
      hand-escaping FTS5 metacharacters.
- [x] **`total_results` semantics**: **pre-truncation count** — ensure it's a cheap
      `COUNT(*)`, not a full materialization.
- [x] **Default text model**: **`Qwen/Qwen3-Embedding-0.6B`** (proven, Apache-2.0,
      MRL-capable). harrier-0.6b is a documented high-quality option; promote to
      default only after verifying ST pooling + winning the eval harness (§9.5/§9.6).

### Out of scope (personal project, trusted tailnet)

- v0.3.x data migration — no in-place migration; re-ingest from source. (`user_version`
  stamp kept only to prevent self-inflicted dev-time schema breakage.)
- Server security hardening (auth / SSRF / path-traversal). Kept: request/batch
  size limits (OOM prevention).
