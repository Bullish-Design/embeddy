# Changelog

All notable changes to the **chonkai + embeddy** release train. Both
packages ship together at the same version initially (policy:
`docs/versioning.md`).

The format is based on [Keep a Changelog](https://keepachangelog.com/), and
this project adheres to [Semantic Versioning](https://semver.org/) with
0.x semantics (see the policy doc).

## [Unreleased]

### Added — Phase 8 / M7 (the scale path, post-v1 gates)

- **QdrantStore** (`embeddy/index/qdrant.py`) — a second `Searchable`
  backend behind the FROZEN M4 protocol (no protocol or wire-shape change):
  dense cosine ANN, payload filters (content_types / source_path_prefix /
  chunk_types / metadata_match) as true pre-filters, collection-level
  quantization (scalar int8 / binary, via the DSN or `create_collection`),
  the full source-op set (upsert/get/list/reindex/delete), and the
  beyond-protocol extras (create_collection / get_chunk / list_chunks /
  list_collections). All API facts verified empirically against
  qdrant-client 1.18.0 (probes; `docs/decisions/0004`).
- **Store selection** — a `store` config section with the one config line
  (`store.url` / `EMBEDDY_STORE_URL`: `qdrant://host[:port]`,
  `qdrant://:memory:`, sqlite DSN, or bare path) and a `build_store`
  factory mirroring `build_provider`. `None` = the sqlite `store_path`
  default, so the bare `app = create_app()` is unchanged. The server
  lifespan + CLI `serve`/`info` read it; the honest-health contract holds
  when qdrant is unreachable (not-ready + reason).
- **Sparse plumbing** (decision 0004, option b): collections carry the
  `sparse` named vector; `add()`/`reindex_source()` accept optional
  `sparse_vectors`; `search_sparse()` is a beyond-protocol extension scored
  with the new additive `Metric.SPARSE_DOT`. Exercised by tests with
  synthetic vectors; a real learned-sparse encoder (bge-m3) is out of
  scope.
- **Tests**: 72 new tests — DSN parsing + factory (100%), QdrantStore unit
  + integration on the hermetic in-memory backend (100%; dense search
  parity vs SqliteStore, source-op lifecycle, pre-filter recall, reindex
  failure leaves old chunks queryable, quantization, sparse, FTS BM25),
  qdrant-backed server contract tests (store selection via the lifespan,
  unreachable-qdrant not-ready, full product surface). qdrant-client is a
  dev dependency (the httpx/fastapi precedent) so the default suite runs
  them; the lazy-import audit now covers the `qdrant_client` root.
- **Docs**: `docs/decisions/0003` (LanceDB revisit at M7 — still not
  adopted; Qdrant is the scale path) and `0004` (adapter decisions: sparse
  status, reindex atomicity, the pure-Python BM25 FTS path, quantization,
  sync-client note); `config.md` (`store` section), `USER_GUIDE` (store
  selection + qdrant caveats), `INTEGRATION` (storage backends),
  `ARCHITECTURE` (the second backend + factory + decisions).

### Known limits (documented, not bugs)

- Qdrant `search_fts` is a pure-Python BM25 scan over payloads (no BM25
  engine; `raw` is a no-op) — see 0004.
- Qdrant `reindex_source` has no transactional atomicity: upsert-first,
  old chunks intact on failure, possible old+new overlap mid-failure (0004).
- Out of scope (not planned): a real learned-sparse encoder (bge-m3), a
  lexical index for the Qdrant FTS path, an async qdrant client, integration
  adapters (Haystack/LlamaIndex).

## [0.1.0] — 2026-08-01 (M6 — initial release)

The first release. Phases M1–M6 of the greenfield rewrite, gated by the
M6 release gate: fresh-env install matrix green, docs match the
implementation, benchmarks runnable, eval gate documented.

### Added — Phase 7 (this release's packaging/docs/benchmarks work)

- **Install matrix** (`scripts/check_install_matrix.py`): fresh-venv
  verification of every extras row — zero extras, `embeddy[server]`,
  `embeddy[client]`, `embeddy[local]`, `embeddy[qdrant]`,
  `chonkai[docling]`, `chonkai[tokenizers]` — wired into CI as a matrix job
  (replaces the M5 import-smoke job).
- **Lazy-import audit** (`scripts/check_lazy_imports.py`): AST-based CI gate
  that extra-only dependencies import only lazily, in the two KNOWN
  module-level entry points (`embeddy.server` → fastapi/starlette,
  `embeddy.cli` → typer), or under `TYPE_CHECKING`.
- **Docs**: `USER_GUIDE.md`, `INTEGRATION.md`, `ARCHITECTURE.md`,
  `retrieval-eval.md`, `versioning.md`; verified `cli.md` and `config.md`
  match the code (M6 "docs match" gate).
- **Benchmarks** (`benchmarks/`, pytest-benchmark + psutil): chonkai
  chunk-quality harness (throughput + invariants) and an embeddy
  search/ingest/resource harness (ingest wall time, vector/hybrid/filtered
  latency, peak-RSS flat-memory profile). Outside pytest testpaths — never
  the default suite; wired to CI as an optional job.
- **Eval gate documented**: thresholds (nDCG@10 ≥ 0.50, recall@10 ≥ 0.80),
  recompute path (`eval/run_eval.py`), and regression semantics
  (`docs/retrieval-eval.md`).
- **Changelog + versioning policy** (`docs/versioning.md`).

### Added — M1–M5 foundations (in this release)

- **chonkai**: `Ingestor` (content-type detection, encoding fallback,
  docling bridge), chunkers (paragraph, markdown with code-fence awareness,
  semchunk token-accurate, tree-sitter for 10 bundled code grammars with
  decorator recovery and granularity), `ValidatedChunker` invariants, token
  counting (tiktoken core + arbitrary-HF via `tokenizers` extra).
- **embeddy core**: typed protocols (`EmbeddingProvider`,
  `RerankerProvider`, `Searchable` incl. source ops), model registry + MRL
  (`resolve_dimension`, truncate-and-renormalize), `SqliteStore`
  (sqlite-vec cosine + FTS5 + sources + atomic reindex + SQL pre-filters),
  pure RRF/weighted fusion, bounded concurrent `IngestPipeline` with typed
  `IngestStats`.
- **embeddy providers**: `LocalProvider` (sentence-transformers),
  `HTTPProvider` (OpenAI-compatible, shared wire with the client),
  rerankers (CrossEncoder / HTTP).
- **embeddy product surface**: FastAPI server (lifespan, honest health,
  OpenAI-compatible `/v1/embeddings`, TEI/Jina `/v1/rerank`, `/api/v1/*`,
  structured errors, CORS, size limits), `EmbeddyClient`, Typer CLI
  (`serve`/`ingest text|file|dir`/`search`/`info`), pydantic-settings
  config (CLI > file > env > defaults).
- **Quality gates**: 463 tests (unit/integration/contract/e2e), 96%
  coverage, ty strict, ruff clean, deterministic eval gate.

### Fixed

- The pre-rewrite findings (CONCEPT §2) — all Critical/High items fixed by
  design through M1–M5: real embeddings (C1), MRL actually truncates (C2),
  cosine-metric vec tables (C3), zero-extras importable (C4), one wire
  protocol (C5), metadata pre-filters (H1), document-role resolution (H2),
  real code granularity + decorators (H3), code-fence-aware markdown (H4),
  token budget enforced (H5), chunking errors collected (H6), atomic
  reindex (H7), real concurrency (H8), env-var config applied (H9), CORS +
  collection metadata + instruction honored (H10), and the M1–M12 medium
  findings (dispositioned in the implementation plan).

### Known limits (documented, not bugs)

- Multimodal + learned-sparse embeddings: local-provider only (not
  expressible over OpenAI `/v1/embeddings`).
- `embeddy[qdrant]` installs `qdrant-client` but ships no adapter yet
  (Phase 8).
- Server auth/SSRF/path-traversal: out of scope (trusted tailnet); only the
  size-limit OOM guards ship.
- No in-place data migration from the pre-rewrite 0.3.x stores — re-ingest
  from source (policy).
