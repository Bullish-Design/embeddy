# chonkai + embeddy

Two libraries in one uv workspace (IMPLEMENTATION_PLAN §2).

| Package | Purpose | Depends on |
|---------|---------|-----------|
| **chonkai** (`packages/chonkai`) | Document processing: ingest + parse + chunk, with guaranteed invariants | — (never imports embeddy) |
| **embeddy** (`packages/embeddy`) | Embed + store + search + serve; typed, metric-honest retrieval | chonkai |

## Layout

```
pyproject.toml            workspace root: members + shared ruff/ty/pytest config
packages/chonkai/         document processing (src/chonkai)
packages/embeddy/         retrieval workflow (src/embeddy)
tests/                    shared integration/e2e tests
benchmarks/               Phase-7 harness + Phase-4 spike (not in testpaths)
docs/                     USER_GUIDE / INTEGRATION / ARCHITECTURE / config / cli
eval/                     retrieval-quality eval gate (default suite)
scripts/                  install-matrix + lazy-import CI checks
.github/workflows/ci.yml  lint / type / test / install-matrix / benchmarks
```

## Quickstart

```bash
devenv shell          # or: uv sync
uv run pytest         # unit + integration + contract + e2e + eval gate
uv run ty check       # strict type check
uv run ruff check .   # lint
python3 scripts/check_lazy_imports.py   # lazy-import audit
python3 scripts/check_install_matrix.py # fresh-venv install matrix (slow; needs network)
```

Docs: [USER_GUIDE](docs/USER_GUIDE.md) · [INTEGRATION](docs/INTEGRATION.md) ·
[ARCHITECTURE](docs/ARCHITECTURE.md) · [config](docs/config.md) ·
[cli](docs/cli.md) · [eval gate](docs/retrieval-eval.md) ·
[versioning](docs/versioning.md) · [CHANGELOG](CHANGELOG.md)

## Status — M6 (Phase 7: packaging, docs, benchmarks — release gate)

M6 is the **release gate** (plan §12): install matrix green, docs match the
implementation, benchmarks runnable, changelog present.

- **Docs**: `docs/USER_GUIDE.md` (how to use), `docs/INTEGRATION.md` (wire
  protocol + upstreams), `docs/ARCHITECTURE.md` (design map),
  `docs/retrieval-eval.md` (the retrieval-quality gate),
  `docs/versioning.md` (release-train policy), `docs/config.md`,
  `docs/cli.md`, `docs/decisions/`.
- **Install matrix** (`scripts/check_install_matrix.py`): every extras row
  verified in fresh venvs — zero extras + all extras — wired to CI.
- **Lazy-import audit** (`scripts/check_lazy_imports.py`): AST-based CI gate
  keeping extra-only deps out of module-level imports (except the two KNOWN
  entry points `embeddy.server`/`embeddy.cli`).
- **Benchmarks** (`benchmarks/`, pytest-benchmark + psutil): chonkai
  chunk-quality + embeddy search/ingest/resource harnesses. Outside pytest
  testpaths — never the default suite; optional CI job
  (`docs/decisions/0002`).
- **CHANGELOG.md** + versioning policy (`docs/versioning.md`).

Phase-7 verification (M5 baseline, no regressions): **463 passed + 3
skipped** (the 3 skips are the `[slow]` model-touching tests — correct),
coverage 96% total (server 95%, client 97%, cli 100%, config 100%), ty
strict, ruff clean, install matrix 7/7 rows green, lazy-import audit clean.

## What ships (M1–M5 foundations)

- `embeddy.server` — FastAPI factory (`create_app` with a DI seam) + a bare
  module-level `app` (`uvicorn embeddy.server:app`); lifespan opens the store,
  builds+loads the provider, closes on shutdown; honest health (`/health/live`,
  `/health/ready`); OpenAI-compatible `/v1/embeddings`, TEI/Jina `/v1/rerank`,
  and `/api/v1/{search,similar,rerank,ingest,collections,chunks}`; CORS from
  config; request/batch size limits (OOM prevention only, CONCEPT §9.7);
  query-role resolution on embed/search routes, document role in the pipeline
  (H2). See `docs/config.md` for the error map.
- `embeddy.client` — `EmbeddyClient`, mirrors every server route, shares
  `build_embeddings_request` with `HTTPProvider` (one wire protocol).
- `embeddy.cli` — `serve` / `ingest text|file|dir` / `search` / `info` (Typer;
  config precedence CLI > file > env > defaults). See `docs/cli.md`.
- `embeddy.config` — `ServerSettings` (`server` section: CORS + the retained
  operational guards), `load_server_settings`; `dotenv_filtering="match_prefix"`
  so one shared `.env` can hold all sections. See `docs/config.md`.
- Test inventory: unit (`packages/embeddy/tests/`), contract/ASGI/CLI
  (`tests/contract/`, tagged `[integration]`), keystone e2e, eval gate.

Earlier milestones: M1 (keystone), M2 (chonkai v1), M3 (real embedding
providers + MRL), M4 (storage & search core, pipeline, eval gate), M5
(server, client, CLI). M6 = this phase (packaging/docs/benchmarks).

Phase 8 (post-v1, not implemented): Qdrant adapter, integration adapters.

M4 foundation (frozen protocols + core):

- `embeddy.protocol` — typed records (`EmbedInput`, `Vector`, `StoredChunk`,
  `ScoredDocument` w/ `.metric`, `CollectionStats`, `SourceMetadata`,
  `SourceId`) and the `EmbeddingProvider` protocol. No `dict[str, Any]`
  crosses any protocol. Drafted at M1, **frozen at M4**.
- `embeddy.registry` — `ModelSpec` + `resolve_dimension` (exact CONCEPT §5.2
  MRL logic) + minimal default registry (Qwen3-0.6B).
- `embeddy.providers.fake` — dev-only deterministic dim-8 `FakeProvider`.
- `embeddy.index` — `Searchable` protocol **including source ops**, and
  `SqliteStore` (aiosqlite + sqlite-vec cosine + FTS5 porter/unicode61 +
  `PRAGMA user_version` stamp). Phase-1 surface: `create_collection`,
  `upsert_source`, `get_source`, `add`, `search_vector`, `search_fts`,
  `stats`.
- `embeddy.search` — `fuse_rrf` (k=60) / `fuse_weighted` pure functions.
- `chonkai` — `IngestResult` / `Chunk` / `SourceMetadata`, `ParagraphChunker`,
  `ValidatedChunker` (non-empty, token budget, line ranges).
- `chonkai.ingest` — content-type detection, encoding-fallback file reading,
  docling bridge (lazy extra), sha256 hashing.
- `chonkai.chunkers` — paragraph (short-merge), markdown (code-fence-aware,
  heading→`parent`), semchunk (token-accurate), tree-sitter (10 bundled code
  grammars + markdown; decorator recovery via the raw parser; granularity
  filter), docling bridge (lazy extra), `get_chunker` factory.
- Tree-sitter language set: **release-required = python / bash / markdown**
  (`RELEASE_REQUIRED_LANGUAGES`); the other 7 bundled grammars (js/ts/rust/go/
  c/cpp/java/ruby) stay supported but are not release-gated. Markdown is a
  two-grammar pair (`markdown` block + `markdown_inline` injection); the block
  grammar is a manifest download on first use (offline-safe after caching,
  cache dir via `tslp.configure(PackConfig(cache_dir=...))`), with a
  pure-Python fallback chunker when it cannot load.
- `chonkai.validated` — invariants on every chunker output + token post-split.
- Type checking: Ty (strict by default) — replaces mypy.

See `.scratch/projects/001-greenfield-rewrite/` for the concept, plan, and
Phase-0 spike evidence.
