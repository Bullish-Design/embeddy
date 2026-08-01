# Kick off Phase 6 — server, client, CLI (M5 product surface) — in this repo
(/home/andrew/Documents/Projects/embeddy). M4 is complete and pushed (commit
cf6c261 on main — storage & search core, pipeline & orchestration, and the
retrieval eval gate are all in). Everything derives from IMPLEMENTATION_PLAN
§8 (Phase 6) and §12 (the M5 gate); the plan is the authority. Do NOT skip
ahead to Phase 7.

════════════════════════════════════════════════════════════════════════
⚠️ DATA-SAFETY RULES — READ TWICE, THESE OVERRIDE EVERYTHING BELOW
════════════════════════════════════════════════════════════════════════
1. NEVER delete or overwrite any file that is NOT safely recoverable from git.
   Before removing ANYTHING, run `git status --porcelain --ignored` and account
   for every untracked (`??`) and ignored (`!!`) entry. If a path is
   untracked/ignored and not obviously disposable build junk, STOP and ask.
2. The `.scratch/` directory tree is SACRED. Do NOT delete, move, rename, or
   edit anything under `.scratch/`. Leave it 100% untouched.
3. Only delete files that are TRACKED IN GIT (restorable via `git restore`).
4. Pre-authorized disposables you MAY delete without asking: `__pycache__/`,
   `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `.devenv/`
   build output, `.venv/`.
5. Use `git rm` for tracked files, NOT raw `rm`, wherever possible. Do NOT
   commit anything until I review it.

FIRST — read these in full before touching anything:
- .scratch/projects/001-greenfield-rewrite/IMPLEMENTATION_PLAN.md  (§8 "Phase 6 —
  Server, client, CLI" is your scope; §12 milestones — M5 "Product surface",
  gate: server bare-start + honest health, client parity; §11 testing;
  §13 risks)
- .scratch/projects/001-greenfield-rewrite/CONCEPT.md  (§5.7 server; §5.8
  client; §9.7 the retained operational guard: request/batch size limits as
  OOM/crash-prevention, NOT security — auth/SSRF/path-traversal are out of
  scope, trusted tailnet; §5.1 role→instruction resolution — the server is a
  CALLER that resolves roles; §3.8 config)
- Current embeddy surface (ALL FROZEN at M4 — the server/client/CLI are pure
  CONSUMERS, never mutators):
  - protocol/embedding.py: EmbeddingProvider — `dimension`, `context_length`,
    `model_name`, `async encode(inputs, instruction)` where `instruction` is
    an ALREADY-RESOLVED string. Callers resolve `role → instruction` via
    `registry.resolve_instruction(model_id, role)` (the ONLY resolution
    point). Providers never see a role (H2).
  - protocol/rerank.py: RerankerProvider — `model_name`,
    `async rerank(query, documents, top_k=None, instruction=None) ->
    list[RerankHit(index, score)]`.
  - index/base.py: Searchable (frozen) — add/delete/search_vector/search_fts/
    stats + source ops (upsert_source/get_source/reindex_source/delete_source/
    list_sources). `SqliteStore` extras BEYOND the protocol: `create_collection`
    and `count_fts` — the server owns collection lifecycle and must call
    create_collection, because the pipeline REQUIRES collections to exist.
  - search.py: `search_hybrid(store, *, collection, query_text, query_vector,
    filters, top_k, mode, weights, retrieve_k, min_score, raw, reranker,
    rerank_top_k, instruction)` -> SearchResult(results, total_results, metric,
    mode, query, collection). `instruction` = resolved QUERY-role string.
  - pipeline.py (frozen, new at M5): IngestPipeline(store, provider, chunker,
    budget, token_counter, instruction, concurrency, ingestor,
    on_file_indexed, reindex_unchanged) with ingest_text / ingest_file /
    ingest_directory / reindex / delete_source -> bool / sync; IngestStats
    (files_attempted/indexed/skipped/deleted, chunks_indexed, errors —
    collected, never raised); FileEvent; PipelineError; generate_source_id.
  - registry.py (resolve_instruction, resolve_dimension, get_model),
    providers/factory.py (build_provider(model, dimension, backend=local|http,
    **kwargs — http needs base_url), providers/http.py (build_embeddings_request
    — the SHARED OpenAI-compatible request builder), providers/rerank.py (remote
    rerank is the TEI/Jina shape `POST {base}/rerank` with {"query","texts",
    "top_n"}, NOT an OpenAI /v1/rerank standard), providers/local.py,
    providers/fake.py (dev-only), errors.py (EmbeddyError base; ProviderError,
    ProviderInputError, HTTPProviderError, WrongDimensionError, RerankError,
    ModelNotLoadedError, PipelineError), budget.py (chunk_budget),
    config.py (EmbedderSettings, PipelineSettings, load_settings,
    load_pipeline_settings — precedence CLI > file > env > defaults via
    _PrecedenceSettings), chonkai public API (Ingestor, get_chunker,
    ValidatedChunker, ChunkBudget, IngestResult).
- tests/e2e/test_keystone.py and eval/test_eval.py (the M4 gate — must keep
  passing; do not regress).

════════════════════════════════════════════════════════════════════════
CURRENT STATE (M5, committed cf6c261) — build on this, don't rebuild it:
════════════════════════════════════════════════════════════════════════
uv workspace, Python >=3.11, dev env Python 3.13 via devenv. All commands run
INSIDE `devenv shell` (uv is at `.devenv/profile/bin/uv` if PATH is bare).
Pins unchanged (pydantic 2.12.5, numpy 2.3.5, sqlite-vec 0.1.9, aiosqlite
0.22.1, pydantic-settings 2.14.2).

- Gate status at M4: 359 passed + 3 skipped (the 3 skips are the [slow]
  model-touching tests — correct), ruff clean, ty clean, one-way dependency
  grep clean, zero-extras `import embeddy` clean. Coverage 96% total;
  pipeline 100% (plan §11: >=90% on providers, index, search, pipeline —
  hold server/client/cli to >=90% as the new M5 core).
- Test layout: unit in packages/embeddy/tests/ (no DB), integration in tests/
  tagged [integration] (real sqlite-vec/FTS5), e2e in tests/e2e/, slow in
  tests/slow/ ([slow]/[gpu] opt-in). asyncio_mode=auto.
- ⚠️ DEV-DEPS GAP (a REQUIRED Phase-6 decision, see work item 1): the root
  `pyproject.toml` dev group has pytest/pytest-asyncio/pytest-cov/hypothesis/
  ruff/ty/httpx ONLY. fastapi, uvicorn and typer are NOT installed anywhere
  (they are declared as `embeddy[server]` / future extras only). ASGI tests,
  contract tests and CLI tests therefore cannot run yet. Follow the httpx
  precedent: httpx is an embeddy extra BUT is in the dev group "so the
  HTTPProvider wire tests" run — add fastapi, uvicorn and typer to the dev
  group the same way, with the same comment pattern. The LIBRARY imports stay
  lazy regardless (see the [tool.ty.overrides] rule below).
- ⚠️ ENV QUIRKS (verified repeatedly): `uv run pytest` works as-is, but a
  plain `uv run python probe.py` cannot import numpy without the prefix:
  LSTD=$(dirname $(find /nix/store -maxdepth 3 -name 'libstdc++.so.6' | head -1))
  LZ=$(dirname $(find /nix/store -maxdepth 3 -name 'libz.so.1' | head -1))
  export LD_LIBRARY_PATH="$LSTD:$LZ"
  sentence-transformers AND lancedb are installed MANUALLY in the dev venv only
  — NOT in uv.lock; `uv sync` removes them and the [slow] tests skip (correct).
- Type checking is **ty**, NOT mypy. [tool.ty.overrides] exists for every
  lazy-import/extra-only module (providers/*, docling, tokens, tests with PIL,
  benchmarks/**). If fastapi/uvicorn/typer land in the dev group the imports
  RESOLVE in the dev env and no override is needed; keep the lazy imports
  anyway so a zero-extras install still imports embeddy cleanly. Any module
  that imports an extra-only dep that is NOT in the dev group needs the
  override (the existing pattern).
- The eval harness is deterministic and offline and runs in the DEFAULT suite
  (do not tag it [slow]).

════════════════════════════════════════════════════════════════════════
PHASE 6 SCOPE — server, client, CLI (plan §8) + the M5 gate (plan §12).
Goal: a server that starts bare and reports HONEST readiness; an httpx client
that mirrors the server exactly (one protocol, shared request builders); a
Typer CLI; and the M5 review gate. All protocols are FROZEN at M4 — if your
work requires reshaping EmbeddingProvider / Searchable / RerankerProvider /
search_hybrid / IngestPipeline / the chonkai public API, STOP and flag it;
any change after the freeze needs a docs/decisions/ record.
════════════════════════════════════════════════════════════════════════

Work items (in this order):

1. **Dev deps + tooling** — add `fastapi`, `uvicorn`, `typer` to the root dev
   dependency group (mirror the documented httpx pattern: "an embeddy extra
   but a dev/test dependency"). Keep every library import LAZY. Extend
   [tool.ty.overrides] only for modules whose deps are NOT in the dev group.
   Verify the bare-import gate still passes: fresh env, no extras,
   `import embeddy` clean.
2. `embeddy/config.py` — add the `server` section: ONLY fields a code path
   actually reads (config = implementation, no dead fields). Expected
   consumers: `cors_origins: tuple[str, ...]`, `max_body_bytes: int`,
   `max_embed_inputs: int`, `max_top_k: int` (the one retained operational
   guard, CONCEPT §9.7). Precedence CLI > file > env > defaults stays (the
   `_PrecedenceSettings` base + a `load_server_settings`).
3. `embeddy/server.py` — FastAPI **factory** with a dependency-injection seam
   (`create_app(store=..., provider=..., reranker=..., settings=...)` so ASGI
   tests inject mocks) AND a module-level `app` built from settings so
   `uvicorn embeddy.server:app` works BARE. Lifespan opens the store
   (SqliteStore.open), ensures/creates collections, builds+loads the provider
   via `build_provider` from settings, closes everything on shutdown.
   Routes:
   - `GET /health/live` (process up) and `GET /health/ready` (provider
     LOADED and store open — honest readiness; 503/false when not).
   - OpenAI-compatible `POST /v1/embeddings` and `POST /v1/rerank` (the rerank
     wire shape follows providers/rerank.py — TEI/Jina; there is no OpenAI
     /v1/rerank standard).
   - `POST /api/v1/search` (+ `/similar`, `/rerank`), `/api/v1/ingest/*`
     (text/file/dir + reindex/delete/sync via IngestPipeline, returning the
     IngestStats shape), `/api/v1/collections` (create/list/stats),
     `/api/v1/chunks` (list/search within a collection).
   - Error map: pydantic ValidationError -> 400, other EmbeddyError -> 500,
     FastAPI 422 -> the structured error shape; document every mapping you add
     (e.g. unknown collection -> 404, size-limit violations -> 413).
   - CORS from config; request/batch size limits enforced (max body, max
     inputs per embed, max top_k) as OOM/crash-prevention ONLY.
   Role resolution (H2 discipline): the server is a CALLER. `/v1/embeddings`
   and `/api/v1/search` resolve the QUERY role via
   `resolve_instruction(model_id, embedder.prompt_role)` (default "query")
   and pass the resolved string to encode/search_hybrid. The DOCUMENT role is
   used only by the pipeline (ingest routes) — never by search routes. A
   regression test must prove the embed/search routes never leak the document
   role's string into a query-side call and vice versa.
4. `embeddy/client.py` — httpx `EmbeddyClient` mirroring EVERY server route
   (embed, rerank, search, ingest, collections, chunks, health). Shares
   request-building with HTTPProvider (`build_embeddings_request` — one
   protocol, never two wire shapes).
5. `embeddy/cli.py` — Typer: `serve` (run uvicorn on the app),
   `ingest text|file|dir`, `search`, `info`; config precedence
   CLI > file > env > defaults.
6. **Docs** — CLI reference + config reference for the sections that now exist
   (`embedder`, `pipeline`, `server` — only implemented fields).

Test plan (plan §8/§11):
- ASGI tests with INJECTED mocks (FakeProvider / in-memory or temp store) —
  proves the dependency-injection seam, not the real model.
- Contract tests: client ↔ live app for EVERY endpoint; payload parity (the
  client and HTTPProvider speak the same wire shape).
- Health: `ready=false` when the provider is not loaded (injected state).
- Request-limit tests: oversized body / too many inputs / top_k > max ->
  structured error, not a crash.
- Error-map tests for each mapping (400/404/413/422/500 shapes).
- CLI tests with the typer runner.
- Keystone e2e, eval gate and all M4 tests keep passing (no regressions).
- Unit (mocked) tests in packages/embeddy/tests/; ASGI/contract/CLI integration
  in tests/ tagged [integration] or a tests/contract/ subdir (follow the plan
  §11 layout); real-store search tests stay [integration].

Acceptance (plan §8): server starts bare and reports honest readiness; client
and server never drift (one protocol, shared builders); CLI documented;
server/client/CLI coverage >=90%.

Working style:
- IMPLEMENTATION_PLAN is the authority — if this prompt and the plan disagree,
  follow the plan and flag the discrepancy.
- Verify API facts empirically with tiny probe scripts BEFORE committing to an
  implementation (e.g. uvicorn bare-start, httpx ASGITransport, typer runner,
  ASGI lifespan behavior with injected mocks).
- Do NOT touch .scratch/. Do NOT commit until I review. Do NOT change the
  frozen protocols; if you believe a change is unavoidable, produce the
  docs/decisions/ record AND flag it before implementing.
- Lazy imports only for extras; zero-extras `import embeddy` must stay clean.
  New extra-only imports need [tool.ty.overrides] (existing pattern).
- `[slow]`/`[gpu]` tests must not run by default — the default suite stays
  fast and offline (all server/client/CLI tests are offline by design).
- When done, show me: git status, the test inventory (per work item), coverage
  numbers (server/client/CLI >=90%), and a summary of what was built vs
  deferred.
