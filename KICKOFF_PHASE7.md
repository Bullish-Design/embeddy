# Kick off Phase 7 — packaging, docs, benchmarks (M6 release gate) — in this repo
(/home/andrew/Documents/Projects/embeddy). M5 is complete and pushed (commit
d937175 on main — server, client, CLI: honest health, client parity, Typer
CLI). Everything derives from IMPLEMENTATION_PLAN §9 (Phase 7) and §12 (the
M6 gate); the plan is the authority. Do NOT skip ahead to Phase 8 (Qdrant
adapter / scale path).

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
- .scratch/projects/001-greenfield-rewrite/IMPLEMENTATION_PLAN.md  (§9 "Phase 7 —
  Packaging, docs, benchmarks" is your scope; §12 milestones — M6 "Release",
  gate: install matrix green, docs match, benchmarks run; §11 testing —
  benchmark layer tools are pytest-benchmark + psutil; §13 risks)
- .scratch/projects/001-greenfield-rewrite/CONCEPT.md  (§8 packaging rules:
  extras split, lazy imports, one-way dependency, importable with zero
  extras; §9.6 the retrieval-quality eval harness — deterministic, offline,
  MECHANICAL gate with 0.50 nDCG@10 / 0.80 recall@10 thresholds; §3.8
  "config = implementation" — docs must list ONLY implemented fields)
- The CURRENT frozen surface (M4 protocols + M5 product surface — Phase 7 is
  VERIFICATION, DOCUMENTATION and MEASUREMENT, NOT new features):
  - embeddy/server.py — create_app(store=, provider=, reranker=, settings=,
    embedder=, pipeline=) DI seam + module-level bare `app`; lifespan opens
    store / builds+loads provider / closes on shutdown; /health/live +
    /health/ready (honest); OpenAI-compatible /v1/embeddings, TEI/Jina
    /v1/rerank, /api/v1/{search,similar,rerank,ingest/*,collections,chunks};
    structured error map (400/404/413/422/500/501/503); CORS from config;
    size limits (OOM prevention only). ⚠️ server.py imports fastapi/starlette
    at MODULE level — it IS the server-extra entry point; embeddy/__init__.py
    never imports server/client/cli (that is what keeps zero-extras
    `import embeddy` clean).
  - embeddy/client.py — EmbeddyClient, mirrors every route, shares
    build_embeddings_request with HTTPProvider (one wire protocol); httpx
    imported lazily inside methods.
  - embeddy/cli.py — Typer: serve / ingest text|file|dir / search / info;
    typer imported at MODULE level (the CLI-extra entry point); console
    script `embeddy` (server extra). Config precedence CLI > file > env >
    defaults.
  - embeddy/config.py — EmbedderSettings / PipelineSettings / ServerSettings
    on `_PrecedenceSettings` (CLI > file > env > defaults,
    dotenv_filtering="match_prefix"); load_settings / load_pipeline_settings /
    load_server_settings.
  - M4 core (all FROZEN): protocol/ (embedding.py, rerank.py, types.py),
    registry.py, index/base.py + index/sqlite.py (incl. the beyond-protocol
    extras create_collection / count_fts / get_chunk / list_chunks /
    list_collections), index/filters.py, index/sources.py, search.py
    (search_hybrid + fuse_*), pipeline.py (IngestPipeline, IngestStats),
    providers/ (factory, http, local, rerank, fake — lazy extras),
    errors.py (incl. ClientError), budget.py. chonkai public API
    (Ingestor, get_chunker, ValidatedChunker, ChunkBudget, ...).
- tests/contract/ (conftest + test_server/test_client/test_cli — the M5
  contract layer), tests/e2e/test_keystone.py, eval/test_eval.py (the M4
  retrieval gate — must keep passing in the DEFAULT suite; do not regress).

════════════════════════════════════════════════════════════════════════
CURRENT STATE (M6, committed d937175) — build on this, don't rebuild it:
════════════════════════════════════════════════════════════════════════
uv workspace, Python >=3.11, dev env Python 3.13 via devenv. All commands run
INSIDE `devenv shell` (uv is at `.devenv/profile/bin/uv` if PATH is bare).
Pins unchanged (pydantic 2.12.5, numpy 2.3.5, sqlite-vec 0.1.9, aiosqlite
0.22.1, pydantic-settings 2.14.2).

- Gate status at M5: **463 passed + 3 skipped** (the 3 skips are the [slow]
  model-touching tests — correct). Coverage: server 95%, client 97%, cli 100%,
  config 100%, total 96% (M5 core >=90%; M4 core: pipeline 100%, search 100%,
  sqlite 97%). ruff clean, ty clean, format clean, one-way dependency grep
  clean, zero-extras import clean.
- Test layout: unit in packages/embeddy/tests/ and packages/chonkai/tests/
  (no DB), integration in tests/ tagged [integration], contract in
  tests/contract/ tagged [integration], e2e in tests/e2e/ ([e2e]), slow in
  tests/slow/ ([slow]/[gpu] opt-in). asyncio_mode=auto.
- Dev group (root pyproject): pytest / pytest-asyncio / pytest-cov /
  hypothesis / ruff / ty / httpx / fastapi / uvicorn / typer — every one
  follows the precedent: "an embeddy extra but a dev/test dependency".
  pytest-benchmark and psutil are NOT in the dev group (plan §11 names them
  for the benchmark layer — work item 4 holds the DECISION).
- ⚠️ ENV QUIRKS (verified repeatedly): a plain `uv run python probe.py` cannot
  import numpy without the prefix:
  LSTD=$(dirname $(find /nix/store -maxdepth 3 -name 'libstdc++.so.6' | head -1))
  LZ=$(dirname $(find /nix/store -maxdepth 3 -name 'libz.so.1' | head -1))
  export LD_LIBRARY_PATH="$LSTD:$LZ"
  sentence-transformers AND lancedb are installed MANUALLY in the dev venv
  only — NOT in uv.lock; `uv sync` removes them (correct: the [slow] tests
  skip and the benchmarks require a manual `uv pip install lancedb`).
- Type checking is **ty**. [tool.ty.overrides] exists for every lazy-import /
  extra-only module (providers/*, docling, tokens, tests with PIL,
  benchmarks/**). New modules that import extra-only deps follow the pattern;
  modules whose deps ARE in the dev group need no override.
- The eval harness is deterministic, offline, runs in the DEFAULT suite (do
  NOT tag it [slow]); thresholds 0.50 nDCG@10 / 0.80 recall@10 are derived
  from the fixed 20-doc corpus and recomputable via `eval/run_eval.py`
  (baseline 0.599 / 0.875; do not re-tune the corpus).
- benchmarks/: bench_backends.py is the Phase-4 LanceDB-vs-sqlite SPIKE
  (spike-only deps, manually installed; `uv sync` removes them). It is NOT
  the plan §9 benchmark harness.
- CI: .github/workflows/ci.yml — lint (ruff + one-way dep grep), type (ty),
  test (pytest coverage gate), import-smoke (fresh venv, zero extras).
- docs/: cli.md + config.md (M5) + decisions/0001-default-search-backend.md.
  USER_GUIDE / INTEGRATION / ARCHITECTURE DO NOT EXIST yet.
- CHANGELOG.md does not exist; both packages at 0.1.0.

════════════════════════════════════════════════════════════════════════
PHASE 7 SCOPE — packaging, docs, benchmarks (plan §9) + the M6 gate (plan
§12). Goal: a fresh-env install matrix that is green; docs that MATCH the
implementation; benchmarks that are runnable and wired to CI (optional job);
a changelog + versioning policy (release train: chonkai + embeddy together).
All protocols AND the M5 wire surface are FROZEN — if your work requires
reshaping EmbeddingProvider / Searchable / RerankerProvider / search_hybrid /
IngestPipeline / the chonkai public API / the server-client wire protocol,
STOP and flag it; any change after the freeze needs a docs/decisions/ record.
════════════════════════════════════════════════════════════════════════

Work items (in this order):

1. **Extras + wheel audit** — the install matrix (plan §9, §12 M6 gate):
   - fresh venv per row: zero extras (`pip install ./packages/chonkai` then
     `./packages/embeddy`), then each extra: `embeddy[server]`,
     `embeddy[client]`, `embeddy[local]`, `embeddy[qdrant]`, `chonkai[docling]`,
     `chonkai[tokenizers]` — each must install cleanly and `import embeddy`
     (and `import chonkai`) stays clean with zero extras.
   - wheel contents inspected (`uv build` / `unzip -l` the wheels): no
     accidental heavy deps in CORE (torch / transformers / docling must not
     be core requirements of either package); the wheel ships only
     src/ packages (hatchling packages = ["src/embeddy"] / ["src/chonkai"]).
   - extend the CI import-smoke job to cover the matrix, or add a job —
     decide and document. (Note: qdrant-client is a real dependency of the
     `qdrant` extra; the extra must resolve even though the adapter lands in
     Phase 8 — the plan declares it at §2.)
2. **Lazy-imports audit + CI gate** — plan §9: grep for
   `import torch|transformers|docling|fastapi` etc.; extra-only imports must
   live ONLY in lazy-loading modules. The audit must accommodate the two
   KNOWN module-level extra entry points: embeddy/server.py (fastapi,
   starlette at module level) and embeddy/cli.py (typer at module level).
   Decide the exact check (a script or grep rule asserting the set of
   modules importing extra-only deps at module level == {server, cli} ∪
   TYPE_CHECKING-only), wire it into CI, and keep the zero-extras
   `import embeddy` smoke green (embeddy/__init__.py must never import
   server/client/cli).
3. **Docs** — USER_GUIDE, INTEGRATION, ARCHITECTURE for the greenfield
   architecture (they do not exist; "rewrite" = write for the new repo).
   docs/cli.md and docs/config.md exist at M5 — verify they MATCH the code
   (M6 gate: "docs match"); extend only for implemented fields (CONCEPT §3.8:
   no dead fields). Only implemented fields, nothing aspirational (no Qdrant
   adapter, no auth).
4. **Benchmarks** — plan §9/§10.10 + §11: a chonkai chunk-quality harness and
   an embeddy search/ingest/resource benchmark harness (the existing
   bench_backends.py is the Phase-4 SPIKE, not the harness). Tools per plan
   §11: pytest-benchmark + psutil. DECISION REQUIRED: add pytest-benchmark /
   psutil to the root dev group (the httpx/fastapi precedent) or keep them
   out of the default suite in a separate benchmark job. Benchmarks must
   NOT run in the default pytest suite (they would blow up the ~15s gate) —
   a separate CI job (optional per plan §9) runs them. Verify empirically
   with tiny probes before committing (e.g. pytest-benchmark's default
   fixture behavior).
5. **Retrieval-quality eval harness** — landed at M4 and already runs in the
   default suite; Phase 7 verifies it is the documented pre-release gate:
   thresholds, how to recompute (eval/run_eval.py), and what a regression
   means go into a doc if not already there. Do NOT re-tune the corpus or the
   thresholds.
6. **Changelog + versioning policy** — CHANGELOG.md covering both packages
   (release train: chonkai + embeddy together initially, plan §9); document
   the policy (README or docs/) — versioning scheme, what triggers a release,
   the M6 install-matrix gate.
7. **M6 gate** — assemble and verify: install matrix green (fresh envs),
   docs match implementation (audit the docs against the code), benchmarks
   runnable, changelog present. All existing tests keep passing; coverage
   does not regress.

Test plan (plan §9/§11):
- The existing 463-test suite keeps passing — Phase 7 adds verification and
  measurement, not features (no regressions; the keystone e2e + eval gate
  stay green in the default suite).
- Extras matrix: scripted checks (fresh venv per extra) — likely a CI job
  extending the import-smoke.
- Lazy-import audit: a CI check (grep/script) — and the zero-extras smoke.
- Docs-match: a review checklist against the code (M6 gate), not a unit test.
- Benchmarks: runnable + a smoke run (separate job / [slow]-style opt-in,
  NEVER the default suite).
- Coverage: core modules stay >=90% (server/client/cli >=90% — do not regress
  the M5 numbers: server 95%, client 97%, cli 100%).

Acceptance (plan §9/§12): fresh-env install matrix green; docs match
implementation; benchmarks runnable and wired to CI (optional job); eval
gate wired + documented; changelog + versioning policy documented; M6 gate
green; no frozen protocol or wire-shape changes.

Working style:
- IMPLEMENTATION_PLAN is the authority — if this prompt and the plan disagree,
  follow the plan and flag the discrepancy.
- Verify API facts empirically with tiny probe scripts BEFORE committing to an
  implementation (e.g. pytest-benchmark fixtures, `uv build` wheel contents,
  pip install of each extra into a fresh venv, hatchling sdist/wheel layout).
- Do NOT touch .scratch/. Do NOT commit until I review. Do NOT change the
  frozen protocols or the wire surface; if you believe a change is
  unavoidable, produce the docs/decisions/ record AND flag it before
  implementing.
- Lazy imports only for extras; zero-extras `import embeddy` must stay clean.
  New extra-only imports need [tool.ty.overrides] (existing pattern) unless
  the dep is in the dev group.
- `[slow]`/`[gpu]` tests must not run by default — the default suite stays
  fast and offline; benchmarks and extras-matrix checks must not run in the
  default pytest suite either.
- When done, show me: git status, the test inventory, coverage numbers
  (per-module, no regressions), the install-matrix results, and a summary of
  what was built vs deferred.
