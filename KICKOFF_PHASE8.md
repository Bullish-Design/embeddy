# Kick off Phase 8 — scale path: Qdrant adapter + store selection (M7 release gate) — in this repo
(/home/andrew/Documents/Projects/embeddy). M6 is complete and pushed (commit
8bc357e on main — install matrix, lazy-import gate, docs, benchmarks,
changelog). Everything derives from IMPLEMENTATION_PLAN §10 (Phase 8) and §12
(M7 gate); the plan is the authority. Phase 8 is the SCALE PATH (post-v1):
a second `Searchable` backend (Qdrant) behind the SAME frozen M4 protocol,
store selection via config, and the LanceDB decision record.

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
   build output, `.venv/`, `.benchmarks/`.
5. Use `git rm` for tracked files, NOT raw `rm`, wherever possible. Do NOT
   commit anything until I review it.

FIRST — read these in full before touching anything:
- .scratch/projects/001-greenfield-rewrite/IMPLEMENTATION_PLAN.md  (§10
  "Phase 8 — Scale path" is your scope; §12 milestones — M7 "Scale", gate:
  Qdrant adapter + LanceDB decision recorded; §11 testing; §13 risks —
  "sqlite-vec pre-v1 breaking changes: isolate behind Searchable" (DONE at
  M4), "LanceDB API churn: spike only; decision record before adoption")
- .scratch/projects/001-greenfield-rewrite/CONCEPT.md  (§3.3 sources are
  first-class — every backend implements source ops; §5.4 the Searchable
  protocol + "Scale: Qdrant adapter. Dense + sparse vectors, payload filters,
  quantization. One config line (`store: qdrant://...`)"; §6.1 bge-m3 —
  the sparse-hybrid registry entry; §7 wire-protocol scope — learned-sparse
  is NOT expressible over OpenAI /v1/embeddings, local-provider only; §8
  packaging — the `embeddy[qdrant]` extra; §9.3 dependencies to watch)
- The CURRENT frozen surface (M4 protocols + M5/M6 product surface — Phase 8
  is a SECOND BACKEND behind the frozen contracts, NOT a reshape):
  - embeddy/protocol/types.py — typed records (EmbedInput, Vector, StoredChunk,
    ScoredDocument w/ .metric, CollectionStats, SourceMetadata, SourceId,
    SearchResult). No dict[str, Any] crosses any protocol.
  - embeddy/index/base.py — the FROZEN `Searchable` protocol INCLUDING source
    ops. The Qdrant adapter implements EXACTLY this contract (below).
  - embeddy/index/sqlite.py — SqliteStore, the default backend. BEYOND-
    PROTOCOL extras (create_collection / get_chunk / list_chunks /
    list_collections / count_fts) — the server's 501 path covers stores
    without them (embeddy/server.py ~line 804).
  - embeddy/server.py — create_app DI seam; the lifespan HARD-CODES
    `SqliteStore.open(server_settings.store_path)` (line ~340) — this is the
    integration point for store selection. Wire surface is FROZEN (all
    routes, the error map 400/404/413/422/500/501/503, CORS, size limits).
  - embeddy/client.py + embeddy/cli.py — FROZEN (mirror every route; one
    wire protocol). Phase 8 does NOT add routes.
  - embeddy/config.py — sections embedder/pipeline/server. NO `store` section
    exists yet (CONCEPT §5.9: store lands with its consuming code path —
    that path is Phase 8). Config = implementation (CONCEPT §3.8): a field
    exists only if a code path reads it.
  - embeddy/registry.py — bge-m3 ALREADY registered (dense 1024, no MRL,
    8192 ctx, query prefix card-exact). Qwen3-0.6B remains the default.
  - embeddy/providers/ — local (ST), http (OpenAI-compat, text-dense only),
    rerank, factory. Sparse (learned) is NOT implemented anywhere yet.
- The M6 state you build on: commit 8bc357e. Tests: 463 passed + 3 skipped
  in ~10s (3 skips = [slow] model-touching), coverage 96% (server 95%,
  client 97%, cli 100%, config 100%, pipeline 100%, search 100%, sqlite 97%).
  ty clean, ruff clean, lazy-import audit clean (scripts/check_lazy_imports.py),
  install matrix 7/7 (scripts/check_install_matrix.py), benchmarks 14 passed.
  CI: lint / type / test / install-matrix / benchmarks(optional) jobs.
  qdrant-client resolves to 1.18.0 against the existing `qdrant` extra
  (`qdrant-client>=1.9`); the extra install row is green but NO adapter ships.

════════════════════════════════════════════════════════════════════════
PHASE 8 SCOPE — the scale path (plan §10) + the M7 gate (plan §12).
════════════════════════════════════════════════════════════════════════
All protocols AND the M5/M6 wire surface are FROZEN — the Qdrant adapter
implements the frozen `Searchable`; it does NOT reshape it. If your work
requires reshaping EmbeddingProvider / Searchable / RerankerProvider /
search_hybrid / IngestPipeline / the chonkai public API / the server-client
wire protocol, STOP and produce a docs/decisions/ record AND flag it before
implementing (plan §12 freeze rule). The eval gate (nDCG@10 ≥ 0.50 /
recall@10 ≥ 0.80) is FROZEN — do not re-tune corpus or thresholds. The
default suite stays fast and offline.

Work items (in this order):

1. **Probe qdrant-client 1.18 empirically BEFORE designing** (the working
   style: verify API facts with tiny probes first). Install into the dev
   venv (`uv pip install "qdrant-client>=1.9"` — it is an extra, NOT a dev
   dep; add a [tool.ty.overrides] entry if a module imports it at type-check
   time... actually the adapter lives behind the `qdrant` extra, so lazy
   import + ty override follows the providers/* pattern). Facts to verify:
   - local in-memory mode (`QdrantClient(":memory:")` / `path=":memory:"`)
     — offline, no docker, usable in the DEFAULT test suite? (tag decision)
   - payload filtering: the frozen SearchFilters -> Qdrant payload filter
     translation (content_types / source_path_prefix / chunk_types /
     metadata_match). NOTE the wire model (server.py SearchFiltersModel)
     converts metadata_match dict -> tuple; the adapter consumes the
     PROTOCOL SearchFilters (tuple of (field, value) pairs) exactly as
     sqlite does.
   - sparse vectors: models.SparseVector / SparseVectorParams / named
     vectors (dense + sparse in one point), hybrid `Query` API.
   - quantization: scalar/binary quantization params API shape (plan §10:
     "quantization").
   - atomic reindex semantics: qdrant has no transactions — define the
     adapter's reindex_source contract (e.g. overwrite-by-source-id batch
     + cleanup of stale chunk ids; document the atomicity semantics — it
     can NOT be the sqlite one-transaction swap; the PROTOCOL says "atomic
     swap ... on failure the old chunks must remain intact and queryable"
     — probe what qdrant offers and DOCUMENT the achievable guarantee).
   - the beyond-protocol extras: which of create_collection / get_chunk /
     list_chunks / list_collections / count_fts the adapter supports (501
     for the rest, matching the server's 501 contract). Qdrant collections
     are implicit on first upsert; decide the create_collection mapping.

2. **QdrantStore** (`embeddy/index/qdrant.py`, lazy `qdrant_client` import
   behind the `embeddy[qdrant]` extra): a `Searchable` implementation with
   dense + sparse + payload filters + quantization as probed. Conform
   EXACTLY to the frozen protocol (see the full signature in §"The frozen
   protocol" below). All source ops (upsert/get/reindex/delete/list). Match
   the SqliteStore semantics where the protocol pins them: ScoredDocument
   carries `.metric` (cosine / bm25 / ...); `min_score` interpreted in the
   metric's true semantics; `total_results` semantics live in search_hybrid
   (not the store) — no drift.

3. **Store selection + config** — CONCEPT §5.4 "one config line":
   - a `store` section in config.py (`store.url` or similar, e.g.
     `store: qdrant://localhost:6333` / `qdrant://:memory:` / sqlite DSN
     or default) — EVERY field must have a code path reading it (§3.8).
   - a `build_store(...)` factory mirroring `build_provider`
     (providers/factory.py pattern) mapping the DSN to SqliteStore |
     QdrantStore.
   - wire it into the server lifespan (replace the hard-coded
     SqliteStore.open at server.py ~line 340) and the CLI `serve` path.
     The bare `app = create_app()` must still work with the sqlite default.
   - the `store_path` server setting stays (sqlite default path); decide
     whether it folds into the `store` DSN or remains (document either way).
   - `store: qdrant://...` with the local provider must work end-to-end via
     `embeddy serve`; the honest-health contract (not ready + reason) must
     hold when qdrant is unreachable.

4. **Sparse path — DECISION REQUIRED (verify before committing):** plan §10
   says "dense + sparse". bge-m3's learned sparse comes from FlagEmbedding
   (or a custom sparse encoder), NOT from sentence-transformers' encode().
   Probe what is achievable: sentence-transformers 3.x cannot emit
   learned-sparse for bge-m3 without the FlagEmbedding package. Decide and
   record (docs/decisions/):
   (a) implement a minimal learned-sparse encoder for bge-m3 (new optional
       dep? heavy?), or
   (b) ship dense-only on Qdrant for v2 with sparse documented as a
       follow-up (the QdrantStore's sparse plumbing still exists but is
       exercised by tests with synthetic sparse vectors), or
   (c) something in between. The EmbeddingProvider protocol has no sparse
       channel — a sparse extension CHANGES the frozen protocol and needs
       the decision record + flag FIRST. Prefer (b) unless the probe makes
       (a) trivial. The FakeProvider + eval gate MUST NOT change.

5. **LanceDB decision record (M7 gate item)** — docs/decisions/0001 says
   "revisit at Phase 8 / M7 if corpus size or query latency becomes a
   problem". Record the revisit (docs/decisions/0003-lancedb-revisit.md or
   amend 0001): the Phase-4 spike numbers (sqlite-vec ~2× slower at 1M,
   ~2.4× slower to ingest, within ~20% on disk), the Qdrant adapter as the
   designed scale path, and the decision (likely: do NOT adopt LanceDB;
   Qdrant is the scale path). No new spike runs unless the plan demands it.

6. **Integration adapters (optional, post-v1, plan §10)** — Haystack
   components / LlamaIndex vector-store interface. Likely DEFER with a
   documented note in the decision record / CHANGELOG. Do NOT build them
   unless asked; flag it if you believe they belong in this phase.

7. **Tests** — no regression in the default suite:
   - unit: payload-filter translation, DSN parsing, reindex semantics,
     quantization params, beyond-protocol extras mapping (mock-free).
   - integration (tag [integration], offline via qdrant local mode IF the
     probe confirms it is hermetic; otherwise a separate opt-in dir like
     tests/slow — NEVER docker-required in the default suite): dense
     search parity vs SqliteStore on the same vectors, source ops,
     pre-filter recall (restrictive filters return full top_k — the M3
     recall contract), reindex failure leaves old chunks queryable.
   - contract/e2e/eval: UNCHANGED and green (the server tests inject a
     store; a QdrantStore-injected server test set is new, not a rewrite).
   - the install-matrix `embeddy[qdrant]` row keeps passing.
   - benchmarks unchanged (benchmarks/ never in the default suite).

8. **Docs (docs must MATCH — M7 gate)** — only implemented fields:
   - config.md: the `store` section (fields with their reading code paths).
   - USER_GUIDE: store selection (one config line), Qdrant caveats (local
     mode, no auth on qdrant, reindex semantics), sparse status per the
     decision.
   - INTEGRATION: qdrant as the storage backend (server-side, not a wire
     change — the wire surface is untouched).
   - ARCHITECTURE: the second backend + store factory + decision records.
   - CHANGELOG + docs/versioning: the M7 release entry.
   - No aspirational claims (no auth, no multi-backend failover unless
     implemented).

9. **M7 gate** — assemble and verify: Qdrant adapter implements the full
   frozen protocol (dense + sparse + payload filters + quantization, source
   ops); store selection config works end-to-end (`store: qdrant://:memory:`
   and a real `qdrant://host:port` when available); LanceDB decision
   recorded; docs match; all existing tests keep passing; coverage does not
   regress below the M6 numbers (server 95%, client 97%, cli 100%, total
   96% — QdrantStore lands with >=90% of its own lines covered, or justify).

════════════════════════════════════════════════════════════════════════
THE FROZEN PROTOCOL (implement this, do not reshape it — index/base.py)
════════════════════════════════════════════════════════════════════════
```python
@runtime_checkable
class Searchable(Protocol):
    # --- chunks ---
    async def add(self, collection: str, chunks: list[StoredChunk],
                  vectors: list[Vector]) -> None: ...
    async def delete(self, collection: str, chunk_ids: list[str]) -> None: ...
    # --- search ---
    async def search_vector(self, collection: str, query_vector: Vector,
                            filters: SearchFilters, top_k: int, *,
                            min_score: float | None = None) -> list[ScoredDocument]: ...
    async def search_fts(self, collection: str, query: str, filters: SearchFilters,
                         top_k: int, *, min_score: float | None = None,
                         raw: bool = False) -> list[ScoredDocument]: ...
    # --- collections ---
    async def stats(self, collection: str) -> CollectionStats: ...
    # --- sources (first-class, every backend implements them) ---
    async def upsert_source(self, collection: str, source: SourceMetadata) -> SourceId: ...
    async def get_source(self, collection: str, path: str) -> SourceMetadata | None: ...
    async def reindex_source(self, collection: str, source: SourceMetadata,
                             chunks: list[StoredChunk], vectors: list[Vector]) -> None: ...
    async def delete_source(self, collection: str, source_id: SourceId) -> None: ...
    async def list_sources(self, collection: str) -> list[SourceMetadata]: ...
```
`SearchFilters` (frozen): content_types: tuple[str,...] | source_path_prefix:
str|None | chunk_types: tuple[str,...] | metadata_match: tuple[tuple[str,str],...].
`min_score` semantics: cosine scores in [0,1] higher=better; BM25 rank <= 0
higher/less-negative=better — NEVER comparable across metrics (CONCEPT §3.4).
`reindex_source` = atomic swap; sqlite does it in one transaction — qdrant has
NO transactions, so document the adapter's achievable guarantee and prove the
"old chunks remain intact on failure" property as far as the backend allows.

════════════════════════════════════════════════════════════════════════
TEST PLAN (plan §9/§11 — reuse the M4/M5 test discipline)
════════════════════════════════════════════════════════════════════════
- Default suite: 463 tests keep passing; eval gate green; coverage >= M6
  numbers; ty clean (with the new [tool.ty.overrides] for the lazy qdrant
  import); ruff clean; lazy-import audit clean (qdrant_client must NOT
  appear at module level anywhere except where the extra contract allows —
  follow the providers/* lazy pattern, NOT the server/cli entry-point
  carve-out, unless the adapter module itself IS a documented entry point —
  it is not; keep it lazy).
- Qdrant layer: offline integration tests (local mode) tagged [integration]
  if hermetic, else a dedicated opt-in dir — NEVER the default suite if it
  needs docker/network. Pre-filter recall parity (restrictive filter returns
  full top_k), source-op lifecycle, reindex failure semantics, stats parity.
- Install matrix: qdrant row green (already green — keep it).
- Benchmarks: unchanged; do not add timing gates.

════════════════════════════════════════════════════════════════════════
ACCEPTANCE (plan §10/§12)
════════════════════════════════════════════════════════════════════════
Qdrant adapter implements the frozen Searchable (dense + sparse + payload
filters + quantization) with offline-tested source ops; store selection
works via one config line; LanceDB decision recorded; docs match; all
existing gates green; no frozen protocol or wire-shape changes without a
docs/decisions/ record + flag.

Working style (unchanged from M6):
- IMPLEMENTATION_PLAN is the authority — if this prompt and the plan
  disagree, follow the plan and flag the discrepancy.
- Verify API facts empirically with tiny probe scripts BEFORE committing to
  an implementation (qdrant local mode, payload filter translation, sparse
  + quantization params, reindex atomicity limits).
- Do NOT touch .scratch/. Do NOT commit until I review. Do NOT change the
  frozen protocols or the wire surface; if you believe a change is
  unavoidable, produce the docs/decisions/ record AND flag it first.
- Lazy imports only for extras; zero-extras `import embeddy` must stay
  clean; new extra-only imports need [tool.ty.overrides] (the providers/*
  pattern) unless the dep is in the dev group.
- `[slow]`/`[gpu]` tests must not run by default; the default suite stays
  fast and offline; benchmarks and extras-matrix checks never run in the
  default pytest suite.
- When done, show me: git status, the test inventory, coverage numbers
  (per-module, no regressions), the install-matrix results, the probe
  evidence for the qdrant API facts, and a summary of built vs deferred.
