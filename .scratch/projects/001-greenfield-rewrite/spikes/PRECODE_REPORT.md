# PRECODE_REPORT — chonkai + embeddy greenfield rewrite, Phase-0 spikes

**Date**: 2026-07-31 · **Repo**: `/home/andrew/Documents/Projects/embeddy`
**Status**: all five pre-code deliverables produced, run, and verified. No
`src/embeddy/` code touched. Phase 1 may begin.

**Verified run environment** (all artifacts ran under `/tmp/spike-venv`, Python
3.13, pydantic 2.x, numpy 2.5.1, sqlite-vec 0.1.9, aiosqlite 0.22.1,
tree-sitter-language-pack 1.13.7):

```bash
LSTD=$(find /nix/store -name 'libstdc++.so.6' | head -1)
LZ=$(find /nix/store -name 'libz.so.1' | head -1)
export LD_LIBRARY_PATH="$(dirname "$LSTD"):$(dirname "$LZ")"
```

> Correction to the given prefix: `libstdc++` alone is NOT sufficient — numpy
> also needs `libz.so.1`, and `find | head -1` picks among several Nix store
> copies whose directories differ. The combined prefix above is the canonical
> one (embedded in every spike header).

---

## Deliverable 1 — Tree-sitter spike (`spikes/treesitter_spike.py`, `SPIKE_RESULT.md`)

**Result: CONFIRMED the CONCEPT §4.4 revised design, with one significant
correction.** 7/7 checks pass.

- **H3 (decorators / parent / granularity): DELIVERED — but only with the
  raw-parser fallback.** `StructureItem.decorators` is declared but **never
  populated** in 1.13.7 (verified for python, javascript, typescript, rust);
  the item `.span` also excludes the decorator lines. The spike recovers
  decorators from the raw parse tree (Python `decorated_definition` wrapper;
  Rust `attribute_item` sibling nodes) and prepends them to the chunk. The
  CONCEPT's claim that "decorators stay attached" is wrong as written — the
  raw parser is **load-bearing, not a churn contingency**.
- Parent comes from the `children` tree (no `context_path` on
  `StructureItem` — per-window only, on `chunk.metadata`). Python methods are
  `kind=Function` nested under the class (JS/TS emit `Method`); normalization
  `function + parent is class → chunk_type='method'` is a Phase-2 decision.
- **H5 (oversized split): DELIVERED.** Re-run `process()` on the definition's
  byte-range slice with `chunk_max_size`; windows tile the slice with zero
  byte loss/dup; header rides on window 1; later windows carry
  `context_path=[name]`. Note: `chunk_max_size` is a **byte** budget — the
  `ValidatedChunker` token invariant is still the contract.
- Broken syntax parses; `metadata.has_error_nodes=True`; structure still emits
  the partial definition — no paragraph fallback needed (confirmed).
- **Grammar cache: all 10 target languages are BUNDLED** (`available_languages()`
  = 11, includes all targets) → offline-safe by default; the other 296
  (`manifest_languages()`) fetch lazily; unknown language → `DownloadError`.
  The CONCEPT §9.3 "despite 306 precompiled grammars" framing is slightly off:
  only ~11 are precompiled into the wheel; 306 is the download manifest.
- Other verified facts: `StructureKind` is a pyo3 enum (`str()` = `'Function'`,
  no `.value`); `ProcessResult` is a pyo3 class, not a dict; all lines are
  0-based; `span` may exclude a trailing newline.

## Deliverable 2 — Core types (`spikes/core_types.py`)

**Result: produced.** `EmbedInput = str | ImageInput`, `Vector` (float32,
L2-normalized, with `normalize_l2`/`assert_unit_vector`/`cosine_similarity`
guards), `StoredChunk`, `ScoredDocument` (carries `.metric` — `Metric` enum:
cosine/bm25/rrf/weighted/rerank), `CollectionStats`, `SourceMetadata`,
`SourceId`. No `dict[str, Any]`. The chonkai→embeddy image flow (no import)
is documented in one paragraph: chonkai keeps its own `Attachment`; embeddy
converts to `ImageInput` at the pipeline boundary.

**One deviation to note:** the file is `core_types.py`, not `types.py` — a
bare `types.py` on `sys.path` shadows the stdlib `types` module and breaks
every import (verified empirically). The M1 artifact
`embeddy/protocol/types.py` is a package-path import and has no collision.

## Deliverable 3 — Protocols (`spikes/protocols.py`, `spikes/test_protocols.py`)

**Result: produced; 22/22 tests pass.** `EmbeddingProvider` (resolved-string
instruction rule — caller resolves via registry `resolve_instruction`),
`Searchable` with the full source-op set (`upsert/get/reindex/delete/list`),
`RerankerProvider` (+ `RerankHit`), `ModelSpec`, `resolve_dimension` (exact
CONCEPT §5.2 logic), `truncate_and_renormalize`, `SearchFilters` (typed,
pre-filter). Test matrix covers: None→native, MRL in/out of range (raises),
non-MRL wrong dim (raises), registry role resolution, unit-norm after
truncation, truncation-too-long raises, and cluster-level ranking consistency
at dim 64 vs 1024.

## Deliverable 4 — Store schema (`spikes/schema.sql`, `spikes/store_schema.py`)

**Result: CONFIRMED the plan; 7/7 checks pass, all three required assertions
proven on a real aiosqlite connection.**

- `collections` / `sources` (`UNIQUE(collection_id, path)`) / `chunks`
  (`source_id` FK `ON DELETE CASCADE`) / per-collection `vec0` with
  `distance_metric=cosine` / FTS5 (`porter unicode61`, external-content) /
  `PRAGMA user_version = 1`.
- Cosine KNN distances match hand-computed cosine to 1e-5; order correct;
  `score = 1.0 - distance` only under cosine (C3 fix verified).
- **Atomic reindex swap proven both ways**: injected `MidSwapError` mid-swap →
  ROLLBACK → old chunks, old vectors, old source metadata all intact; success
  path commits cleanly.
- FTS5 proven: BM25 match + porter stemming (`'betas'` matches `'beta'`).

**New findings the backend must encode (from the spike, worth a plan note):**
1. vec0 tables with a `TEXT PRIMARY KEY` column take **no rowid** in INSERT
   (`INSERT INTO v(id, embedding) VALUES (...)`).
2. **aiosqlite runs the connection on a worker thread — the
   `db.isolation_level` property cannot be set from the caller thread**
   (sqlite3 threading check, verified). Atomicity must be expressed with the
   implicit transaction + `commit()`/`rollback()`; the spike does exactly this.

## Deliverable 5 — Eval harness (`spikes/eval/`)

**Result: produced; deterministic, zero models; 8 tests incl. the gate.**

- Fixed corpus: 20 docs (fictional "acme" platform, distinctive keywords),
  8 queries, qrels.
- `FakeProvider`: dim-8, sha256-seeded signed-hash TF-IDF — deterministic
  across runs/machines. (Dense random word vectors were measured and rejected:
  at dim 8 the noise swamps the shared-word signal; signed single-dim hashing
  is the best of six measured variants — see `SPIKE_RESULT` appendix note.)
- Runner is store-agnostic (`InMemoryStore`; duck-typed against `Searchable`).
- **Baseline: mean nDCG@10 = 0.599, mean recall@10 = 0.875**; random-provider
  control = 0.234 / 0.438. Gate thresholds 0.50 / 0.80 — clean separation,
  regression-detecting. One known brittleness: q4 ("rate limit") recalls 0.0
  (opposite-sign dim collisions at dim 8) — documented in the test.
- **Caveat (honest scoping): the fake gate is MECHANICAL** — it catches gross
  regressions (empty/garbage chunks, broken fusion, dimension mismatch) and is
  fully deterministic. Absolute retrieval quality across REAL models is gated
  by the Phase-3 `[slow]` sentence-transformers integration tests, per plan §5.

---

## Concept/plan edits now required (apply before or at Phase 1 start)

1. **CONCEPT §4.4** — rewrite the "decorators stay attached" claim: in 1.13.7
   `StructureItem.decorators`/`.visibility`/`.signature`/`.doc_comment` are
   inert; decorators MUST be recovered via the raw parser (Python
   `decorated_definition`, Rust `attribute_item` siblings) and prepended to
   chunk content. Move "raw-parser fallback" from the risk-mitigation column
   to a core component of the treesitter chunker (it is load-bearing now).
2. **CONCEPT §4.4 / plan §4 (treesitter)** — add the operational facts:
   `chunk_max_size` is a byte budget (token invariant still on
   `ValidatedChunker`); `StructureKind` is a pyo3 enum (`str()` →
   `'Function'`); `ProcessResult` is attribute-access only; all spans are
   0-based (convert to 1-based at the public boundary); `span` may exclude a
   trailing newline; 10 target grammars are bundled (offline-safe), 296 are a
   download manifest.
3. **CONCEPT §9.3** — correct the "306 precompiled grammars" phrasing: ~11 are
   precompiled; 306 is the remote manifest; v1 targets are all bundled so the
   cache-dir/offline concern applies only to non-target languages.
4. **Plan Phase 4 (aiosqlite)** — add the threading constraint: never set
   `db.isolation_level` from the caller thread; use implicit transactions +
   `commit()`/`rollback()` for the atomic reindex. Add the vec0 TEXT-PK insert
   syntax note (no `rowid`).
5. **CONCEPT §9.6 / plan §7 (eval gate)** — state that the fake-provider gate
   is mechanical (regression detection + determinism) and that absolute
   quality gates live in the `[slow]` ST tests; document the 0.50/0.80
   thresholds as derived from this fixed corpus (`eval/run_eval.py` recomputes).
6. **Repo environment** — `devenv.lock` was refreshed with `devenv update`
   (the pinned devenv module was incompatible with devenv CLI 2.1.2 and the
   shell would not evaluate; after refresh, `devenv shell` works — verified).
   This lock change is committed with the spikes.

## Artifacts committed

```
.scratch/projects/001-greenfield-rewrite/spikes/
├── core_types.py          (deliverable 2 — → embeddy/protocol/types.py at M1)
├── protocols.py           (deliverable 3 — → embeddy/protocol/* at M1, freeze M4)
├── test_protocols.py      (resolve_dimension + MRL matrix, 22 tests)
├── schema.sql             (deliverable 4 — canonical DDL, user_version=1)
├── store_schema.py        (deliverable 4 proof — aiosqlite, 7/7)
├── treesitter_spike.py    (deliverable 1 — 7/7)
├── SPIKE_RESULT.md        (deliverable 1 report)
└── eval/                  (deliverable 5 — corpus, fake provider, store,
    ├── corpus.py            runner, run_eval CLI, test_eval gate; 8 tests)
    ├── fake_provider.py
    ├── store.py
    ├── runner.py
    ├── run_eval.py
    ├── test_eval.py
    └── __init__.py
```

**Final run summary**: treesitter 7/7 · protocols 22/22 · store schema 7/7 ·
eval 8/8 (29 pytest total, 0 failures). All five deliverables confirm the
plan; the one contradiction (decorators) has a verified fix and a required
CONCEPT edit above.
