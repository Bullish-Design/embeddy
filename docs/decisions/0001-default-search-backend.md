# 0001 — Default search backend: sqlite-vec + FTS5 (LanceDB spike decision)

- **Date**: 2026-08-01
- **Status**: Accepted (recorded at M4, plan §4/§8 decision point)
- **Revisit**: Phase 8 / M7 (scale path) — see Consequences

## Context

The storage layer needs a default embedded backend. sqlite-vec + FTS5 is
the incumbent (Phase-1 slice, spike-proven schema); LanceDB is the
documented spike candidate (CONCEPT §5.4, plan §8) — "embedded
vector+FTS+hybrid in one package" is its selling point. The plan §13 risk
register flags both sqlite-vec (pre-v1 breaking changes) and LanceDB
(API churn). The decision must be evidence-based: a benchmark spike at
100k and 1M vectors with filters (`benchmarks/bench_backends.py`),
plus verified API facts from probes.

## Decision

**Keep sqlite-vec + FTS5 as the default Searchable backend for v1.**
Do NOT adopt LanceDB as a second backend now. The LanceDB spike artifacts
stay in `benchmarks/` (script + measured results); the decision is
revisited at the Phase-8 scale path (M7) if corpus size or query latency
becomes a problem, at which point the designed scale path is the Qdrant
adapter (CONCEPT §5.4), not a second embedded backend.

## Evidence

### Verified API facts (probes, 2026-08-01)

sqlite-vec 0.1.9:

- vec0 auxiliary columns pre-filter **in-scan** (`EQ`/`IN`/`NOT_EQUALS`/
  range comparisons); `LIKE` is rejected with "Only one of EQUALS,
  GREATER_THAN, LESS_THAN_OR_EQUAL, LESS_THAN, GREATER_THAN_OR_EQUAL,
  NOT_EQUALS". Path prefixes compile to a half-open byte-range constraint
  (probed, incl. multi-byte/emoji suffixes).
- KNN `distance` constraints are honored in-scan (`distance <= 1 - min_score`
  pushes the cosine `min_score` threshold into the scan).
- Auxiliary columns **reject NULL** — the store coerces optional fields to "".
- A JOIN-then-filter KNN query (the M3 shape) returns fewer than k rows under
  a restrictive filter (43/50 probed) — the recall hole this phase closes.

LanceDB 0.36 (installed spike-only in the dev venv):

- `create_fts_index()` is **deprecated** as of 0.25.0 → `create_index("col",
  config=FTS())`; the `FTS()` config takes no column argument and no
  `replace=` (the signature moved twice).
- Hybrid search: `table.search(query_type="hybrid").vector(v).text(t)` —
  the `vector=`/`query=` keyword-arg shape raises TypeError in 0.36.
- Filters push down into the flat scan (`where("chunk_type = '…'")`,
  `LIKE` supported); `LanceModel` declares the vector column via a runtime
  `Vector(dim)` call.

### Benchmark (this machine, 8 cores / 62 GB RAM; full tables in
`benchmarks/README.md`)

| n, dim=256, top_k=50 | sqlite-vec | LanceDB |
|---|---|---|
| ingest 100k / 1M (s) | 27.7 / 296 | 12.2 / 124 |
| size 100k / 1M (bytes) | 131M / 1.32G | 110M / 1.10G |
| search p50 100k / 1M (ms) | 68 / 696 | 53 / 368 |
| filtered p50 100k / 1M (ms) | 77 / 748 | 81 / 579 |

Both are flat scans; latency scales linearly with N. sqlite-vec is ~2×
slower at 1M and ~2.4× slower to ingest (text-serialized vectors), within
~20% on disk. Neither is interactive-critical at personal-project scale.

## Why sqlite-vec wins for v1

1. **Zero new footprint, one-wire FTS.** FTS5 (porter + unicode61, BM25,
   external content) is already proven in this repo (spike-verified stemming,
   `tests/test_store_phase4.py`). LanceDB's tantivy FTS would require
   re-verifying tokenization/stemming parity for a feature embeddy already
   owns.
2. **API surface within our control.** sqlite-vec's behaviors were verified
   empirically once and are isolated behind `Searchable` (plan §13
   mitigation). LanceDB's API moved three times within the probe window
   (FTS index, hybrid search, FTS config) — a live churn risk for a
   personal project with no dedicated maintenance budget.
3. **Dependency discipline.** sqlite-vec/aiosqlite are small core deps
   already; zero-extras `import embeddy` stays clean. LanceDB drags pyarrow
   (~100 MB of wheels) and its own storage engine into the dependency graph
   for a feature we do not need at v1 scale.
4. **Storage model.** sqlite-vec is one .db file (trivially backed up /
   inspected with any sqlite tool); LanceDB is a directory tree with version
   files. The re-ingest-from-source policy (no migration) favors the simpler
   file.
5. **Performance parity within the target range.** 100k×256 both ~50–80 ms;
   at 1M the ~2× gap is acceptable for interactive single-user use, and the
   designed scale path (Qdrant, Phase 8) exists for real scale.

## Consequences

- LanceDB remains a documented spike-only dependency (`benchmarks/`,
  manual pip install, ty override, NOT in uv.lock). `uv sync` removes it.
- The `SqliteStore` pre-filter design (vec0 auxiliary columns, byte-range
  prefix constraints, distance pushdown) is the accepted implementation and
  is NOT changed by this decision.
- At M7, before adopting LanceDB or Qdrant, rerun
  `benchmarks/bench_backends.py` at the projected corpus size; revisit this
  record if sqlite-vec ingest or query latency becomes the bottleneck.
- If a future sqlite-vec release ships a real ANN index, re-benchmark before
  any scale-path decision.
