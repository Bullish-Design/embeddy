# 0003 — LanceDB revisit at M7: still not adopted; Qdrant is the scale path

- **Date**: 2026-08-01
- **Status**: Accepted (the M7 revisit of `0001-default-search-backend.md`,
  recorded at the Phase-8 gate per the 0001 "Revisit: Phase 8 / M7" note)
- **Revisit**: only if a future sqlite-vec release ships a real ANN index
  that changes the benchmark, or if the corpus/query regime materially
  changes (see 0001 Consequences)

## Context

`0001` (M4) chose sqlite-vec + FTS5 as the default backend and named Qdrant
the designed scale path (CONCEPT §5.4), with a revisit at Phase 8 / M7 "if
corpus size or query latency becomes a problem". Phase 8 is the scale path:
the Qdrant adapter (`0004`) is now implemented. This record closes the
LanceDB question at M7.

## Decision

**Do NOT adopt LanceDB.** The scale path is the Qdrant adapter
(`embeddy/index/qdrant.py`, `0004`), not a second embedded backend. No new
spike runs were needed: the Phase-4 numbers below already bound the
question, and nothing about embeddy's corpus regime (personal-project
scale, one-user server on a trusted tailnet) changed since.

## Evidence (from the Phase-4 spike, `benchmarks/bench_backends.py`)

| n, dim=256, top_k=50 | sqlite-vec | LanceDB |
|---|---|---|
| ingest 100k / 1M (s) | 27.7 / 296 | 12.2 / 124 |
| size 100k / 1M (bytes) | 131M / 1.32G | 110M / 1.10G |
| search p50 100k / 1M (ms) | 68 / 696 | 53 / 368 |
| filtered p50 100k / 1M (ms) | 77 / 748 | 81 / 579 |

sqlite-vec is ~2× slower at 1M and ~2.4× slower to ingest, within ~20% on
disk. LanceDB's API churn (three breaking surface moves within the probe
window) and its pyarrow footprint (~100 MB of wheels) remain disqualifying
at this scale — the 0001 analysis stands unchanged.

## Why Qdrant, not LanceDB, is the scale path

1. **Managed, not embedded.** Qdrant runs as a server (or in-process
   `:memory:` for tests); it takes the storage engine out of embeddy's
   dependency graph instead of adding another embedded engine (pyarrow +
   Lance file format) alongside sqlite-vec.
2. **Sparse + quantization as first-class features.** Qdrant's named sparse
   vectors and collection-level quantization map directly to the plan §10
   scope ("dense + sparse, payload filters, quantization"); LanceDB has no
   sparse-vector support.
3. **The Searchable protocol is the isolation seam.** Both backends
   implement the SAME frozen protocol (0001's sqlite-vec isolation
   mitigation, now proven by the Qdrant adapter); the choice is a config
   line (`store.url`), not a code fork.

## Consequences

- LanceDB stays a spike-only dependency (`benchmarks/`, manual install,
  ty override, never in uv.lock) — unchanged from 0001.
- The default backend remains sqlite-vec + FTS5 (zero-config).
- If a future sqlite-vec release ships a real ANN index, re-benchmark
  before any default-backend change; Qdrant remains the documented scale
  path until then.
- See `0004` for the Qdrant adapter's decisions (sparse status, reindex
  atomicity, the FTS path).
