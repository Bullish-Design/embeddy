# Benchmarks — search backends (Phase-4 spike, plan §6 work item 6)

`bench_backends.py` compares the two embedded vector-store candidates that
decide the default backend (plan §4/§8):

| Backend | Role |
|---------|------|
| **sqlite-vec + FTS5** (embeddy `SqliteStore`) | the incumbent default — what embeddy ships |
| **LanceDB** | the documented spike candidate (embedded vector+FTS+hybrid in one package) |

The result is recorded in `docs/decisions/0001-default-search-backend.md`.

## Why LanceDB is not in uv.lock

LanceDB is a **spike-only dependency**: `pip install lancedb` into the dev
venv by hand (`uv run pip install lancedb`). It is NOT a project dependency —
`uv sync` removes it, exactly like sentence-transformers. The benchmark
imports it lazily inside `bench_lancedb()` and ty is configured to ignore the
spike module (`[tool.ty.overrides]` in the root pyproject, `benchmarks/**`).

## Running

Inside `devenv shell`, with the canonical library path prefix (numpy needs
libz, not just libstdc++):

```bash
export LD_LIBRARY_PATH="$(dirname $(find /nix/store -maxdepth 3 -name 'libstdc++.so.6' | head -1)):$(dirname $(find /nix/store -maxdepth 3 -name 'libz.so.1' | head -1))"
uv run python benchmarks/bench_backends.py --n 100000 --dim 256 --json /tmp/bench_100k.json
uv run python benchmarks/bench_backends.py --n 1000000 --dim 256 --json /tmp/bench_1m.json
```

`--n` = number of vectors, `--dim` = embedding dimension, `--topk` = search
top_k (default 50). Each run measures ingest wall-time, on-disk size, and
warm search latency (p50/p95 over 60 queries) for unfiltered / filtered
(chunk_type + content_type EQ) / prefix-filtered (path) top-k, plus LanceDB's
FTS-index build and hybrid latency.

## Measured results (2026-08-01, this machine: 8 cores, 62 GB RAM)

### n=100,000, dim=256, top_k=50

| metric | sqlite-vec | LanceDB |
|--------|-----------|---------|
| ingest (s) | 27.7 | 12.2 |
| size (bytes) | 131,256,320 | 109,883,978 |
| search top-k ms (p50/p95) | 67.7 / 90.1 | 53.4 / 120.7 |
| filtered ms (p50/p95) | 77.1 / 106.5 | 81.2 / 117.7 |
| prefix-filtered ms (p50/p95) | 79.5 / 93.6 | 80.4 / 107.3 |
| FTS index build (s) | — | 0.5 |
| hybrid ms (p50/p95) | — | 56.5 / 68.5 |

### n=1,000,000, dim=256, top_k=50

| metric | sqlite-vec | LanceDB |
|--------|-----------|---------|
| ingest (s) | 296.1 | 123.8 |
| size (bytes) | 1,315,676,160 | 1,101,851,438 |
| search top-k ms (p50/p95) | 696.2 / 768.5 | 368.1 / 482.0 |
| filtered ms (p50/p95) | 748.4 / 812.6 | 578.6 / 683.9 |
| prefix-filtered ms (p50/p95) | 741.1 / 808.8 | 583.1 / 647.7 |
| FTS index build (s) | — | 6.3 |
| hybrid ms (p50/p95) | — | 390.1 / 485.3 |

Reading: both are brute-force flat scans (sqlite-vec 0.1.9 ships no ANN
index; LanceDB flat search is arrow/SIMD-accelerated). Search latency scales
linearly with N — ~0.7 s / 0.4 s at 1M×256 — and neither is interactive-
critical for a personal-project corpus. sqlite-vec is ~2× slower at 1M and
~2.4× slower to ingest (text-serialized vectors), within ~20% on disk.

## Recall note

Both backends pre-filter during the scan: a restrictive filter returns the
full top_k. For sqlite-vec this required the vec0 auxiliary columns
(probed; `LIKE` is rejected in-scan, `EQ/IN/comparison` honored); LanceDB
pushdowns predicates into its flat scan. The M3 recall hole is fixed on the
embeddy side in `tests/test_store_phase4.py::test_prefilter_recall_*`.
