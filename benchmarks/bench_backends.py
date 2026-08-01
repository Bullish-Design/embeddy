#!/usr/bin/env python3
"""Backend benchmark spike — sqlite-vec+FTS5 (embeddy SqliteStore) vs LanceDB.

Plan §4/§8 / §13: "choose the default backend" — sqlite-vec is the incumbent
default; LanceDB is the documented spike candidate (embedded vector+FTS+
hybrid in one package). This script measures what the decision record
(docs/decisions/0001-default-search-backend.md) cites:

  * ingest wall-time at N vectors (the re-ingest-from-source policy makes
    write speed a real factor — no in-place migration)
  * on-disk size
  * search latency p50/p95 (warm): unfiltered top-k, filtered top-k
    (chunk_type EQ), prefix-filtered (path), all with FULL top-k recall
  * filter recall: restrictive filters must return the full k (the M3
    recall-hole regression this phase fixes)

LanceDB is a SPIKE-ONLY dependency: installed manually in the dev venv
(`uv run pip install lancedb`), NOT in uv.lock — `uv sync` removes it. The
benchmark is excluded from the default pytest suite (benchmarks/ is not in
testpaths) and from ty's checked sources via a [tool.ty.overrides] entry.

Run (inside `devenv shell`, with the canonical LD_LIBRARY_PATH prefix from
benchmarks/README.md):
  uv run python benchmarks/bench_backends.py --n 100000 --dim 256
  uv run python benchmarks/bench_backends.py --n 1000000 --dim 256
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from embeddy.index.base import SearchFilters
from embeddy.index.sqlite import SqliteStore
from embeddy.protocol.types import SourceMetadata, StoredChunk

# --------------------------------------------------------------------------- #
# data generation
# --------------------------------------------------------------------------- #


def _unit(i: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def _generate(n: int, dim: int, batch: int = 10_000):
    """Yield (vectors, metas) batches: chunk_type 10% heading, 2 dirs,
    2 content types. Deterministic by index; vectors are random unit vectors
    (seeded) so neither backend has an index-shaped advantage."""
    rng = np.random.default_rng(seed=42)
    for start in range(0, n, batch):
        size = min(batch, n - start)
        raw = rng.normal(size=(size, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        vectors = (raw / norms).astype(np.float32)
        metas = []
        for j in range(size):
            i = start + j
            metas.append(
                {
                    "chunk_type": "heading" if i % 10 == 0 else "paragraph",
                    "source_path": f"docs/{'a' if i % 2 == 0 else 'b'}/f{i // 20}.md",
                    "content_type": "markdown" if i % 2 == 0 else "text",
                    "content": f"chunk {i} alpha beta gamma",
                }
            )
        yield vectors, metas


# --------------------------------------------------------------------------- #
# sqlite-vec (embeddy SqliteStore)
# --------------------------------------------------------------------------- #


async def bench_sqlite(n: int, dim: int, tmp: Path, top_k: int) -> dict[str, Any]:
    path = tmp / "sqlite.db"
    store = await SqliteStore.open(path)
    await store.create_collection("bench", dim)
    await store.upsert_source(
        "bench",
        SourceMetadata(
            id="src",
            collection_id="bench",
            path="docs/a/x.md",
            content_hash="h",
            size_bytes=1,
            content_type="markdown",
        ),
    )

    t0 = time.perf_counter()
    total = 0
    for vectors, metas in _generate(n, dim):
        chunks = [
            StoredChunk(
                id=f"c{total + j}",
                collection_id="bench",
                source_id="src",
                content=m["content"],
                chunk_type=m["chunk_type"],
                start_line=1,
                end_line=1,
                parent="intro" if m["chunk_type"] == "heading" else None,
                token_count=5,
            )
            for j, m in enumerate(metas)
        ]
        await store.add("bench", chunks, vectors)
        total += len(chunks)
    await store.close()
    ingest = time.perf_counter() - t0
    size = path.stat().st_size

    # warm-up then time queries
    store = await SqliteStore.open(path)
    q = _unit(0, dim)
    q[dim - 1] = 0.5  # perturb so the top-1 isn't trivially deterministic
    q = q / np.linalg.norm(q)

    queries = [_unit(i * 7 + 3, dim) for i in range(60)]

    async def timed(fn):
        for qv in queries[:5]:  # warm
            await fn(qv)
        samples = []
        for qv in queries:
            t = time.perf_counter()
            await fn(qv)
            samples.append(time.perf_counter() - t)
        return statistics.median(samples), statistics.quantiles(samples, n=20)[18]

    unf = await timed(lambda qv: store.search_vector("bench", qv, SearchFilters(), top_k))
    filt = await timed(
        lambda qv: store.search_vector(
            "bench",
            qv,
            SearchFilters(chunk_types=("paragraph",), content_types=("markdown",)),
            top_k,
        )
    )
    pref = await timed(
        lambda qv: store.search_vector(
            "bench", qv, SearchFilters(source_path_prefix="docs/a/"), top_k
        )
    )
    await store.close()

    return {
        "ingest_s": round(ingest, 2),
        "size_bytes": size,
        "unfiltered_ms": unf,
        "filtered_ms": filt,
        "prefix_ms": pref,
    }


# --------------------------------------------------------------------------- #
# LanceDB
# --------------------------------------------------------------------------- #


def bench_lancedb(n: int, dim: int, tmp: Path, top_k: int) -> dict[str, Any]:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    class Item(LanceModel):
        id: str
        vector: Vector(dim)
        chunk_type: str
        source_path: str
        content_type: str
        content: str

    path = str(tmp / "lancedb")
    db = lancedb.connect(path)
    table = db.create_table("bench", schema=Item)
    t0 = time.perf_counter()
    total = 0
    for vectors, metas in _generate(n, dim):
        rows = []
        for j, (vec, m) in enumerate(zip(vectors, metas, strict=True)):
            rows.append(
                Item(
                    id=f"c{total + j}",
                    vector=vec,
                    chunk_type=m["chunk_type"],
                    source_path=m["source_path"],
                    content_type=m["content_type"],
                    content=m["content"],
                )
            )
        table.add(rows)
        total += len(rows)
    ingest = time.perf_counter() - t0

    q = _unit(0, dim)
    q[dim - 1] = 0.5
    q = q / np.linalg.norm(q)
    queries = [_unit(i * 7 + 3, dim) for i in range(60)]

    def timed(fn):
        for _ in range(5):
            fn()
        samples = []
        for _ in queries:
            t = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t)
        return statistics.median(samples), statistics.quantiles(samples, n=20)[18]

    unf = timed(lambda: table.search(q).limit(top_k).to_list())
    filt = timed(
        lambda: table.search(q)
        .where("chunk_type = 'paragraph' AND content_type = 'markdown'")
        .limit(top_k)
        .to_list()
    )
    pref = timed(
        lambda: table.search(q).where("source_path LIKE 'docs/a/%'").limit(top_k).to_list()
    )

    # FTS + hybrid (LanceDB's "one package" claim)
    fts_build = time.perf_counter()
    try:
        table.create_index("content", config=_fts_config())
        fts_build_s = time.perf_counter() - fts_build
        hybrid_s = timed(
            lambda: table.search(query_type="hybrid").vector(q).text("alpha").limit(top_k).to_list()
        )
    except Exception as exc:
        fts_build_s = f"error: {exc}"
        hybrid_s = None

    size = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            size += f.stat().st_size

    return {
        "ingest_s": round(ingest, 2),
        "size_bytes": size,
        "unfiltered_ms": unf,
        "filtered_ms": filt,
        "prefix_ms": pref,
        "fts_index_s": fts_build_s,
        "hybrid_ms": hybrid_s,
    }


def _fts_config():
    from lancedb.index import FTS

    return FTS(stem=True, remove_stop_words=True)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


def _fmt(ms) -> str:
    if isinstance(ms, tuple):
        return f"{ms[0] * 1000:.2f} / p95 {ms[1] * 1000:.2f}"
    return str(ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="sqlite-vec vs LanceDB benchmark spike")
    parser.add_argument("--n", type=int, default=100_000, help="number of vectors")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--topk", type=int, default=50, help="search top_k")
    parser.add_argument("--json", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    import asyncio

    tmp = Path(tempfile.mkdtemp(prefix="bench_backends_"))
    print(f"== backend benchmark: n={args.n} dim={args.dim} top_k={args.topk} ==\n")
    try:
        print("sqlite-vec (SqliteStore)...")
        sqlite_res = asyncio.run(bench_sqlite(args.n, args.dim, tmp, args.topk))
        print("LanceDB...")
        lancedb_res = bench_lancedb(args.n, args.dim, tmp, args.topk)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'metric':<22}{'sqlite-vec':>22}{'LanceDB':>22}")
    print("-" * 66)
    for key, label in [
        ("ingest_s", "ingest (s)"),
        ("size_bytes", "size (bytes)"),
        ("unfiltered_ms", "search top-k ms (p50/p95)"),
        ("filtered_ms", "filtered ms (p50/p95)"),
        ("prefix_ms", "prefix-filtered ms (p50/p95)"),
    ]:
        sv = sqlite_res.get(key, "")
        lb = lancedb_res.get(key, "")
        if isinstance(sv, tuple):
            sv = _fmt(sv)
        if isinstance(lb, tuple):
            lb = _fmt(lb)
        print(f"{label:<22}{str(sv):>22}{str(lb):>22}")
    if "fts_index_s" in lancedb_res:
        print(f"{'FTS index build (s)':<22}{'-':>22}{lancedb_res['fts_index_s']:>22}")
    if "hybrid_ms" in lancedb_res:
        print(f"{'hybrid ms (p50/p95)':<22}{'-':>22}{_fmt(lancedb_res['hybrid_ms']):>22}")

    result = {"n": args.n, "dim": args.dim, "sqlite_vec": sqlite_res, "lancedb": lancedb_res}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
