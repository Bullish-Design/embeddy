"""STORE SCHEMA SPIKE — prove the sqlite schema end-to-end with aiosqlite.

Run: LSTD=$(find /nix/store -name 'libstdc++.so.6' | head -1); LZ=$(find /nix/store -name 'libz.so.1' | head -1) \
     LD_LIBRARY_PATH="$(dirname "$LSTD"):$(dirname "$LZ")" /tmp/spike-venv/bin/python \
     .scratch/projects/001-greenfield-rewrite/spikes/store_schema.py

Proves, on a real in-memory DB (sqlite-vec extension loaded on the
connection thread):
  1. schema.sql applies; PRAGMA user_version stamped.
  2. 3 vectors inserted; cosine KNN distances match hand-computed cosine
     similarity (score = 1.0 - distance under cosine).
  3. FTS5 external-content sync works (porter+unicode61 tokenizer).
  4. atomic reindex_source swap: a mid-swap failure leaves the OLD chunks
     fully intact and queryable (fixes H7).
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import sys
from pathlib import Path

import aiosqlite
import numpy as np
import sqlite_vec

SCHEMA = Path(__file__).parent / "schema.sql"
COLLECTION = "demo"
DIM = 4

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- hand-computed cosine ------------------------------------------------------

VECS = {
    "c1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "c2": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "c3": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
}
QUERY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def hand_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def init(db: aiosqlite.Connection) -> None:
    await db.execute("PRAGMA foreign_keys = ON")
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)
    schema_sql = SCHEMA.read_text().format(collection=COLLECTION, dim=DIM)
    await db.executescript(schema_sql)
    await db.commit()


async def main() -> int:
    print("== store schema spike (aiosqlite + sqlite-vec " + sqlite_vec.__version__ + ") ==")
    db = await aiosqlite.connect(":memory:")
    db.row_factory = sqlite3.Row
    await init(db)
    await db.execute("PRAGMA user_version = 1")
    await db.commit()

    # 1. user_version stamp
    row = (await db.execute_fetchall("PRAGMA user_version"))[0][0]
    check("schema applied, user_version stamped", row == 1, f"user_version={row}")
    vec_version = (await db.execute_fetchall("SELECT vec_version()"))[0][0]
    check("sqlite-vec loaded", vec_version == "v0.1.9", vec_version)

    # 2. insert a source + 3 chunks/vectors; cosine KNN correctness
    await db.execute(
        "INSERT INTO collections(id, vector_dimension) VALUES (?, ?)",
        (COLLECTION, DIM),
    )
    await db.execute(
        "INSERT INTO sources(id, collection_id, path, content_hash, size_bytes) "
        "VALUES ('src1', ?, 'docs/a.md', 'hash1', 100)",
        (COLLECTION,),
    )
    chunks = [
        ("c1", COLLECTION, "src1", "alpha beta", "paragraph", 1, 1, None, None, 2),
        ("c2", COLLECTION, "src1", "alpha gamma", "paragraph", 2, 2, None, None, 2),
        ("c3", COLLECTION, "src1", "gamma delta", "heading", 3, 3, "intro", None, 2),
    ]
    await db.executemany(
        "INSERT INTO chunks(id, collection_id, source_id, content, chunk_type, "
        "start_line, end_line, parent, granularity, token_count) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        chunks,
    )
    await db.executemany(
        f'INSERT INTO "v_{COLLECTION}"(id, embedding) VALUES (?, ?)',
        [(i, "[" + ",".join(f"{x:.6f}" for x in v) + "]") for i, v in VECS.items()],
    )
    # keep FTS in sync (the backend owns this; we prove the mechanism)
    for cid, _, _, content, ctype, _, _, parent, _, _ in chunks:
        rowid = (await db.execute_fetchall(
            "SELECT rowid FROM chunks WHERE id = ?", (cid,)))[0][0]
        await db.execute(
            f"INSERT INTO chunks_fts(rowid, content, chunk_type, parent) "
            f"VALUES (?, ?, ?, ?)", (rowid, content, ctype, parent),
        )
    await db.commit()

    q = "[" + ",".join(f"{x:.6f}" for x in QUERY) + "]"
    rows = await db.execute_fetchall(
        f'SELECT id, distance FROM "v_{COLLECTION}" WHERE embedding MATCH ? AND k = 3',
        (q,),
    )
    got = {r["id"]: r["distance"] for r in rows}
    expected = {
        i: 1.0 - hand_cosine(QUERY, VECS[i]) for i in VECS
    }
    order_ok = list(got) == sorted(got, key=lambda i: expected[i])
    scores_ok = all(
        math.isclose(got[i], expected[i], abs_tol=1e-5) for i in VECS
    )
    check("cosine KNN: order matches hand-computed",
          order_ok, f"order={list(got)}")
    check("cosine KNN: distances match hand-computed cosine",
          scores_ok,
          f"got={ {i: round(got[i], 6) for i in got} } "
          f"want={ {i: round(expected[i], 6) for i in expected} }")
    check("score semantics: 1.0 - distance == cosine similarity",
          math.isclose(1.0 - got["c1"], 1.0, abs_tol=1e-6))

    # 3. FTS5 (porter+unicode61) works
    fts = await db.execute_fetchall(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'alpha' ORDER BY rank"
    )
    check("FTS5 BM25 finds 'alpha'", len(fts) == 2, f"hits={len(fts)}")
    fts_stem = await db.execute_fetchall(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'betas'"
    )  # porter stems 'betas' -> 'beta'
    check("FTS5 porter stemming ('betas' matches 'beta')", len(fts_stem) == 1,
          f"hits={len(fts_stem)}")

    # 4. atomic reindex_source: inject a mid-swap failure -> old chunks intact.
    # aiosqlite keeps its connection on a worker thread, so the
    # `db.isolation_level` property cannot be toggled from the caller thread
    # (sqlite3 threading check — verified). The default isolation_level=""
    # already opens an implicit transaction on the first DML statement, so
    # atomicity is expressed with rollback()/commit() alone.
    new_chunks = [
        ("c1", COLLECTION, "src1", "alpha beta v2", "paragraph", 1, 1, None, None, 2),
        ("c4", COLLECTION, "src1", "new stuff", "paragraph", 4, 4, None, None, 2),
    ]
    new_vecs = {
        "c1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "c4": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    class MidSwapError(RuntimeError):
        pass

    try:
        # swap-in: delete old chunks+vectors, write new ones, bump source meta
        await db.execute("DELETE FROM chunks WHERE source_id = 'src1'")
        await db.execute(f"DELETE FROM \"v_{COLLECTION}\" WHERE id IN ('c1','c2','c3')")
        await db.execute("UPDATE sources SET content_hash='hash2', size_bytes=200 "
                         "WHERE id = 'src1'")
        await db.executemany(
            "INSERT INTO chunks(id, collection_id, source_id, content, chunk_type, "
            "start_line, end_line, parent, granularity, token_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            new_chunks,
        )
        await db.executemany(
            f'INSERT INTO "v_{COLLECTION}"(id, embedding) VALUES (?, ?)',
            [(i, "[" + ",".join(f"{x:.6f}" for x in v) + "]") for i, v in new_vecs.items()],
        )
        raise MidSwapError("injected mid-swap failure")
    except MidSwapError:
        await db.rollback()

    old_count = (await db.execute_fetchall(
        "SELECT COUNT(*) FROM chunks WHERE source_id = 'src1'"))[0][0]
    check("reindex rollback: old chunks intact", old_count == 3, f"count={old_count}")
    rows_after = await db.execute_fetchall(
        f'SELECT id, distance FROM "v_{COLLECTION}" WHERE embedding MATCH ? AND k = 3',
        (q,),
    )
    ids_after = {r["id"] for r in rows_after}
    check("reindex rollback: old vectors queryable",
          ids_after == {"c1", "c2", "c3"}, f"ids={sorted(ids_after)}")
    src_meta = (await db.execute_fetchall(
        "SELECT content_hash FROM sources WHERE id = 'src1'"))[0][0]
    check("reindex rollback: source metadata reverted",
          src_meta == "hash1", src_meta)

    # 5. reindex SUCCESS path — atomic swap commits cleanly
    try:
        await db.execute("DELETE FROM chunks WHERE source_id = 'src1'")
        await db.execute(f"DELETE FROM \"v_{COLLECTION}\" WHERE id IN ('c1','c2','c3')")
        await db.executemany(
            "INSERT INTO chunks(id, collection_id, source_id, content, chunk_type, "
            "start_line, end_line, parent, granularity, token_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            new_chunks,
        )
        await db.executemany(
            f'INSERT INTO "v_{COLLECTION}"(id, embedding) VALUES (?, ?)',
            [(i, "[" + ",".join(f"{x:.6f}" for x in v) + "]") for i, v in new_vecs.items()],
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    final_count = (await db.execute_fetchall(
        "SELECT COUNT(*) FROM chunks WHERE source_id = 'src1'"))[0][0]
    check("reindex success: new chunks swapped in", final_count == 2,
          f"count={final_count}")

    await db.close()

    print(f"\n{'PASS' if not FAILURES else 'FAIL'} — "
          f"{7 - len(FAILURES)}/7 checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
