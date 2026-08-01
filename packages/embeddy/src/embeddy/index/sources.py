"""Source operations — the transactional machinery behind SqliteStore's
Searchable source ops (CONCEPT §3.3, plan §6 work item 1).

The public methods live on SqliteStore (sqlite.py) and delegate here; this
module owns the row-level DML that add() and reindex_source() share, so the
chunks/vec0/FTS write paths can never drift apart.

Invariants encoded here (spike-verified, spikes/store_schema.py 7/7):

  * REINDEX = ATOMIC SWAP in ONE implicit transaction (fixes H7). The caller
    expresses the swap with commit()/rollback() only — aiosqlite runs the
    connection on a worker thread and `db.isolation_level` must NEVER be set
    from the caller thread (sqlite3 threading check, verified in the spike).
    On failure the OLD chunks, vectors and source metadata must remain
    intact and queryable.
  * vec0 tables with a TEXT PRIMARY KEY take no rowid: INSERT specifies the
    columns explicitly (`INSERT INTO v(id, embedding, ...)`).
  * vec0 auxiliary columns REJECT NULL (probed 2026-08-01) — optional chunk
    fields (parent, granularity) and optional source fields (content_type)
    are coerced to "" in the vector row; the chunks/sources tables keep
    real NULLs.
  * FTS5 external-content cleanup order: delete FTS rows BEFORE the content
    rows (the FTS delete reads the content row to un-index it; deleting
    content first leaves stale, queryable FTS entries — probed).
"""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from embeddy.protocol.types import StoredChunk, Vector

# vec0 aux-column insert: chunk/source filterable fields denormalized onto
# the vector row so SearchFilters compile to in-scan constraints.
_VEC_INSERT = (
    'INSERT INTO "v_{collection}"'
    "(id, embedding, chunk_type, parent, granularity, content_type, source_path) "
    "VALUES (?,?,?,?,?,?,?)"
)


async def insert_chunks_with_vectors(
    db: aiosqlite.Connection,
    *,
    collection: str,
    chunks: list[StoredChunk],
    vectors: list[Vector],
    source_lookup: dict[str, tuple[str | None, str]],
) -> None:
    """Write chunks + vec0 rows + FTS5 rows. Validation (dimension, unit
    norm, chunk/vector pairing, source existence) is the CALLER's job; the
    caller owns the transaction (commit/rollback)."""
    if not chunks:
        return

    await db.executemany(
        "INSERT INTO chunks(id, collection_id, source_id, content, "
        "chunk_type, start_line, end_line, parent, granularity, "
        "token_count) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                c.id,
                collection,
                c.source_id,
                c.content,
                c.chunk_type,
                c.start_line,
                c.end_line,
                c.parent,
                c.granularity,
                c.token_count,
            )
            for c in chunks
        ],
    )

    rowids = await _chunk_rowids(db, [c.id for c in chunks])

    await db.executemany(
        _VEC_INSERT.format(collection=collection),
        [
            (
                c.id,
                serialize_vector(v),
                c.chunk_type,
                c.parent or "",
                c.granularity or "",
                _source_field(source_lookup, c.source_id, 0),
                _source_field(source_lookup, c.source_id, 1),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )

    await db.executemany(
        "INSERT INTO chunks_fts(rowid, content, chunk_type, parent) VALUES (?,?,?,?)",
        [(rowids[c.id], c.content, c.chunk_type, c.parent) for c in chunks],
    )


async def delete_chunk_index_rows(
    db: aiosqlite.Connection,
    *,
    collection: str,
    chunk_ids: Sequence[str],
) -> None:
    """Remove vec0 + FTS5 rows for `chunk_ids`. The chunks content rows are
    the CALLER's job and must be deleted AFTER this (the FTS delete reads
    chunks rows to un-index them)."""
    if not chunk_ids:
        return
    placeholders = ",".join("?" for _ in chunk_ids)
    ids = list(chunk_ids)
    await db.execute(
        "DELETE FROM chunks_fts WHERE rowid IN "
        f"(SELECT rowid FROM chunks WHERE id IN ({placeholders}))",
        ids,
    )
    await db.execute(f'DELETE FROM "v_{collection}" WHERE id IN ({placeholders})', ids)


async def _chunk_rowids(db: aiosqlite.Connection, ids: Sequence[str]) -> dict[str, int]:
    placeholders = ",".join("?" for _ in ids)
    rows = await db.execute_fetchall(
        f"SELECT id, rowid FROM chunks WHERE id IN ({placeholders})", list(ids)
    )
    return {row["id"]: int(row["rowid"]) for row in rows}


def _source_field(lookup: dict[str, tuple[str | None, str]], source_id: str, index: int) -> str:
    """content_type (index 0) / path (index 1) from the caller's source
    lookup; missing entries degrade to "" (defensive — validation upstream
    guarantees presence)."""
    value = lookup.get(source_id, (None, ""))[index]
    return value or ""


def serialize_vector(vec: Vector) -> str:
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32)
    return "[" + ",".join(f"{float(x):.9g}" for x in arr) + "]"


__all__ = ["delete_chunk_index_rows", "insert_chunks_with_vectors", "serialize_vector"]
