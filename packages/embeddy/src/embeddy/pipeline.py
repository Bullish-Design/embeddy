"""Pipeline & orchestration — bounded, overlapping, source-aware ingest.

Phase 5 core (IMPLEMENTATION_PLAN §7, CONCEPT §5.6). The pipeline is a PURE
CONSUMER of the FROZEN M4 protocols — `Searchable` (index/base.py),
`EmbeddingProvider` (protocol/embedding.py) and the chonkai public API. It
never reshapes them; a change to any frozen signature requires a
docs/decisions/ record (plan §12).

Design, from plan §7 / CONCEPT §5.6 / §3.6:

  * Bounded worker pool: `asyncio.Semaphore` + one task per file;
    `concurrency` bounds the number of files in any phase simultaneously.
    read/chunk/embed/write phases OVERLAP ACROSS files (each file runs its
    phases sequentially inside the semaphore); memory stays flat — chunk
    contents and vectors are per-file and released after each store write,
    never accumulated corpus-wide.
  * Document-role instruction: the pipeline is the CALLER that resolves
    roles (CONCEPT §5.1). It resolves the DOCUMENT role via
    `registry.resolve_instruction(provider.model_name, "document")` (when
    not injected) and passes the RESOLVED string to `encode` — the H2 bug
    class (document content embedded with the query instruction) is
    impossible by construction.
  * Chunk budget: `budget.chunk_budget(provider.context_length)` — never a
    hardcoded number (plan §13 cross-package contract).
  * Source ops: `ingest_text`, `ingest_file`, `ingest_directory`, `reindex`
    (ATOMIC — calls `SqliteStore.reindex_source`, never delete-then-
    reingest), `delete_source`, `sync` (incremental new/modified/deleted
    diff against `store.list_sources`).
  * SourceId generation: hash of (collection, path) — the protocol
    contract (protocol/types.py SourceId docstring); chunk ids are
    f"{source_id}:{seq}".
  * Dedup policy (documented): dedup is SOURCE-level, keyed on
    (collection, path) + content_hash. Identical content at a DIFFERENT
    path is a NEW source (the store's UNIQUE(collection_id, path) enforces
    this); unchanged content at the SAME path is SKIPPED by default, or
    reindexed when `reindex_unchanged=True`.
  * Errors are COLLECTED into typed `IngestStats`, never raised out of the
    pool (fixes H6) — the chunker failure path is recorded, the pool
    drains, and the caller inspects `stats.errors`.

Collection lifecycle: `Searchable` has no `create_collection` (that is a
SqliteStore extension); the pipeline therefore requires the collection to
ALREADY EXIST (the Phase-6 server owns collection lifecycle). A missing
collection raises `PipelineError` BEFORE any work starts.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import cast

import numpy as np

from chonkai import (
    BaseChunker,
    Chunk,
    ChunkBudget,
    Ingestor,
    IngestResult,
    ValidatedChunker,
    get_chunker,
)
from embeddy.budget import chunk_budget
from embeddy.config import DEFAULT_PIPELINE_CONCURRENCY
from embeddy.errors import EmbeddyError, ProviderError
from embeddy.index.base import Searchable
from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.protocol.types import (
    EmbedInput,
    SourceId,
    SourceMetadata,
    StoredChunk,
    Vector,
    assert_unit_vector,
)
from embeddy.registry import resolve_instruction

__all__ = [
    "FileEvent",
    "FileStatus",
    "IngestPipeline",
    "IngestStats",
    "PipelineError",
    "SourceError",
    "SourcePhase",
    "generate_source_id",
]


# --------------------------------------------------------------------------- #
# typed records
# --------------------------------------------------------------------------- #


class SourcePhase(str, Enum):
    """The pipeline phase a per-source error occurred in. Field semantics:
    READ = file I/O / ingest failed; CHUNK = chunking failed (incl. the
    ValidatedChunker invariants); EMBED = the provider failed or returned
    malformed vectors; STORE = the Searchable write failed."""

    READ = "read"
    CHUNK = "chunk"
    EMBED = "embed"
    STORE = "store"


class FileStatus(str, Enum):
    """The terminal status of one file in a run: INDEXED (stored or
    atomically reindexed), SKIPPED (unchanged content, dedup policy),
    DELETED (sync removal), FAILED (a phase error was collected)."""

    INDEXED = "indexed"
    SKIPPED = "skipped"
    DELETED = "deleted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SourceError:
    """One collected per-source failure (fixes H6: collected, never raised).

    `error_type` is the exception's class name (e.g. "ChunkValidationError",
    "ProviderError", "sqlite3.IntegrityError") so callers can decide retry
    vs ignore without string-matching messages.
    """

    path: str
    phase: SourcePhase
    message: str
    error_type: str = "Exception"

    def __str__(self) -> str:
        return f"{self.phase.value} error for {self.path!r} [{self.error_type}]: {self.message}"


@dataclass(frozen=True, slots=True)
class FileEvent:
    """One per-file progress event, delivered via the `on_file_indexed`
    callback. Delivered EXACTLY once per file, in SUBMISSION order
    (path-sorted for directory/sync operations; sync deletions precede
    ingest events). The callback must not raise; if it does, the error
    propagates and the run stops (the callback is the caller's code).

    `status` is one of FileStatus; `chunks` is the number of StoredChunks
    written (0 for skipped/deleted/failed); `error` is the collected error
    message for failed files.
    """

    path: str
    source_id: SourceId
    status: FileStatus
    chunks: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestStats:
    """Typed result of one ingest/reindex/sync run. FIELD SEMANTICS:

    files_attempted — sources submitted to the run (directory/sync: every
        file found under the root; ingest_text/ingest_file/reindex: 1).
        Includes skipped and failed files.
    files_indexed   — sources whose FULL pipeline (read/chunk/embed/write)
        succeeded: new sources stored, changed sources atomically
        reindexed, unchanged sources reindexed when reindex_unchanged=True.
    files_skipped   — existing sources with UNCHANGED content skipped by
        the dedup policy (reindex_unchanged=False, default).
    files_deleted   — sources removed by sync's delete pass (0 for every
        other operation).
    chunks_indexed  — total StoredChunks written (for a reindexed source
        this is the NEW chunk count). Always 0 for skipped/deleted/failed.
    errors          — per-source failures in SUBMISSION order, phase-tagged
        (SourceError). NEVER raised out of the pool (fixes H6); a run with
        errors returns normally and the caller inspects this tuple.
    """

    files_attempted: int
    files_indexed: int
    files_skipped: int
    files_deleted: int
    chunks_indexed: int
    errors: tuple[SourceError, ...]

    @property
    def files_failed(self) -> int:
        """Number of sources that failed (len(errors) — a convenience for
        callers; the tuple is the source of truth)."""
        return len(self.errors)


@dataclass(frozen=True, slots=True)
class _FileOutcome:
    """Internal per-file result before aggregation into IngestStats."""

    path: str
    status: FileStatus
    source_id: SourceId = ""
    chunks: int = 0
    error: SourceError | None = None


class PipelineError(EmbeddyError):
    """A whole-operation precondition failed (missing directory, unknown
    collection, reindex of a non-existent source). Per-source failures are
    COLLECTED in IngestStats — PipelineError is for operation-level errors
    only."""


def generate_source_id(collection: str, path: str) -> SourceId:
    """Stable SourceId — hash of (collection, path), per the protocol
    contract (embeddy/protocol/types.py SourceId docstring: "generated by
    the pipeline (hash of collection + path)"). Content-independent, so it
    is stable across re-ingests and identical across backends. The pipeline
    OWNS this generation; chunk ids follow f"{source_id}:{seq}"."""
    digest = hashlib.sha256(f"{collection}\0{path}".encode()).hexdigest()
    return f"src-{digest}"


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #


class IngestPipeline:
    """Bounded, overlapping, source-aware ingest (plan §7 / CONCEPT §5.6).

    The pipeline is a PURE CONSUMER of the frozen M4 protocols and owns:
      * SourceId generation (generate_source_id — hash of collection+path)
      * document-role instruction resolution (registry — H2 by construction)
      * the chunk budget (budget.chunk_budget(provider.context_length))
      * error collection (typed IngestStats) + progress events (FileEvent)

    The COLLECTION must already exist (create it with
    SqliteStore.create_collection; Searchable itself has no
    create_collection). A missing collection raises PipelineError before any
    work starts.
    """

    def __init__(
        self,
        *,
        store: Searchable,
        provider: EmbeddingProvider,
        chunker: BaseChunker | None = None,
        budget: ChunkBudget | None = None,
        token_counter: Callable[[str], int] | None = None,
        instruction: str | None = None,
        concurrency: int = DEFAULT_PIPELINE_CONCURRENCY,
        ingestor: Ingestor | None = None,
        on_file_indexed: Callable[[FileEvent], None] | None = None,
        reindex_unchanged: bool = False,
    ) -> None:
        """Construct the pipeline.

        `chunker` — a fixed BaseChunker wrapped in ValidatedChunker. When
        None, a chunker is selected PER SOURCE via chonkai `get_chunker`
        (auto strategy: content-type routing — semchunk for text,
        tree-sitter for code/markdown, docling for rich documents).
        `budget` — chonkai ChunkBudget; when None, derived from
        `provider.context_length` via `budget.chunk_budget` (never a
        hardcoded number).
        `token_counter` — deterministic counter for ValidatedChunker
        (defaults to chonkai's tiktoken counter with an offline chars/4
        fallback).
        `instruction` — an ALREADY-RESOLVED document-role instruction
        string. When None, the pipeline resolves it from the registry
        (`resolve_instruction(provider.model_name, "document")`) — the
        provider never sees a role (fixes H2).
        `concurrency` — the worker-pool bound (asyncio.Semaphore); must be
        >= 1. Default mirrors PipelineSettings.concurrency.
        `ingestor` — chonkai Ingestor (file reading + content-type
        detection); a new one is created when not provided.
        `on_file_indexed` — per-file progress callback (FileEvent, exactly
        once per file in submission order).
        `reindex_unchanged` — dedup policy: False (default) skips sources
        whose content is unchanged at the same path; True reindexes them
        anyway (atomic swap).
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self._store = store
        self._provider = provider
        self._budget = budget if budget is not None else chunk_budget(provider.context_length)
        self._token_counter = token_counter
        self._concurrency = concurrency
        self._ingestor = ingestor or Ingestor()
        self._on_file_indexed = on_file_indexed
        self._reindex_unchanged = reindex_unchanged
        self._instruction = (
            instruction
            if instruction is not None
            else resolve_instruction(provider.model_name, "document")
        )
        self._fixed_chunker = (
            ValidatedChunker(chunker, budget=self._budget, token_counter=token_counter)
            if chunker is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # single-source ops
    # ------------------------------------------------------------------ #

    async def ingest_text(
        self,
        text: str,
        *,
        collection: str,
        path: str = "<memory>",
        content_type: str | None = None,
        mtime: datetime | None = None,
    ) -> IngestStats:
        """Ingest raw text as one source (no file I/O). Identical content
        at the same path is skipped per the dedup policy."""
        await self._require_collection(collection)
        ingest = self._ingestor.ingest_text(text, path=path, content_type=content_type, mtime=mtime)
        existing = await self._store.get_source(collection, path)
        outcome = await self._process_source(collection, ingest, existing)
        self._emit_outcome(outcome)
        return self._stats_from_outcomes([outcome])

    async def ingest_file(self, path: str | Path, *, collection: str) -> IngestStats:
        """Ingest one file. Unchanged content at the same path is skipped
        (or reindexed with `reindex_unchanged=True`); identical content at
        a DIFFERENT path is a new source."""
        await self._require_collection(collection)
        canonical = str(path)
        existing = await self._store.get_source(collection, canonical)
        try:
            ingest = self._ingestor.ingest_file(path)
        except Exception as exc:
            outcome = self._failed(canonical, SourcePhase.READ, exc)
            self._emit_outcome(outcome)
            return self._stats_from_outcomes([outcome])
        outcome = await self._process_source(collection, ingest, existing)
        self._emit_outcome(outcome)
        return self._stats_from_outcomes([outcome])

    async def reindex(self, path: str | Path, *, collection: str) -> IngestStats:
        """ATOMIC re-index of one EXISTING source: re-read, re-chunk,
        re-embed, then `store.reindex_source` — the chunk set and metadata
        swap in ONE transaction, never delete-then-reingest (fixes H7).
        Raises PipelineError when no source is indexed at `path` (use
        ingest_file to ADD a new source)."""
        await self._require_collection(collection)
        canonical = str(path)
        existing = await self._store.get_source(collection, canonical)
        if existing is None:
            raise PipelineError(
                f"no indexed source at {canonical!r} in collection {collection!r}; "
                "use ingest_file to add a new source"
            )
        try:
            ingest = self._ingestor.ingest_file(path)
        except Exception as exc:
            outcome = self._failed(canonical, SourcePhase.READ, exc)
            self._emit_outcome(outcome)
            return self._stats_from_outcomes([outcome])
        outcome = await self._process_source(collection, ingest, existing)
        self._emit_outcome(outcome)
        return self._stats_from_outcomes([outcome])

    async def delete_source(self, collection: str, path: str) -> bool:
        """Delete the source at `path` (cascade: chunks, vectors, FTS rows).
        Returns True when a source existed and was deleted, False when there
        was nothing to delete. Emits no FileEvent (single explicit op)."""
        await self._require_collection(collection)
        existing = await self._store.get_source(collection, path)
        if existing is None:
            return False
        await self._store.delete_source(collection, existing.id)
        return True

    # ------------------------------------------------------------------ #
    # directory ops
    # ------------------------------------------------------------------ #

    async def ingest_directory(self, directory: str | Path, *, collection: str) -> IngestStats:
        """Ingest every file under `directory` (recursive, path-sorted)
        through the bounded worker pool. New files are stored, changed
        files are atomically reindexed, unchanged files are skipped (dedup
        policy). NO deletion — use sync() for the incremental diff. Store
        paths are relative to `directory` (posix separators)."""
        root = Path(directory)
        self._check_directory(root)
        await self._require_collection(collection)
        files = _walk_files(root)
        return await self._run_pool(collection, files, root)

    async def sync(self, directory: str | Path, *, collection: str) -> IngestStats:
        """Incremental sync — new / modified / deleted diff against
        `store.list_sources` (plan §7). Deletions (sources missing from the
        directory) are removed FIRST (cascade, path-sorted, DELETED events
        delivered), then every remaining file goes through the pool: new
        files stored, changed files atomically reindexed, unchanged files
        skipped. Store paths are relative to `directory` (posix)."""
        root = Path(directory)
        self._check_directory(root)
        await self._require_collection(collection)
        files = _walk_files(root)
        present = {p.relative_to(root).as_posix() for p in files}
        deleted = 0
        for source in await self._store.list_sources(collection):  # path-sorted
            if source.path in present:
                continue
            await self._store.delete_source(collection, source.id)
            deleted += 1
            self._emit(
                FileEvent(
                    path=source.path,
                    source_id=source.id,
                    status=FileStatus.DELETED,
                )
            )
        stats = await self._run_pool(collection, files, root)
        return IngestStats(
            files_attempted=stats.files_attempted,
            files_indexed=stats.files_indexed,
            files_skipped=stats.files_skipped,
            files_deleted=deleted,
            chunks_indexed=stats.chunks_indexed,
            errors=stats.errors,
        )

    # ------------------------------------------------------------------ #
    # the bounded worker pool
    # ------------------------------------------------------------------ #

    async def _run_pool(
        self,
        collection: str,
        files: list[Path],
        root: Path,
    ) -> IngestStats:
        """Run the bounded pool over `files`. Every file yields EXACTLY ONE
        outcome; outcomes are emitted (callback) in SUBMISSION order even
        when completions arrive out of order (completed-but-not-yet-emitted
        outcomes buffer as tiny records, never chunk content). Errors are
        collected into IngestStats — nothing propagates out of the pool."""
        if not files:
            return self._stats_from_outcomes([])
        semaphore = asyncio.Semaphore(self._concurrency)
        slots: list[_FileOutcome | None] = [None] * len(files)
        next_emit = 0

        async def worker(index: int, path: Path) -> None:
            nonlocal next_emit
            rel = path.relative_to(root).as_posix()
            async with semaphore:
                outcome = await self._process_file(collection, path, rel)
            slots[index] = outcome
            # emit in submission order; sync block (no awaits) so this is
            # atomic on the event loop
            while next_emit < len(slots):
                outcome = slots[next_emit]
                if outcome is None:
                    break
                self._emit_outcome(outcome)
                next_emit += 1

        await asyncio.gather(*(worker(i, p) for i, p in enumerate(files)))
        outcomes = [o for o in slots if o is not None]
        return self._stats_from_outcomes(outcomes)

    async def _process_file(self, collection: str, path: Path, rel: str) -> _FileOutcome:
        """One file's full pipeline: lookup existing source, read, then
        process. Read errors are collected (never raised). The ingest
        result is re-keyed to its canonical collection-relative path
        (`rel`) so the store, the SourceId and the events all agree —
        directory ops read via full filesystem paths but STORE canonical
        paths."""
        existing = await self._store.get_source(collection, rel)
        try:
            ingest = self._ingestor.ingest_file(path)
        except Exception as exc:
            return self._failed(rel, SourcePhase.READ, exc)
        return await self._process_source(collection, _rekey_ingest(ingest, rel), existing)

    async def _process_source(
        self,
        collection: str,
        ingest: IngestResult,
        existing: SourceMetadata | None,
    ) -> _FileOutcome:
        """read->chunk->embed->write for one source. EVERY phase failure is
        collected into the outcome (fixes H6); this method never raises."""
        path = ingest.source.path
        try:
            source_meta = self._to_source_meta(collection, ingest)
        except Exception as exc:
            return self._failed(path, SourcePhase.READ, exc)
        # dedup policy: unchanged content at the same path = skip (default)
        # or reindex (reindex_unchanged=True). Identical content at a
        # DIFFERENT path is a NEW source (store UNIQUE(collection_id, path)).
        if (
            existing is not None
            and existing.content_hash == source_meta.content_hash
            and not self._reindex_unchanged
        ):
            return _FileOutcome(path=path, status=FileStatus.SKIPPED, source_id=existing.id)
        try:
            chunks = self._chunk(ingest)
        except Exception as exc:
            return self._failed(path, SourcePhase.CHUNK, exc)
        try:
            vectors = await self._embed([c.content for c in chunks])
        except Exception as exc:
            return self._failed(path, SourcePhase.EMBED, exc)
        stored = self._to_stored(collection, ingest, chunks, vectors)
        try:
            await self._write_source(
                collection, source_meta, stored, vectors, exists=existing is not None
            )
        except Exception as exc:
            return self._failed(path, SourcePhase.STORE, exc)
        return _FileOutcome(
            path=path,
            status=FileStatus.INDEXED,
            source_id=source_meta.id,
            chunks=len(stored),
        )

    async def _write_source(
        self,
        collection: str,
        source: SourceMetadata,
        chunks: list[StoredChunk],
        vectors: list[Vector],
        *,
        exists: bool,
    ) -> None:
        """Write one source, never delete-then-reingest (plan §7):

        * EXISTING source -> store.reindex_source ONLY: chunk set + source
          metadata swap in ONE transaction — on failure the OLD chunks,
          vectors, FTS rows and metadata remain intact and queryable (H7,
          spike-proven in Phase 4). The pipeline never pre-updates metadata
          (a pre-update would break the old-metadata guarantee).
        * NEW source -> upsert_source (create the row) then add (chunks +
          vectors + FTS in one transaction); a failed add is best-effort
          cleaned up by deleting the just-created source row.
        """
        if exists:
            await self._store.reindex_source(collection, source, chunks, vectors)
            return
        await self._store.upsert_source(collection, source)
        try:
            await self._store.add(collection, chunks, vectors)
        except BaseException:
            try:
                await self._store.delete_source(collection, source.id)
            except Exception:
                pass  # keep the original error; the dangling row is harmless
            raise

    # ------------------------------------------------------------------ #
    # per-phase helpers
    # ------------------------------------------------------------------ #

    def _chunk(self, ingest: IngestResult) -> list[Chunk]:
        """Chunk one source. Uses the fixed chunker when provided, else a
        content-type-routed chunker per source (get_chunker auto). Always
        wrapped in ValidatedChunker (invariants + token budget). Chunker
        failures are the caller's job to collect (SourcePhase.CHUNK)."""
        if self._fixed_chunker is not None:
            return self._fixed_chunker.chunk(ingest)
        inner = get_chunker(ingest.source.content_type)
        validated = ValidatedChunker(inner, budget=self._budget, token_counter=self._token_counter)
        return validated.chunk(ingest)

    async def _embed(self, texts: list[str]) -> list[Vector]:
        """Embed chunk contents with the RESOLVED document instruction and
        enforce the protocol boundary (protocol/embedding.py: "the pipeline
        wrapper, not in each adapter"): count, dimension and unit-norm
        guards live HERE. Violations are ProviderError (collected as an
        EMBED-phase error)."""
        vectors = await self._provider.encode(
            cast(list[EmbedInput], texts), instruction=self._instruction
        )
        if len(vectors) != len(texts):
            raise ProviderError(f"provider returned {len(vectors)} vectors for {len(texts)} inputs")
        dim = self._provider.dimension
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != dim:
                raise ProviderError(
                    f"provider returned vector shape {arr.shape}; expected ({dim},)"
                )
            assert_unit_vector(arr)
        return list(vectors)

    def _to_source_meta(self, collection: str, ingest: IngestResult) -> SourceMetadata:
        """chonkai SourceMetadata -> embeddy SourceMetadata (the boundary
        conversion — the two records are deliberately separate types)."""
        src = ingest.source
        return SourceMetadata(
            id=generate_source_id(collection, src.path),
            collection_id=collection,
            path=src.path,
            content_hash=src.content_hash or "",
            size_bytes=src.size_bytes,
            mtime=src.mtime,
            content_type=src.content_type,
        )

    def _to_stored(
        self,
        collection: str,
        ingest: IngestResult,
        chunks: list[Chunk],
        vectors: list[Vector],
    ) -> list[StoredChunk]:
        """chonkai Chunk -> embeddy StoredChunk; ids follow the
        f"{source_id}:{seq}" convention (protocol contract)."""
        source_id = generate_source_id(collection, ingest.source.path)
        return [
            StoredChunk(
                id=f"{source_id}:{seq}",
                collection_id=collection,
                source_id=source_id,
                content=c.content,
                chunk_type=c.chunk_type,
                start_line=c.start_line,
                end_line=c.end_line,
                parent=c.parent,
                granularity=c.granularity,
                token_count=c.token_count,
            )
            for seq, c in enumerate(chunks)
        ]

    # ------------------------------------------------------------------ #
    # aggregation + events
    # ------------------------------------------------------------------ #

    def _failed(self, path: str, phase: SourcePhase, exc: Exception) -> _FileOutcome:
        return _FileOutcome(
            path=path,
            status=FileStatus.FAILED,
            error=SourceError(
                path=path, phase=phase, message=str(exc), error_type=type(exc).__name__
            ),
        )

    def _event(self, outcome: _FileOutcome) -> FileEvent:
        return FileEvent(
            path=outcome.path,
            source_id=outcome.source_id,
            status=outcome.status,
            chunks=outcome.chunks,
            error=outcome.error.message if outcome.error is not None else None,
        )

    def _emit_outcome(self, outcome: _FileOutcome) -> None:
        self._emit(self._event(outcome))

    def _emit(self, event: FileEvent) -> None:
        if self._on_file_indexed is not None:
            self._on_file_indexed(event)

    @staticmethod
    def _stats_from_outcomes(outcomes: list[_FileOutcome]) -> IngestStats:
        return IngestStats(
            files_attempted=len(outcomes),
            files_indexed=sum(1 for o in outcomes if o.status is FileStatus.INDEXED),
            files_skipped=sum(1 for o in outcomes if o.status is FileStatus.SKIPPED),
            files_deleted=0,
            chunks_indexed=sum(o.chunks for o in outcomes if o.status is FileStatus.INDEXED),
            errors=tuple(o.error for o in outcomes if o.error is not None),
        )

    # ------------------------------------------------------------------ #
    # preconditions
    # ------------------------------------------------------------------ #

    async def _require_collection(self, collection: str) -> None:
        """The collection must exist BEFORE any work (Searchable has no
        create_collection; SqliteStore.create_collection is the caller's
        job). stats() is the portable existence probe."""
        try:
            await self._store.stats(collection)
        except Exception as exc:
            raise PipelineError(
                f"collection {collection!r} is not ready — create it first "
                f"(e.g. SqliteStore.create_collection): {exc}"
            ) from exc

    @staticmethod
    def _check_directory(root: Path) -> None:
        if not root.exists():
            raise PipelineError(f"no such directory: {root}")
        if not root.is_dir():
            raise PipelineError(f"not a directory: {root}")


def _walk_files(root: Path) -> list[Path]:
    """Recursive, path-sorted list of regular files under `root` (the
    stable submission order the pool and its events follow)."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def _rekey_ingest(ingest: IngestResult, path: str) -> IngestResult:
    """Re-key an ingest result to its canonical collection path.

    Directory/sync ops read via full filesystem paths but store canonical
    collection-relative paths; the source record, SourceId and events must
    all use the canonical path. The content hash is path-independent
    (sha256 of bytes), so only the path changes."""
    if ingest.source.path == path:
        return ingest
    from dataclasses import replace

    return replace(ingest, source=replace(ingest.source, path=path))
