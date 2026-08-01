"""Pipeline unit tests — plan §7 test plan, unit layer (no DB).

Covers, with a mocked store/provider/ingestor (FakeStore implements the
frozen Searchable surface in memory; RecordingProvider records
instruction/encode calls):
  * bounded concurrency honored (max in-flight <= concurrency) + phase
    overlap + flat-memory proxy (adds interleave with reads)
  * dedup: identical content at two paths = two sources; unchanged content
    at the same path = skip (or reindex per policy)
  * error collection: chunker/embed/store/read failures are RECORDED in
    IngestStats, never raised; the pool drains (fixes H6)
  * H2 regression: chunk content is embedded with the resolved DOCUMENT
    instruction, never the query role's
  * chunk budget derived from provider.context_length (never hardcoded)
  * progress events: every file exactly once, in submission order, even
    when completions arrive out of order
  * source ops: ingest_text / ingest_file / ingest_directory / reindex /
    delete_source / sync (incremental new/modified/deleted diff)
  * stable SourceId generation (hash of collection + path)

Real-store atomicity (H7 mid-swap rollback) is integration-tested in
tests/test_pipeline_phase5.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from chonkai import (
    BaseChunker,
    Chunk,
    ChunkBudget,
    ChunkValidationError,
    IngestResult,
    ParagraphChunker,
)
from embeddy import (
    IngestPipeline,
    IngestStats,
    PipelineError,
    generate_source_id,
    load_pipeline_settings,
)
from embeddy.errors import ProviderError
from embeddy.index.base import SearchFilters
from embeddy.pipeline import FileEvent, FileStatus, SourcePhase
from embeddy.protocol.types import (
    CollectionStats,
    EmbedInput,
    SourceMetadata,
    StoredChunk,
    Vector,
)
from embeddy.registry import resolve_instruction

# deterministic word-count counter: chunks never depend on tiktoken here
tok = lambda s: len(s.split())  # noqa: E731

DIM = 8


@dataclass
class _ReadWriteTracker:
    """Tracks read completion vs store-write progress for the overlap/
    flat-memory proxy (writes must interleave with reads)."""

    reads: int = 0
    adds_at_read: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# mocks
# --------------------------------------------------------------------------- #


class FakeStore:
    """In-memory Searchable for unit tests (no DB — plan §11 unit layer).

    `FakeStore(collection="acme")` pre-registers the collection (the analog
    of SqliteStore.create_collection); a store without it models a missing
    collection and the pipeline must raise PipelineError.
    """

    def __init__(self, collection: str | None = None) -> None:
        self._sources: dict[str, dict[str, SourceMetadata]] = {}
        self._chunks: dict[str, dict[str, dict[str, StoredChunk]]] = {}
        self._vectors: dict[str, Vector] = {}
        self.add_calls: list[int] = []  # chunk counts per add() call
        self.reindexed: list[str] = []  # source ids passed to reindex_source
        self.upserted: list[str] = []  # source ids passed to upsert_source
        self.deleted: list[str] = []  # source ids passed to delete_source
        if collection is not None:
            self._sources[collection] = {}

    # --- assertion helpers ------------------------------------------------
    def source_count(self, collection: str) -> int:
        return len(self._sources.get(collection, {}))

    def chunk_count(self, collection: str) -> int:
        return sum(len(cs) for cs in self._chunks.get(collection, {}).values())

    def chunk_ids(self, collection: str) -> list[str]:
        return sorted(cid for src in self._chunks.get(collection, {}).values() for cid in src)

    def chunk_contents(self, collection: str) -> list[tuple[str, int]]:
        """(content, token_count) pairs, sorted by chunk id."""
        return sorted(
            (c.content, c.token_count)
            for src in self._chunks.get(collection, {}).values()
            for c in src.values()
        )

    # --- Searchable surface ------------------------------------------------
    async def add(self, collection: str, chunks: list[StoredChunk], vectors: list[Vector]) -> None:
        self.add_calls.append(len(chunks))
        col = self._chunks.setdefault(collection, {})
        for c, v in zip(chunks, vectors, strict=True):
            col.setdefault(c.source_id, {})[c.id] = c
            self._vectors[c.id] = v

    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            for src in self._chunks.get(collection, {}).values():
                src.pop(cid, None)
            self._vectors.pop(cid, None)

    async def search_vector(
        self,
        collection: str,
        query_vector: Vector,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list:
        return []

    async def search_fts(
        self,
        collection: str,
        query: str,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
        raw: bool = False,
    ) -> list:
        return []

    async def stats(self, collection: str) -> CollectionStats:
        if collection not in self._sources:
            raise RuntimeError(f"unknown collection {collection!r}")
        return CollectionStats(
            collection_id=collection,
            chunk_count=self.chunk_count(collection),
            source_count=self.source_count(collection),
            vector_dimension=DIM,
        )

    async def upsert_source(self, collection: str, source: SourceMetadata) -> str:
        self._sources.setdefault(collection, {})[source.path] = source
        self.upserted.append(source.id)
        return source.id

    async def get_source(self, collection: str, path: str) -> SourceMetadata | None:
        return self._sources.get(collection, {}).get(path)

    async def reindex_source(
        self,
        collection: str,
        source: SourceMetadata,
        chunks: list[StoredChunk],
        vectors: list[Vector],
    ) -> None:
        self.reindexed.append(source.id)
        self._sources[collection][source.path] = source
        self._chunks[collection][source.id] = {c.id: c for c in chunks}
        for c, v in zip(chunks, vectors, strict=True):
            self._vectors[c.id] = v

    async def delete_source(self, collection: str, source_id: str) -> None:
        self.deleted.append(source_id)
        for path, s in list(self._sources.get(collection, {}).items()):
            if s.id == source_id:
                del self._sources[collection][path]
        self._chunks.get(collection, {}).pop(source_id, None)

    async def list_sources(self, collection: str) -> list[SourceMetadata]:
        return sorted(self._sources.get(collection, {}).values(), key=lambda s: s.path)


class RecordingProvider:
    """Fake EmbeddingProvider that records (inputs, instruction) per encode
    call and returns deterministic unit basis vectors. `model_name` is
    overrideable with a REGISTRY model id so the pipeline can resolve the
    document instruction (the H2 test). `encode_sleep` makes encode slow
    (concurrency tests); `fail_on` makes it raise for matching text."""

    dimension = DIM
    context_length = 4096
    model_name = "fake/provider"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        context_length: int | None = None,
        encode_sleep: float = 0.0,
        fail_on: set[str] | None = None,
    ) -> None:
        if model_name is not None:
            self.model_name = model_name
        if context_length is not None:
            self.context_length = context_length
        self.encode_sleep = encode_sleep
        self.fail_on = set(fail_on or ())
        self.calls: list[tuple[list[EmbedInput], str | None]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[Vector]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for item in inputs:
                if isinstance(item, str) and any(f in item for f in self.fail_on):
                    raise ProviderError(f"encode failed for {item[:20]!r}")
            if self.encode_sleep:
                await asyncio.sleep(self.encode_sleep)
            self.calls.append((list(inputs), instruction))
            return [self._unit(i) for i in range(len(inputs))]
        finally:
            self.in_flight -= 1

    @staticmethod
    def _unit(i: int) -> Vector:
        v = np.zeros(DIM, dtype=np.float32)
        v[i % DIM] = 1.0
        return v


class FakeIngestor:
    """Canned chonkai.Ingestor stand-in: no file I/O. Mirrors the public
    surface the pipeline calls (ingest_file / ingest_text). `files` maps
    path -> text (keyed by the exact string the pipeline passes);
    `fail` paths raise OSError on ingest."""

    def __init__(self, files: dict[str, str] | None = None, fail: set[str] | None = None) -> None:
        self.files = dict(files or {})
        self.fail = set(fail or ())

    def ingest_file(self, path: str | Path) -> IngestResult:
        key = str(path)
        if key in self.fail:
            raise OSError(f"cannot read {key}")
        text = self.files.get(key)
        if text is None:
            raise FileNotFoundError(f"no canned text for {key}")
        return IngestResult.from_text(text, path=key, content_type="text")

    def ingest_text(
        self,
        text: str,
        *,
        path: str = "<memory>",
        content_type: str | None = None,
        mtime=None,
    ) -> IngestResult:
        return IngestResult.from_text(text, path=path, content_type=content_type, mtime=mtime)


class BoomChunker(BaseChunker):
    """Raises ChunkValidationError when the text contains a marker."""

    def __init__(self, marker: str = "boom") -> None:
        self.marker = marker

    def chunk(self, ingest: IngestResult, budget: ChunkBudget | None = None) -> list[Chunk]:
        if self.marker in ingest.text:
            raise ChunkValidationError(f"boom: {ingest.text[:10]}")
        return [
            Chunk(
                content=ingest.text.strip(),
                start_line=1,
                end_line=1,
                chunk_type="paragraph",
            )
        ]


def make_pipeline(
    store: FakeStore,
    *,
    provider: RecordingProvider | None = None,
    instruction: str | None = "doc",  # sentinel; H2 tests pass None to resolve
    concurrency: int = 4,
    chunker: BaseChunker | None = None,
    **kw,
) -> IngestPipeline:
    provider = provider or RecordingProvider()
    chunker = chunker or ParagraphChunker(token_counter=tok)
    kw.setdefault("token_counter", tok)
    return IngestPipeline(
        store=store,
        provider=provider,
        chunker=chunker,
        instruction=instruction,
        concurrency=concurrency,
        **kw,
    )


def _write(tmp_path: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)


# --------------------------------------------------------------------------- #
# source-id generation (protocol contract)
# --------------------------------------------------------------------------- #


def test_generate_source_id_stable_hash_of_collection_and_path() -> None:
    a = generate_source_id("acme", "docs/a.md")
    assert a == generate_source_id("acme", "docs/a.md")  # stable across calls
    assert a.startswith("src-")
    assert a != generate_source_id("acme", "docs/b.md")  # path-sensitive
    assert a != generate_source_id("other", "docs/a.md")  # collection-sensitive
    assert len(a) == 4 + 64  # "src-" + sha256 hex


# --------------------------------------------------------------------------- #
# ingest_text basics + metadata round-trip
# --------------------------------------------------------------------------- #


async def test_ingest_text_indexes_one_source() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    stats = await pipeline.ingest_text(
        "first paragraph.\n\nsecond paragraph.", path="a.txt", collection="acme"
    )
    assert stats == IngestStats(
        files_attempted=1,
        files_indexed=1,
        files_skipped=0,
        files_deleted=0,
        chunks_indexed=2,
        errors=(),
    )
    assert stats.files_failed == 0
    assert store.source_count("acme") == 1
    assert store.chunk_count("acme") == 2
    sid = generate_source_id("acme", "a.txt")
    assert store.reindexed == []  # new source: upsert + add, never reindex
    assert store.upserted == [sid]
    assert store.chunk_ids("acme") == [f"{sid}:0", f"{sid}:1"]  # f"{source_id}:{seq}"


async def test_ingest_text_metadata_roundtrip() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    await pipeline.ingest_text(
        "some content", path="docs/guide.md", content_type="markdown", collection="acme"
    )
    src = await store.get_source("acme", "docs/guide.md")
    assert src is not None
    assert src.id == generate_source_id("acme", "docs/guide.md")
    assert src.content_type == "markdown"
    assert src.size_bytes == len("some content")
    assert src.content_hash  # sha256 hex from chonkai
    assert src.path == "docs/guide.md"


async def test_empty_text_indexes_source_with_zero_chunks() -> None:
    """Documented behavior: a source is indexed even when it produces zero
    chunks (the source row records existence; nothing is searchable)."""
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    stats = await pipeline.ingest_text("", path="empty.txt", collection="acme")
    assert stats.files_indexed == 1 and stats.chunks_indexed == 0
    assert store.source_count("acme") == 1


# --------------------------------------------------------------------------- #
# dedup policy (plan §7: source-level dedup)
# --------------------------------------------------------------------------- #


async def test_identical_content_at_two_paths_is_two_sources() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    s1 = await pipeline.ingest_text("same content", path="a.txt", collection="acme")
    s2 = await pipeline.ingest_text("same content", path="b.txt", collection="acme")
    assert s1.files_indexed == 1 and s2.files_indexed == 1
    assert store.source_count("acme") == 2  # two sources, same bytes
    ids = {s.id for s in await store.list_sources("acme")}
    assert len(ids) == 2
    # chunk ids are namespaced per source
    chunk_ids = store.chunk_ids("acme")
    assert len(chunk_ids) == 2
    assert any(cid.startswith(generate_source_id("acme", "a.txt")) for cid in chunk_ids)
    assert any(cid.startswith(generate_source_id("acme", "b.txt")) for cid in chunk_ids)


async def test_reingest_unchanged_content_is_skip_noop() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    s1 = await pipeline.ingest_text("hello world", path="a.txt", collection="acme")
    s2 = await pipeline.ingest_text("hello world", path="a.txt", collection="acme")
    assert s1.files_indexed == 1
    assert s2.files_skipped == 1 and s2.files_indexed == 0
    assert store.reindexed == []  # a skip never calls reindex_source
    assert len(store.add_calls) == 1  # only the first ingest wrote
    assert store.source_count("acme") == 1


async def test_modified_content_reindexes_via_atomic_swap() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    sid = generate_source_id("acme", "a.txt")
    await pipeline.ingest_text("version one", path="a.txt", collection="acme")
    stats = await pipeline.ingest_text(
        "version two totally different", path="a.txt", collection="acme"
    )
    assert stats.files_indexed == 1 and stats.files_skipped == 0
    assert store.reindexed == [sid]  # exactly ONE atomic swap, no upsert
    assert store.upserted == [sid]  # upsert only on the first (new) write
    assert store.chunk_contents("acme") == [("version two totally different", 4)]


async def test_reindex_unchanged_policy_true_reindexes() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, reindex_unchanged=True)
    sid = generate_source_id("acme", "a.txt")
    await pipeline.ingest_text("same", path="a.txt", collection="acme")
    stats = await pipeline.ingest_text("same", path="a.txt", collection="acme")
    assert stats.files_indexed == 1 and stats.files_skipped == 0
    assert store.reindexed == [sid]  # the unchanged source was re-swapped


# --------------------------------------------------------------------------- #
# error collection (fixes H6) — collected, never raised, pool drains
# --------------------------------------------------------------------------- #


async def test_chunker_error_recorded_not_raised() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, chunker=BoomChunker("boom"))
    stats = await pipeline.ingest_text("boom text", path="a.txt", collection="acme")
    assert stats.files_indexed == 0
    assert stats.files_failed == 1
    (err,) = stats.errors
    assert err.path == "a.txt"
    assert err.phase is SourcePhase.CHUNK
    assert err.error_type == "ChunkValidationError"
    assert "boom" in err.message
    assert store.source_count("acme") == 0


async def test_pool_drains_and_collects_mixed_errors(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "good1.txt": "fine text one",
            "boom.txt": "boom text",
            "good2.txt": "fine text two",
        },
    )
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, chunker=BoomChunker("boom"), concurrency=2)
    stats = await pipeline.ingest_directory(tmp_path, collection="acme")
    assert stats.files_attempted == 3
    assert stats.files_indexed == 2  # the pool drained despite the failure
    assert stats.files_failed == 1
    (err,) = stats.errors
    assert err.path == "boom.txt" and err.phase is SourcePhase.CHUNK


async def test_embed_error_recorded(tmp_path: Path) -> None:
    _write(tmp_path, {"bad.txt": "contains bad token", "ok.txt": "fine"})
    store = FakeStore(collection="acme")
    provider = RecordingProvider(fail_on={"bad"})
    pipeline = make_pipeline(store, provider=provider)
    stats = await pipeline.ingest_directory(tmp_path, collection="acme")
    assert stats.files_indexed == 1
    assert len(stats.errors) == 1
    err = stats.errors[0]
    assert err.path == "bad.txt" and err.phase is SourcePhase.EMBED
    assert err.error_type == "ProviderError"


async def test_embed_wrong_vector_shape_recorded() -> None:
    class WrongDimProvider(RecordingProvider):
        async def encode(self, inputs, instruction=None):
            del instruction
            return [np.ones(3, dtype=np.float32) for _ in inputs]

    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, provider=WrongDimProvider())
    stats = await pipeline.ingest_text("content", path="a.txt", collection="acme")
    (err,) = stats.errors
    assert err.phase is SourcePhase.EMBED
    assert "shape" in err.message


async def test_embed_wrong_count_recorded() -> None:
    class EmptyProvider(RecordingProvider):
        async def encode(self, inputs, instruction=None):
            del instruction
            return []

    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, provider=EmptyProvider())
    stats = await pipeline.ingest_text("content", path="a.txt", collection="acme")
    (err,) = stats.errors
    assert err.phase is SourcePhase.EMBED
    assert "0 vectors" in err.message


async def test_store_error_recorded_and_old_data_intact() -> None:
    class FailingStore(FakeStore):
        async def reindex_source(self, collection, source, chunks, vectors) -> None:
            if source.path == "a.txt":
                raise RuntimeError("simulated disk full")
            return await super().reindex_source(collection, source, chunks, vectors)

    store = FailingStore(collection="acme")
    pipeline = make_pipeline(store)
    await pipeline.ingest_text("version one", path="a.txt", collection="acme")
    stats = await pipeline.ingest_text("version two", path="a.txt", collection="acme")
    assert stats.files_indexed == 0
    (err,) = stats.errors
    assert err.phase is SourcePhase.STORE
    assert "disk full" in err.message
    # the failed write left the OLD source fully intact (atomicity at the
    # pipeline level: the old chunks are still queryable)
    assert store.source_count("acme") == 1
    assert store.chunk_contents("acme") == [("version one", 2)]


async def test_new_source_write_failure_cleans_up_source_row() -> None:
    class FailingAddStore(FakeStore):
        async def add(self, collection, chunks, vectors) -> None:
            self.add_calls.append(len(chunks))
            raise RuntimeError("add exploded")

    store = FailingAddStore(collection="acme")
    pipeline = make_pipeline(store)
    stats = await pipeline.ingest_text("content", path="a.txt", collection="acme")
    assert stats.files_indexed == 0
    (err,) = stats.errors
    assert err.phase is SourcePhase.STORE
    # best-effort cleanup deleted the just-created source row
    assert store.deleted == [generate_source_id("acme", "a.txt")]
    assert store.source_count("acme") == 0


async def test_read_error_recorded(tmp_path: Path) -> None:
    store = FakeStore(collection="acme")
    ok = str(tmp_path / "ok.txt")
    broken = str(tmp_path / "broken.txt")
    pipeline = make_pipeline(store, ingestor=FakeIngestor({ok: "y"}, fail={broken}))
    _write(tmp_path, {"broken.txt": "x", "ok.txt": "y"})
    stats = await pipeline.ingest_directory(tmp_path, collection="acme")
    assert stats.files_indexed == 1
    (err,) = stats.errors
    assert err.path == "broken.txt" and err.phase is SourcePhase.READ


# --------------------------------------------------------------------------- #
# H2 regression: document instruction, never the query role's (CONCEPT §5.1)
# --------------------------------------------------------------------------- #


async def test_h2_chunk_content_embedded_with_document_instruction() -> None:
    """The pipeline resolves the DOCUMENT role and hands the provider the
    resolved string — the H2 bug class (document content embedded with the
    query instruction) is impossible by construction."""
    provider = RecordingProvider(model_name="Qwen/Qwen3-VL-Embedding-2B")
    pipeline = make_pipeline(
        store=FakeStore(collection="acme"), provider=provider, instruction=None
    )
    await pipeline.ingest_text("some chunk content", path="a.txt", collection="acme")
    doc_instruction = resolve_instruction("Qwen/Qwen3-VL-Embedding-2B", "document")
    assert doc_instruction == "Represent the user's input."
    assert provider.calls, "the provider must have been called"
    # the pipeline resolved the DOCUMENT role from the registry and handed
    # the provider the resolved STRING — never a role. (Qwen3-VL registers
    # the same prompt for both roles; the query-role difference is proven
    # separately for Qwen3-0.6B in test_h2_qwen3_document_role_resolves_to_empty_string.)
    for _, instruction in provider.calls:
        assert instruction == doc_instruction


async def test_h2_qwen3_document_role_resolves_to_empty_string() -> None:
    """Qwen3 registers NO prompt for the document role (""); the pipeline
    must pass exactly that resolved string — never the query wrapper."""
    provider = RecordingProvider(model_name="Qwen/Qwen3-Embedding-0.6B")
    pipeline = make_pipeline(
        store=FakeStore(collection="acme"), provider=provider, instruction=None
    )
    await pipeline.ingest_text("chunk text", path="a.txt", collection="acme")
    assert resolve_instruction("Qwen/Qwen3-Embedding-0.6B", "document") == ""
    assert provider.calls and all(instruction == "" for _, instruction in provider.calls)


# --------------------------------------------------------------------------- #
# chunk budget (plan §13: provider.context_length drives it, never hardcoded)
# --------------------------------------------------------------------------- #


async def test_budget_derived_from_provider_context_length() -> None:
    class TinyContextProvider(RecordingProvider):
        context_length = 600  # -> chunk budget = 600 - 512 = 88 tokens

    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, provider=TinyContextProvider())
    # one ~120-word paragraph: must be POST-SPLIT into pieces <= 88 tokens
    text = " ".join(f"word{i}" for i in range(120))
    stats = await pipeline.ingest_text(text, path="big.txt", collection="acme")
    assert stats.files_indexed == 1
    assert stats.chunks_indexed == 2  # 88 + 32
    contents = store.chunk_contents("acme")
    assert [t for _, t in contents] == [88, 32]  # exactly the derived budget


# --------------------------------------------------------------------------- #
# concurrency + overlap + flat memory (CONCEPT §3.6, plan §7)
# --------------------------------------------------------------------------- #


async def test_concurrency_config_honored_and_phases_overlap(tmp_path: Path) -> None:
    n = 10
    texts = {f"f{i:02d}.txt": f"document {i} with token {i}" for i in range(n)}
    _write(tmp_path, texts)

    settings = load_pipeline_settings(concurrency=3)  # config -> pool bound
    store = FakeStore(collection="acme")
    tracker = _ReadWriteTracker()

    class TrackingIngestor(FakeIngestor):
        def ingest_file(self, path):
            result = super().ingest_file(path)
            tracker.reads += 1
            tracker.adds_at_read.append(len(store.add_calls))
            return result

    provider = RecordingProvider(encode_sleep=0.01)
    pipeline = make_pipeline(
        store,
        provider=provider,
        concurrency=settings.concurrency,
        ingestor=TrackingIngestor({str(tmp_path / k): v for k, v in texts.items()}),
    )
    stats = await pipeline.ingest_directory(tmp_path, collection="acme")
    assert stats.files_indexed == n

    # bound honored: never more than `concurrency` encodes in flight
    assert provider.max_in_flight <= settings.concurrency, (
        f"max in-flight {provider.max_in_flight} must be <= {settings.concurrency}"
    )
    # overlap happened: at least two files were in flight at once
    assert provider.max_in_flight >= 2

    # flat-memory proxy: writes interleave with reads (streaming), so the
    # pipeline never read the whole corpus before writing anything, and
    # never finished all writes before finishing all reads (no unbounded
    # accumulation of chunk content/vectors).
    adds = tracker.adds_at_read
    assert len(set(adds)) >= 2, "adds must interleave with reads"
    assert max(adds) < n, "writes must not all complete before the last read"
    # each store write carries exactly ONE file's chunks (no cross-file batching)
    assert store.add_calls and all(c == 1 for c in store.add_calls)


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        make_pipeline(FakeStore(collection="acme"), concurrency=0)


# --------------------------------------------------------------------------- #
# progress events (plan §7: per-file callback, exactly once, in order)
# --------------------------------------------------------------------------- #


async def test_progress_events_in_order_despite_out_of_order_completion(
    tmp_path: Path,
) -> None:
    texts = {"a.txt": "aaa slow doc", "b.txt": "bbb", "c.txt": "ccc", "d.txt": "ddd"}
    _write(tmp_path, texts)

    class SlowFirstProvider(RecordingProvider):
        async def encode(self, inputs, instruction=None):
            head = inputs[0] if isinstance(inputs[0], str) else ""
            await asyncio.sleep(0.05 if "aaa" in head else 0.001)
            self.calls.append((list(inputs), instruction))
            return [self._unit(i) for i in range(len(inputs))]

    events: list[FileEvent] = []
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(
        store,
        provider=SlowFirstProvider(),
        concurrency=4,
        on_file_indexed=events.append,
        ingestor=FakeIngestor({str(tmp_path / k): v for k, v in texts.items()}),
    )
    await pipeline.ingest_directory(tmp_path, collection="acme")
    # submission order = path-sorted; completion was out of order (a.txt last)
    assert [e.path for e in events] == ["a.txt", "b.txt", "c.txt", "d.txt"]
    assert len(events) == 4  # every file exactly once
    assert all(e.status is FileStatus.INDEXED for e in events)
    assert all(e.chunks == 1 for e in events)
    assert all(e.source_id for e in events)  # source ids populated


async def test_single_source_ops_emit_one_event() -> None:
    events: list[FileEvent] = []
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, on_file_indexed=events.append)
    stats = await pipeline.ingest_text("x", path="a.txt", collection="acme")
    assert stats.files_indexed == 1
    assert len(events) == 1
    assert events[0].path == "a.txt"
    assert events[0].status is FileStatus.INDEXED
    assert events[0].chunks == 1
    assert events[0].error is None


# --------------------------------------------------------------------------- #
# directory + sync ops
# --------------------------------------------------------------------------- #


async def test_ingest_directory_stats_and_skip(tmp_path: Path) -> None:
    _write(tmp_path, {"a.txt": "alpha", "b.txt": "beta", "c.txt": "gamma"})
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, concurrency=2)
    first = await pipeline.ingest_directory(tmp_path, collection="acme")
    assert first.files_attempted == 3 and first.files_indexed == 3
    assert first.chunks_indexed == 3
    second = await pipeline.ingest_directory(tmp_path, collection="acme")
    assert second.files_attempted == 3 and second.files_indexed == 0
    assert second.files_skipped == 3  # unchanged content -> skip
    assert store.source_count("acme") == 3


async def test_ingest_empty_directory(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    pipeline = make_pipeline(FakeStore(collection="acme"))
    stats = await pipeline.ingest_directory(sub, collection="acme")
    assert stats == IngestStats(0, 0, 0, 0, 0, ())
    assert stats.files_failed == 0


async def test_ingest_directory_preconditions(tmp_path: Path) -> None:
    pipeline = make_pipeline(FakeStore(collection="acme"))
    with pytest.raises(PipelineError, match="no such directory"):
        await pipeline.ingest_directory(tmp_path / "nope", collection="acme")
    f = tmp_path / "afile.txt"
    f.write_text("x")
    with pytest.raises(PipelineError, match="not a directory"):
        await pipeline.ingest_directory(f, collection="acme")


async def test_missing_collection_raises_before_work() -> None:
    store = FakeStore()  # no pre-registered collection
    pipeline = make_pipeline(store)
    with pytest.raises(PipelineError, match="not ready"):
        await pipeline.ingest_text("x", path="a.txt", collection="acme")


async def test_sync_incremental_new_modified_deleted(tmp_path: Path) -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, ingestor=FakeIngestor({}))
    await pipeline.ingest_text("alpha content", path="a.txt", collection="acme")
    await pipeline.ingest_text("old content", path="old.txt", collection="acme")
    await pipeline.ingest_text("c one", path="c.txt", collection="acme")

    # the sync directory: a.txt unchanged, c.txt modified, new.txt new,
    # old.txt missing (-> deleted)
    _write(tmp_path, {"a.txt": "alpha content", "c.txt": "c TWO content", "new.txt": "brand new"})

    events: list[FileEvent] = []
    sync_pipeline = make_pipeline(
        store,
        ingestor=FakeIngestor(
            {
                str(tmp_path / "a.txt"): "alpha content",
                str(tmp_path / "c.txt"): "c TWO content",
                str(tmp_path / "new.txt"): "brand new",
            }
        ),
        on_file_indexed=events.append,
    )
    stats = await sync_pipeline.sync(tmp_path, collection="acme")
    assert stats.files_deleted == 1
    assert stats.files_indexed == 2  # c.txt reindexed + new.txt
    assert stats.files_skipped == 1  # a.txt unchanged
    assert stats.chunks_indexed == 2
    assert store.source_count("acme") == 3
    assert {s.path for s in await store.list_sources("acme")} == {
        "a.txt",
        "c.txt",
        "new.txt",
    }
    # event order: deletions first (path-sorted), then pool events in
    # submission order; every file exactly once
    assert [(e.path, e.status) for e in events] == [
        ("old.txt", FileStatus.DELETED),
        ("a.txt", FileStatus.SKIPPED),
        ("c.txt", FileStatus.INDEXED),
        ("new.txt", FileStatus.INDEXED),
    ]


# --------------------------------------------------------------------------- #
# reindex / delete_source public ops
# --------------------------------------------------------------------------- #


async def test_reindex_public_method_uses_atomic_swap() -> None:
    store = FakeStore(collection="acme")
    sid = generate_source_id("acme", "a.txt")
    pipeline = make_pipeline(store, ingestor=FakeIngestor({"a.txt": "version two content"}))
    await pipeline.ingest_text("version one content", path="a.txt", collection="acme")
    stats = await pipeline.reindex("a.txt", collection="acme")
    assert stats.files_indexed == 1
    assert store.reindexed == [sid]  # one atomic swap, never delete+reingest
    assert store.upserted == [sid]  # no new upsert on reindex
    assert store.chunk_contents("acme") == [("version two content", 3)]


async def test_reindex_missing_source_raises() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store)
    with pytest.raises(PipelineError, match="no indexed source"):
        await pipeline.reindex("ghost.txt", collection="acme")


async def test_reindex_read_failure_recorded() -> None:
    store = FakeStore(collection="acme")
    await make_pipeline(store).ingest_text("v1", path="a.txt", collection="acme")
    pipeline = make_pipeline(store, ingestor=FakeIngestor({}, fail={"a.txt"}))
    stats = await pipeline.reindex("a.txt", collection="acme")
    (err,) = stats.errors
    assert err.phase is SourcePhase.READ and err.path == "a.txt"


async def test_delete_source_returns_existence() -> None:
    store = FakeStore(collection="acme")
    sid = generate_source_id("acme", "a.txt")
    pipeline = make_pipeline(store)
    await pipeline.ingest_text("x", path="a.txt", collection="acme")
    assert await pipeline.delete_source("acme", "a.txt") is True
    assert store.deleted == [sid]
    assert store.source_count("acme") == 0
    assert await pipeline.delete_source("acme", "a.txt") is False  # gone


async def test_ingest_file_public_op(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("file content words")
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, ingestor=FakeIngestor({str(f): "file content words"}))
    stats = await pipeline.ingest_file(f, collection="acme")
    assert stats.files_indexed == 1
    assert store.source_count("acme") == 1
    # canonical path stored as given
    assert (await store.list_sources("acme"))[0].path == str(f)


async def test_ingest_file_read_failure_recorded() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, ingestor=FakeIngestor({}, fail={"gone.txt"}))
    stats = await pipeline.ingest_file("gone.txt", collection="acme")
    (err,) = stats.errors
    assert err.path == "gone.txt" and err.phase is SourcePhase.READ
    assert err.error_type == "OSError"


async def test_source_error_str_format() -> None:
    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, chunker=BoomChunker("boom"))
    stats = await pipeline.ingest_text("boom", path="a.txt", collection="acme")
    (err,) = stats.errors
    s = str(err)
    assert "chunk error for 'a.txt'" in s
    assert "ChunkValidationError" in s


async def test_malformed_ingest_metadata_recorded() -> None:
    """A chonkai IngestResult missing its content_hash cannot become an
    embeddy SourceMetadata (which requires one) — the failure is COLLECTED
    as a READ-phase error, never raised."""
    from chonkai import SourceMetadata as ChonkaiSourceMetadata

    class MalformedIngestor(FakeIngestor):
        def ingest_file(self, path):
            return IngestResult(
                source=ChonkaiSourceMetadata(path=str(path), size_bytes=1, content_hash=None),
                text="content",
            )

    store = FakeStore(collection="acme")
    pipeline = make_pipeline(store, ingestor=MalformedIngestor())
    stats = await pipeline.ingest_file("a.txt", collection="acme")
    (err,) = stats.errors
    assert err.phase is SourcePhase.READ
    assert "content_hash" in err.message
    assert store.source_count("acme") == 0


async def test_cleanup_failure_keeps_original_error() -> None:
    """When a NEW source's add() fails AND the best-effort cleanup row-
    delete ALSO fails, the COLLECTED error is the original store error —
    the cleanup failure never masks it."""

    class FailingAddStore(FakeStore):
        async def add(self, collection, chunks, vectors) -> None:
            self.add_calls.append(len(chunks))
            raise RuntimeError("add exploded")

        async def delete_source(self, collection, source_id) -> None:
            self.deleted.append(source_id)
            raise RuntimeError("cleanup also failed")

    store = FailingAddStore(collection="acme")
    pipeline = make_pipeline(store)
    stats = await pipeline.ingest_text("content", path="a.txt", collection="acme")
    (err,) = stats.errors
    assert err.phase is SourcePhase.STORE
    assert err.message == "add exploded"  # the ORIGINAL error, not the cleanup one


def test_rekey_ingest_helper() -> None:
    """The canonical-path re-key is a no-op when paths already match and
    re-keys (text + hash preserved) otherwise."""
    from embeddy.pipeline import _rekey_ingest

    ingest = IngestResult.from_text("text", path="x.txt", content_type="text")
    assert _rekey_ingest(ingest, "x.txt") is ingest  # identity no-op
    rekeyed = _rekey_ingest(ingest, "docs/x.txt")
    assert rekeyed is not ingest
    assert rekeyed.source.path == "docs/x.txt"
    assert rekeyed.text == "text"
    assert rekeyed.source.content_hash == ingest.source.content_hash
    assert rekeyed.source.content_type == "text"
