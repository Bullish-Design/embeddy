"""embeddy search / ingest / resource harness (plan §9, plan §11 tooling).

Measures the retrieval workflow on a deterministic synthetic corpus with the
FakeProvider + real SqliteStore:

  * ingest: wall time for N sources through IngestPipeline (read/chunk/
    embed/write, bounded pool) — and the resource profile (peak RSS) of the
    same ingest.
  * search: vector and hybrid latency on the ingested corpus (p50/p95 via
    repeated samples), including a filtered variant.

Assertions are correctness/regression checks only (stats, result shape,
memory bound) — NEVER wall-clock thresholds (timing is CI-noise; the
pytest-benchmark numbers are the measured output, not a gate).

Like all of benchmarks/, this is OUTSIDE pytest testpaths — the default
suite never collects it. Run explicitly:

    uv run pytest benchmarks/ --no-cov
"""

from __future__ import annotations

import asyncio
import resource
import time
from dataclasses import dataclass

import numpy as np
import pytest

from chonkai import ParagraphChunker
from embeddy import IngestPipeline
from embeddy.index.base import SearchFilters
from embeddy.index.sqlite import SqliteStore
from embeddy.providers.fake import FakeProvider
from embeddy.search import search_hybrid

DIM = 8
N_INGEST = 150
N_DOCS = 200
N_SEARCHES = 30


def tok(s: str) -> int:
    """Deterministic word-count token counter."""
    return len(s.split())


TOPICS = [
    ("rate limit", "rate limit backoff exponential jitter requests per minute"),
    ("token expiry", "token expiry refresh mint revocation 30 days"),
    ("chunking", "chunk token budget overlap offsets line ranges"),
    ("hybrid search", "hybrid search rrf fusion vector fulltext bm25"),
    ("storage", "storage vector index sqlite fts5 cosine distance"),
    ("pipeline", "pipeline ingest concurrency worker pool memory"),
    ("authentication", "auth login session key rotation"),
    ("retrieval", "retrieval ndcg recall ranking quality"),
]


def _doc(i: int) -> str:
    topic, words = TOPICS[i % len(TOPICS)]
    return f"{topic}: " + " ".join(words.split()[j % len(words.split())] for j in range(60))


@dataclass(frozen=True, slots=True)
class LoadedCorpus:
    """An ingested in-memory corpus + its store, ready for search."""

    store: SqliteStore


@pytest.fixture(scope="module")
def corpus() -> LoadedCorpus:
    """One ingested corpus shared by all search benchmarks (deterministic)."""

    async def _load() -> LoadedCorpus:
        store = await SqliteStore.open(":memory:")
        await store.create_collection("acme", DIM)
        pipeline = IngestPipeline(
            store=store,
            provider=FakeProvider(),
            chunker=ParagraphChunker(token_counter=tok),
            token_counter=tok,
            instruction="doc",
            concurrency=4,
        )
        for i in range(N_DOCS):
            await pipeline.ingest_text(
                _doc(i), collection="acme", path=f"doc-{i}.txt", content_type="text"
            )
        return LoadedCorpus(store)

    return asyncio.run(_load())


def _query_vector(query_text: str) -> np.ndarray:
    """Query vector, computed in SYNC context only (never inside a running
    loop — asyncio.run() rejects a running loop)."""
    return asyncio.run(FakeProvider().encode([query_text]))[0]


# --------------------------------------------------------------------------- #
# ingest: throughput + resource profile
# --------------------------------------------------------------------------- #


def test_bench_ingest(benchmark):
    """Full ingest path for N sources: store create + pipeline ingest_text."""

    async def _run() -> int:
        store = await SqliteStore.open(":memory:")
        try:
            await store.create_collection("acme", DIM)
            pipeline = IngestPipeline(
                store=store,
                provider=FakeProvider(),
                chunker=ParagraphChunker(token_counter=tok),
                token_counter=tok,
                instruction="doc",
                concurrency=4,
            )
            indexed = 0
            for i in range(N_INGEST):
                stats = await pipeline.ingest_text(_doc(i), collection="acme", path=f"doc-{i}.txt")
                indexed += stats.files_indexed
            return indexed
        finally:
            await store.close()

    # sync wrapper: pytest-benchmark times a plain callable
    assert benchmark(lambda: asyncio.run(_run())) == N_INGEST


def test_ingest_resource_profile():
    """Flat-memory regression: peak RSS stays bounded during a 300-doc ingest.

    ru_maxrss is process-wide peak RSS (Linux, KB) — a generous bound that
    only trips on a pathological accumulation (the plan §7 flat-memory
    contract); the measured value is printed for the record."""

    async def _ingest() -> int:
        store = await SqliteStore.open(":memory:")
        try:
            await store.create_collection("acme", DIM)
            pipeline = IngestPipeline(
                store=store,
                provider=FakeProvider(),
                chunker=ParagraphChunker(token_counter=tok),
                token_counter=tok,
                instruction="doc",
                concurrency=4,
            )
            for i in range(300):
                await pipeline.ingest_text(_doc(i), collection="acme", path=f"doc-{i}.txt")
            stats = await store.stats("acme")
            return stats.source_count
        finally:
            await store.close()

    assert asyncio.run(_ingest()) == 300
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[resource] peak RSS after 300-doc ingest: {peak_kb / 1024:.1f} MiB")
    assert peak_kb / 1024 < 2048, f"peak RSS {peak_kb / 1024:.1f} MiB exceeds the 2 GiB bound"


# --------------------------------------------------------------------------- #
# search: vector / hybrid / filtered latency on the fixed corpus
# --------------------------------------------------------------------------- #


def test_bench_search_vector(benchmark, corpus):
    qv = _query_vector("rate limit backoff")

    async def _run() -> list:
        return await corpus.store.search_vector("acme", qv, SearchFilters(), top_k=10)

    result = benchmark(lambda: asyncio.run(_run()))
    assert len(result) >= 1


def test_bench_search_hybrid(benchmark, corpus):
    qv = _query_vector("rate limit backoff")

    async def _run():
        return await search_hybrid(
            corpus.store,
            collection="acme",
            query_text="rate limit backoff",
            query_vector=qv,
            top_k=10,
        )

    result = benchmark(lambda: asyncio.run(_run()))
    assert len(result.results) >= 1


def test_bench_search_filtered(benchmark, corpus):
    qv = _query_vector("storage index")
    filters = SearchFilters(content_types=("text",))

    async def _run() -> list:
        return await corpus.store.search_vector("acme", qv, filters, top_k=10)

    result = benchmark(lambda: asyncio.run(_run()))
    assert len(result) >= 1


def test_search_latency_percentiles(corpus):
    """p50/p95 latency over N_SEARCHES queries (printed; no threshold gate)."""

    async def _one(q: str, qv: np.ndarray):
        return await search_hybrid(
            corpus.store,
            collection="acme",
            query_text=q,
            query_vector=qv,
            top_k=10,
        )

    times = []
    for i in range(N_SEARCHES):
        q, _ = TOPICS[i % len(TOPICS)]
        qv = _query_vector(q)  # sync context
        t0 = time.perf_counter_ns()
        result = asyncio.run(_one(q, qv))
        t1 = time.perf_counter_ns()
        assert len(result.results) >= 1
        times.append((t1 - t0) / 1_000_000)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1]
    print(f"\n[search] hybrid p50={p50:.2f}ms p95={p95:.2f}ms over {N_SEARCHES} queries")
