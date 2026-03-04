# benchmarks/bench_search.py
"""Search benchmarks: vector, fulltext, hybrid latency + throughput.

Run with a real model (requires pre-populated DB):
    python -m benchmarks.bench_search

Run as pytest (with mocked dependencies):
    python -m pytest benchmarks/bench_search.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from benchmarks.helpers import (
    BenchmarkReport,
    ResourceSnapshot,
    TimingResult,
    generate_texts,
)
from embeddy.models import SearchMode, SearchResult, SearchResults


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


async def bench_vector_search(
    search_service,
    queries: list[str],
    collection: str = "default",
    top_k: int = 10,
) -> TimingResult:
    """Benchmark vector (KNN) search latency."""
    start = time.monotonic()
    for q in queries:
        await search_service.search(
            query=q,
            collection=collection,
            top_k=top_k,
            mode=SearchMode.VECTOR,
        )
    elapsed = time.monotonic() - start
    return TimingResult(
        name="vector_search",
        elapsed_seconds=elapsed,
        iterations=len(queries),
        items_processed=len(queries),
        extra={"top_k": top_k, "mode": "vector"},
    )


async def bench_fulltext_search(
    search_service,
    queries: list[str],
    collection: str = "default",
    top_k: int = 10,
) -> TimingResult:
    """Benchmark full-text (BM25) search latency."""
    start = time.monotonic()
    for q in queries:
        await search_service.search(
            query=q,
            collection=collection,
            top_k=top_k,
            mode=SearchMode.FULLTEXT,
        )
    elapsed = time.monotonic() - start
    return TimingResult(
        name="fulltext_search",
        elapsed_seconds=elapsed,
        iterations=len(queries),
        items_processed=len(queries),
        extra={"top_k": top_k, "mode": "fulltext"},
    )


async def bench_hybrid_search(
    search_service,
    queries: list[str],
    collection: str = "default",
    top_k: int = 10,
) -> TimingResult:
    """Benchmark hybrid (vector + fulltext + fusion) search latency."""
    start = time.monotonic()
    for q in queries:
        await search_service.search(
            query=q,
            collection=collection,
            top_k=top_k,
            mode=SearchMode.HYBRID,
        )
    elapsed = time.monotonic() - start
    return TimingResult(
        name="hybrid_search",
        elapsed_seconds=elapsed,
        iterations=len(queries),
        items_processed=len(queries),
        extra={"top_k": top_k, "mode": "hybrid"},
    )


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


async def run_search_benchmarks(
    search_service,
    num_queries: int = 20,
    collection: str = "default",
) -> BenchmarkReport:
    """Run the full search benchmark suite."""
    queries = generate_texts(num_queries, sentences_per_text=2, seed=300)

    report = BenchmarkReport(
        suite_name="search",
        metadata={"num_queries": num_queries, "collection": collection},
    )
    report.resource_before = ResourceSnapshot.capture()

    for top_k in [5, 10, 20]:
        report.add(await bench_vector_search(search_service, queries[:10], collection, top_k))
        report.add(await bench_fulltext_search(search_service, queries[:10], collection, top_k))
        report.add(await bench_hybrid_search(search_service, queries[:10], collection, top_k))

    report.resource_after = ResourceSnapshot.capture()
    return report


# ---------------------------------------------------------------------------
# Mock search service
# ---------------------------------------------------------------------------


def _make_mock_search_service():
    """Create a mock search service returning synthetic results."""
    svc = MagicMock()

    def _make_results(query, **kwargs):
        return SearchResults(
            results=[
                SearchResult(
                    chunk_id=f"chunk_{i}",
                    content=f"Result {i} for '{query}'",
                    score=0.95 - i * 0.05,
                    source_path=f"file_{i}.py",
                )
                for i in range(min(kwargs.get("top_k", 10), 5))
            ],
            query=query,
            collection=kwargs.get("collection", "default"),
            total_results=5,
            mode=kwargs.get("mode", SearchMode.HYBRID),
            elapsed_ms=5.0,
        )

    svc.search = AsyncMock(side_effect=lambda **kw: _make_results(**kw))
    return svc


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


class TestSearchBenchmarks:
    """Search benchmark infrastructure tests."""

    def test_vector_search_benchmark(self) -> None:
        svc = _make_mock_search_service()
        queries = generate_texts(5, sentences_per_text=2)
        result = asyncio.run(bench_vector_search(svc, queries))
        assert result.items_processed == 5
        assert result.elapsed_seconds >= 0

    def test_fulltext_search_benchmark(self) -> None:
        svc = _make_mock_search_service()
        queries = generate_texts(5, sentences_per_text=2)
        result = asyncio.run(bench_fulltext_search(svc, queries))
        assert result.items_processed == 5

    def test_hybrid_search_benchmark(self) -> None:
        svc = _make_mock_search_service()
        queries = generate_texts(5, sentences_per_text=2)
        result = asyncio.run(bench_hybrid_search(svc, queries))
        assert result.items_processed == 5

    def test_full_search_suite(self) -> None:
        svc = _make_mock_search_service()
        report = asyncio.run(run_search_benchmarks(svc, num_queries=10))
        assert report.suite_name == "search"
        assert len(report.results) == 9  # 3 modes * 3 top_k values
        json_str = report.to_json()
        assert "search" in json_str


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Search benchmarks require a pre-populated database.")
    print("Run with: python -m benchmarks.bench_search")
    print("Using mock data for demonstration...")

    svc = _make_mock_search_service()
    report = asyncio.run(run_search_benchmarks(svc))
    report.print_summary()
