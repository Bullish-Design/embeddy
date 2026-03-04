# benchmarks/bench_encoding.py
"""Encoding benchmarks: single-text, batch, query, document throughput.

Run with a real model:
    python -m benchmarks.bench_encoding

Run as pytest-benchmark (with mocked model):
    python -m pytest benchmarks/bench_encoding.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from benchmarks.helpers import (
    BenchmarkReport,
    ResourceSnapshot,
    TimingResult,
    generate_texts,
)


# ---------------------------------------------------------------------------
# Core benchmark functions (work with real or mocked embedder)
# ---------------------------------------------------------------------------


async def bench_single_encode(embedder, texts: list[str], iterations: int = 10) -> TimingResult:
    """Benchmark encoding single texts one at a time."""
    subset = texts[:iterations]
    start = time.monotonic()
    for text in subset:
        await embedder.encode(text)
    elapsed = time.monotonic() - start
    return TimingResult(
        name="single_encode",
        elapsed_seconds=elapsed,
        iterations=len(subset),
        items_processed=len(subset),
    )


async def bench_batch_encode(embedder, texts: list[str], batch_size: int = 32) -> TimingResult:
    """Benchmark batch encoding."""
    start = time.monotonic()
    await embedder.encode(texts[:batch_size])
    elapsed = time.monotonic() - start
    return TimingResult(
        name=f"batch_encode_{batch_size}",
        elapsed_seconds=elapsed,
        iterations=1,
        items_processed=min(batch_size, len(texts)),
        extra={"batch_size": batch_size},
    )


async def bench_query_encode(embedder, queries: list[str]) -> TimingResult:
    """Benchmark query encoding (with query instruction)."""
    start = time.monotonic()
    for q in queries:
        await embedder.encode_query(q)
    elapsed = time.monotonic() - start
    return TimingResult(
        name="query_encode",
        elapsed_seconds=elapsed,
        iterations=len(queries),
        items_processed=len(queries),
    )


async def bench_document_encode(embedder, docs: list[str]) -> TimingResult:
    """Benchmark document encoding (with document instruction)."""
    start = time.monotonic()
    for doc in docs:
        await embedder.encode_document(doc)
    elapsed = time.monotonic() - start
    return TimingResult(
        name="document_encode",
        elapsed_seconds=elapsed,
        iterations=len(docs),
        items_processed=len(docs),
    )


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


async def run_encoding_benchmarks(embedder, num_texts: int = 50) -> BenchmarkReport:
    """Run the full encoding benchmark suite."""
    texts = generate_texts(num_texts, sentences_per_text=5)
    queries = generate_texts(10, sentences_per_text=2, seed=100)
    docs = generate_texts(10, sentences_per_text=10, seed=200)

    report = BenchmarkReport(
        suite_name="encoding",
        metadata={
            "model_name": embedder.model_name,
            "dimension": embedder.dimension,
            "num_texts": num_texts,
        },
    )
    report.resource_before = ResourceSnapshot.capture()

    report.add(await bench_single_encode(embedder, texts, iterations=10))
    for batch_size in [8, 16, 32]:
        report.add(await bench_batch_encode(embedder, texts, batch_size=batch_size))
    report.add(await bench_query_encode(embedder, queries))
    report.add(await bench_document_encode(embedder, docs))

    report.resource_after = ResourceSnapshot.capture()
    return report


# ---------------------------------------------------------------------------
# Mock embedder for pytest-benchmark tests
# ---------------------------------------------------------------------------


def _make_mock_embedder(dimension: int = 2048):
    """Create a mock embedder that returns random vectors."""
    from embeddy.models import Embedding

    def _make_embedding():
        vec = np.random.randn(dimension).astype(np.float32).tolist()
        return Embedding(vector=vec, model_name="mock-model", normalized=True)

    emb = MagicMock()
    emb.model_name = "mock-model"
    emb.dimension = dimension
    emb.encode = AsyncMock(
        side_effect=lambda inputs, **kw: [_make_embedding() for _ in (inputs if isinstance(inputs, list) else [inputs])]
    )
    emb.encode_query = AsyncMock(side_effect=lambda text: _make_embedding())
    emb.encode_document = AsyncMock(side_effect=lambda text: _make_embedding())
    return emb


# ---------------------------------------------------------------------------
# Pytest-benchmark tests (run with: pytest benchmarks/bench_encoding.py)
# ---------------------------------------------------------------------------


class TestEncodingBenchmarks:
    """Benchmark tests using mock embedder to verify infrastructure."""

    def test_single_encode_benchmark(self) -> None:
        embedder = _make_mock_embedder()
        result = asyncio.run(bench_single_encode(embedder, generate_texts(10), iterations=5))
        assert result.items_processed == 5
        assert result.elapsed_seconds >= 0
        assert result.items_per_second >= 0

    def test_batch_encode_benchmark(self) -> None:
        embedder = _make_mock_embedder()
        result = asyncio.run(bench_batch_encode(embedder, generate_texts(32), batch_size=16))
        assert result.items_processed == 16
        assert result.elapsed_seconds >= 0

    def test_query_encode_benchmark(self) -> None:
        embedder = _make_mock_embedder()
        queries = generate_texts(5, sentences_per_text=2)
        result = asyncio.run(bench_query_encode(embedder, queries))
        assert result.items_processed == 5

    def test_document_encode_benchmark(self) -> None:
        embedder = _make_mock_embedder()
        docs = generate_texts(5, sentences_per_text=10)
        result = asyncio.run(bench_document_encode(embedder, docs))
        assert result.items_processed == 5

    def test_full_suite_produces_report(self) -> None:
        embedder = _make_mock_embedder()
        report = asyncio.run(run_encoding_benchmarks(embedder, num_texts=20))
        assert report.suite_name == "encoding"
        assert len(report.results) == 6  # single + 3 batch + query + document
        assert report.resource_before is not None
        assert report.resource_after is not None

        # Verify JSON serialization works
        json_str = report.to_json()
        assert "encoding" in json_str

    def test_timing_result_properties(self) -> None:
        tr = TimingResult(name="test", elapsed_seconds=2.0, iterations=10, items_processed=100)
        assert tr.items_per_second == 50.0
        assert tr.avg_seconds == 0.2
        d = tr.to_dict()
        assert d["name"] == "test"
        assert d["items_per_second"] == 50.0


# ---------------------------------------------------------------------------
# Standalone runner (for real model benchmarks)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from embeddy.config import EmbedderConfig
    from embeddy.embedding import Embedder

    print("Loading model for encoding benchmarks...")
    config = EmbedderConfig()
    embedder = Embedder(config)

    report = asyncio.run(run_encoding_benchmarks(embedder))
    report.print_summary()
    print(report.to_json())
