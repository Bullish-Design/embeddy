# benchmarks/bench_pipeline.py
"""Pipeline benchmarks: end-to-end ingest throughput.

Measures text ingest, file ingest, and directory ingest performance
through the full pipeline (ingest -> chunk -> embed -> store).

Run with a real model:
    python -m benchmarks.bench_pipeline

Run as pytest (with mocked dependencies):
    python -m pytest benchmarks/bench_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from benchmarks.helpers import (
    BenchmarkReport,
    ResourceSnapshot,
    TimingResult,
    generate_code_snippet,
    generate_texts,
)
from embeddy.models import IngestStats


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


async def bench_text_ingest(
    pipeline,
    texts: list[str],
) -> TimingResult:
    """Benchmark ingesting text through the pipeline."""
    start = time.monotonic()
    total_chunks = 0
    for text in texts:
        stats = await pipeline.ingest_text(text)
        total_chunks += stats.chunks_stored
    elapsed = time.monotonic() - start
    return TimingResult(
        name="text_ingest",
        elapsed_seconds=elapsed,
        iterations=len(texts),
        items_processed=len(texts),
        extra={"total_chunks_stored": total_chunks},
    )


async def bench_file_ingest(
    pipeline,
    files: list[Path],
) -> TimingResult:
    """Benchmark ingesting files through the pipeline."""
    start = time.monotonic()
    total_chunks = 0
    for f in files:
        stats = await pipeline.ingest_file(f)
        total_chunks += stats.chunks_stored
    elapsed = time.monotonic() - start
    return TimingResult(
        name="file_ingest",
        elapsed_seconds=elapsed,
        iterations=len(files),
        items_processed=len(files),
        extra={"total_chunks_stored": total_chunks},
    )


async def bench_directory_ingest(
    pipeline,
    directory: Path,
) -> TimingResult:
    """Benchmark ingesting a directory through the pipeline."""
    start = time.monotonic()
    stats = await pipeline.ingest_directory(directory)
    elapsed = time.monotonic() - start
    return TimingResult(
        name="directory_ingest",
        elapsed_seconds=elapsed,
        iterations=1,
        items_processed=stats.files_processed,
        extra={
            "chunks_created": stats.chunks_created,
            "chunks_stored": stats.chunks_stored,
        },
    )


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


async def run_pipeline_benchmarks(
    pipeline,
    num_texts: int = 20,
    num_files: int = 10,
) -> BenchmarkReport:
    """Run the full pipeline benchmark suite."""
    texts = generate_texts(num_texts, sentences_per_text=8)

    report = BenchmarkReport(
        suite_name="pipeline",
        metadata={"num_texts": num_texts, "num_files": num_files},
    )
    report.resource_before = ResourceSnapshot.capture()

    # Text ingest benchmark
    report.add(await bench_text_ingest(pipeline, texts))

    # File ingest benchmark — create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        files: list[Path] = []
        for i in range(num_files):
            f = tmp / f"file_{i}.py"
            f.write_text(generate_code_snippet(lines=30, seed=i))
            files.append(f)

        report.add(await bench_file_ingest(pipeline, files))

        # Directory ingest benchmark
        report.add(await bench_directory_ingest(pipeline, tmp))

    report.resource_after = ResourceSnapshot.capture()
    return report


# ---------------------------------------------------------------------------
# Mock pipeline
# ---------------------------------------------------------------------------


def _make_mock_pipeline():
    """Create a mock pipeline returning synthetic IngestStats."""
    pipeline = MagicMock()

    def _text_stats(*args, **kwargs):
        return IngestStats(
            files_processed=1,
            chunks_created=3,
            chunks_embedded=3,
            chunks_stored=3,
            elapsed_seconds=0.01,
        )

    def _file_stats(*args, **kwargs):
        return IngestStats(
            files_processed=1,
            chunks_created=5,
            chunks_embedded=5,
            chunks_stored=5,
            elapsed_seconds=0.02,
        )

    def _dir_stats(*args, **kwargs):
        return IngestStats(
            files_processed=10,
            chunks_created=50,
            chunks_embedded=50,
            chunks_stored=50,
            elapsed_seconds=0.5,
        )

    pipeline.ingest_text = AsyncMock(side_effect=_text_stats)
    pipeline.ingest_file = AsyncMock(side_effect=_file_stats)
    pipeline.ingest_directory = AsyncMock(side_effect=_dir_stats)
    return pipeline


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


class TestPipelineBenchmarks:
    """Pipeline benchmark infrastructure tests."""

    def test_text_ingest_benchmark(self) -> None:
        pipeline = _make_mock_pipeline()
        texts = generate_texts(5)
        result = asyncio.run(bench_text_ingest(pipeline, texts))
        assert result.items_processed == 5
        assert result.extra["total_chunks_stored"] == 15  # 5 texts * 3 chunks each

    def test_file_ingest_benchmark(self, tmp_path) -> None:
        pipeline = _make_mock_pipeline()
        files = []
        for i in range(3):
            f = tmp_path / f"f_{i}.py"
            f.write_text(f"x = {i}")
            files.append(f)
        result = asyncio.run(bench_file_ingest(pipeline, files))
        assert result.items_processed == 3
        assert result.extra["total_chunks_stored"] == 15  # 3 files * 5 chunks each

    def test_directory_ingest_benchmark(self, tmp_path) -> None:
        pipeline = _make_mock_pipeline()
        (tmp_path / "a.py").write_text("x = 1")
        result = asyncio.run(bench_directory_ingest(pipeline, tmp_path))
        assert result.items_processed == 10  # from mock

    def test_full_pipeline_suite(self) -> None:
        pipeline = _make_mock_pipeline()
        report = asyncio.run(run_pipeline_benchmarks(pipeline, num_texts=5, num_files=3))
        assert report.suite_name == "pipeline"
        assert len(report.results) == 3
        json_str = report.to_json()
        assert "pipeline" in json_str


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pipeline benchmarks using mock data...")
    pipeline = _make_mock_pipeline()
    report = asyncio.run(run_pipeline_benchmarks(pipeline))
    report.print_summary()
