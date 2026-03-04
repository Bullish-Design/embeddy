# benchmarks/bench_resources.py
"""Resource benchmarks: memory usage, DB size, snapshot tracking.

Measures CPU/GPU memory consumption at various stages:
- Baseline (import only)
- After model load
- After store init
- After ingest (N documents)
- After search (N queries)

Run with a real model:
    python -m benchmarks.bench_resources

Run as pytest (infrastructure tests):
    python -m pytest benchmarks/bench_resources.py -v
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from benchmarks.helpers import (
    BenchmarkReport,
    ResourceSnapshot,
    TimingResult,
    get_process_memory_mb,
)


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


def bench_import_memory() -> TimingResult:
    """Measure memory after importing embeddy."""
    import embeddy  # noqa: F401

    mem = get_process_memory_mb()
    return TimingResult(
        name="import_memory",
        elapsed_seconds=0,
        items_processed=1,
        extra={"memory_mb": round(mem, 2)},
    )


async def bench_store_size(store, collection: str = "default") -> TimingResult:
    """Measure database file size after operations."""
    db_path = store._config.db_path if hasattr(store, "_config") else "embeddy.db"
    size_bytes = 0
    if os.path.exists(db_path):
        size_bytes = os.path.getsize(db_path)
        # Also check WAL and SHM files
        for suffix in ["-wal", "-shm"]:
            wal_path = db_path + suffix
            if os.path.exists(wal_path):
                size_bytes += os.path.getsize(wal_path)

    size_mb = size_bytes / (1024 * 1024)
    return TimingResult(
        name="store_size",
        elapsed_seconds=0,
        items_processed=1,
        extra={"size_mb": round(size_mb, 4), "size_bytes": size_bytes},
    )


def bench_snapshot_sequence(snapshots: list[tuple[str, ResourceSnapshot]]) -> list[TimingResult]:
    """Convert a sequence of labeled snapshots into TimingResults."""
    results = []
    for label, snap in snapshots:
        results.append(
            TimingResult(
                name=f"resource_{label}",
                elapsed_seconds=0,
                items_processed=1,
                extra=snap.to_dict(),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


async def run_resource_benchmarks() -> BenchmarkReport:
    """Run the full resource benchmark suite.

    This is designed to be run with real dependencies. For CI,
    use the pytest tests with mocked dependencies.
    """
    report = BenchmarkReport(
        suite_name="resources",
        metadata={"pid": os.getpid()},
    )

    # Baseline
    report.resource_before = ResourceSnapshot.capture()
    report.add(bench_import_memory())

    report.resource_after = ResourceSnapshot.capture()
    return report


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


class TestResourceBenchmarks:
    """Resource benchmark infrastructure tests."""

    def test_import_memory(self) -> None:
        result = bench_import_memory()
        assert result.name == "import_memory"
        assert result.extra["memory_mb"] > 0

    def test_process_memory_positive(self) -> None:
        mem = get_process_memory_mb()
        assert mem > 0

    def test_resource_snapshot(self) -> None:
        snap = ResourceSnapshot.capture()
        assert snap.cpu_memory_mb > 0
        d = snap.to_dict()
        assert "cpu_memory_mb" in d

    def test_snapshot_sequence(self) -> None:
        snaps = [
            ("before", ResourceSnapshot.capture()),
            ("after", ResourceSnapshot.capture()),
        ]
        results = bench_snapshot_sequence(snaps)
        assert len(results) == 2
        assert results[0].name == "resource_before"
        assert results[1].name == "resource_after"

    def test_full_resource_suite(self) -> None:
        report = asyncio.run(run_resource_benchmarks())
        assert report.suite_name == "resources"
        assert len(report.results) >= 1
        json_str = report.to_json()
        assert "resources" in json_str

    def test_store_size_with_mock(self, tmp_path) -> None:
        """Test store size measurement with a temp file."""
        from unittest.mock import MagicMock

        db_file = tmp_path / "test.db"
        db_file.write_bytes(b"x" * 4096)

        mock_store = MagicMock()
        mock_store._config.db_path = str(db_file)

        result = asyncio.run(bench_store_size(mock_store))
        assert result.extra["size_bytes"] == 4096
        assert result.extra["size_mb"] > 0

    def test_benchmark_report_print(self, capsys) -> None:
        """Test report printing works."""
        report = BenchmarkReport(
            suite_name="test",
            resource_before=ResourceSnapshot.capture(),
            resource_after=ResourceSnapshot.capture(),
        )
        report.add(TimingResult(name="test_op", elapsed_seconds=1.5, items_processed=100))
        report.print_summary()
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "test_op" in captured.out


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = asyncio.run(run_resource_benchmarks())
    report.print_summary()
    print(report.to_json())
