# benchmarks/helpers.py
"""Shared utilities for benchmark scripts.

Provides sample text generation, timing helpers, resource measurement,
and report formatting used across all benchmark scripts.
"""

from __future__ import annotations

import json
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Sample text generation
# ---------------------------------------------------------------------------

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require careful tuning of hyperparameters.",
    "Embeddings represent semantic meaning as dense vectors.",
    "SQLite provides a serverless database engine with full SQL support.",
    "Python's asyncio library enables concurrent programming with coroutines.",
    "Vector search finds similar items by comparing embedding distances.",
    "Full-text search uses inverted indexes for keyword matching.",
    "Hybrid search combines vector and keyword approaches for better recall.",
    "Chunking breaks documents into smaller pieces for embedding.",
    "RAG pipelines retrieve relevant context before generating responses.",
    "The transformer architecture revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on relevant input tokens.",
    "Matryoshka Representation Learning enables flexible embedding dimensions.",
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Reciprocal Rank Fusion combines multiple ranked lists effectively.",
    "Content deduplication prevents storing redundant information.",
    "WAL mode in SQLite enables concurrent reads with single-writer access.",
    "FastAPI provides automatic OpenAPI schema generation for REST APIs.",
    "Async-native design ensures non-blocking I/O for high throughput.",
    "The cosine similarity metric measures angular distance between vectors.",
]


def generate_text(num_sentences: int = 5, seed: int | None = None) -> str:
    """Generate a random paragraph of text from sample sentences."""
    rng = random.Random(seed)
    return " ".join(rng.choices(_SENTENCES, k=num_sentences))


def generate_texts(count: int, sentences_per_text: int = 5, seed: int = 42) -> list[str]:
    """Generate multiple random text samples."""
    return [generate_text(sentences_per_text, seed=seed + i) for i in range(count)]


def generate_code_snippet(lines: int = 20, seed: int | None = None) -> str:
    """Generate a random Python code snippet."""
    rng = random.Random(seed)
    funcs = []
    for i in range(max(1, lines // 5)):
        name = "func_" + "".join(rng.choices(string.ascii_lowercase, k=6))
        args = ", ".join(rng.choices(["x", "y", "z", "data", "config", "items"], k=rng.randint(1, 3)))
        body_lines = [
            f"    {rng.choice(['result', 'value', 'output'])} = {rng.choice(['x', 'y', 'z'])} + {rng.randint(1, 100)}"
        ]
        body_lines.append(f"    return {rng.choice(['result', 'value', 'output'])}")
        funcs.append(f"def {name}({args}):\n" + "\n".join(body_lines))
    return "\n\n\n".join(funcs)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Result from a timed operation."""

    name: str
    elapsed_seconds: float
    iterations: int = 1
    items_processed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def items_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.items_processed / self.elapsed_seconds

    @property
    def avg_seconds(self) -> float:
        if self.iterations <= 0:
            return 0.0
        return self.elapsed_seconds / self.iterations

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "iterations": self.iterations,
            "items_processed": self.items_processed,
            "items_per_second": round(self.items_per_second, 2),
            "avg_seconds": round(self.avg_seconds, 6),
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Resource measurement
# ---------------------------------------------------------------------------


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_gpu_memory_mb() -> float | None:
    """Get GPU memory usage in MB (NVIDIA only)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return mem_info.used / (1024 * 1024)
    except Exception:
        return None


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""

    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float | None = None

    @classmethod
    def capture(cls) -> ResourceSnapshot:
        return cls(
            timestamp=time.monotonic(),
            cpu_memory_mb=get_process_memory_mb(),
            gpu_memory_mb=get_gpu_memory_mb(),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cpu_memory_mb": round(self.cpu_memory_mb, 2),
        }
        if self.gpu_memory_mb is not None:
            d["gpu_memory_mb"] = round(self.gpu_memory_mb, 2)
        return d


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Collection of benchmark results."""

    suite_name: str
    results: list[TimingResult] = field(default_factory=list)
    resource_before: ResourceSnapshot | None = None
    resource_after: ResourceSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, result: TimingResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "suite": self.suite_name,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }
        if self.resource_before:
            d["resource_before"] = self.resource_before.to_dict()
        if self.resource_after:
            d["resource_after"] = self.resource_after.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Benchmark Suite: {self.suite_name}")
        print(f"{'=' * 60}")

        if self.metadata:
            for k, v in self.metadata.items():
                print(f"  {k}: {v}")
            print()

        for r in self.results:
            print(f"  {r.name}:")
            print(f"    elapsed     : {r.elapsed_seconds:.4f}s")
            if r.iterations > 1:
                print(f"    iterations  : {r.iterations}")
                print(f"    avg/iter    : {r.avg_seconds:.6f}s")
            if r.items_processed > 0:
                print(f"    items       : {r.items_processed}")
                print(f"    throughput  : {r.items_per_second:.2f} items/s")
            for k, v in r.extra.items():
                print(f"    {k}: {v}")
            print()

        if self.resource_before and self.resource_after:
            delta_mb = self.resource_after.cpu_memory_mb - self.resource_before.cpu_memory_mb
            print(f"  Memory delta: {delta_mb:+.2f} MB")
            print(f"    before: {self.resource_before.cpu_memory_mb:.2f} MB")
            print(f"    after:  {self.resource_after.cpu_memory_mb:.2f} MB")

        print(f"{'=' * 60}\n")
