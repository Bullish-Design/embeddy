# tests/test_wikipedia_example.py
"""Tests for the Wikipedia example scripts (Phase 11).

Tests the download, ingest, search, and benchmark modules using mocks
so no real model loading, network access, or GPU is required.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: add examples/ to sys.path so we can import the modules
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "wikipedia"


@pytest.fixture(autouse=True)
def _patch_sys_path() -> None:
    """Ensure examples/wikipedia is importable."""
    path_str = str(EXAMPLES_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# download.py tests
# ---------------------------------------------------------------------------


class TestDownload:
    """Tests for the Wikipedia download module."""

    def test_import(self) -> None:
        import download  # noqa: F811

        assert hasattr(download, "Article")
        assert hasattr(download, "download_simple_wikipedia")
        assert hasattr(download, "save_articles")
        assert hasattr(download, "load_articles")

    def test_article_dataclass(self) -> None:
        from download import Article

        article = Article(title="Python", text="Python is a language.", article_id="1")
        assert article.title == "Python"
        assert article.text == "Python is a language."
        assert article.article_id == "1"

    def test_article_to_dict(self) -> None:
        from download import Article

        article = Article(title="Foo", text="Bar baz.", article_id="42")
        d = article.to_dict()
        assert d == {"title": "Foo", "text": "Bar baz.", "article_id": "42"}

    def test_article_from_dict(self) -> None:
        from download import Article

        d = {"title": "Foo", "text": "Bar baz.", "article_id": "42"}
        article = Article.from_dict(d)
        assert article.title == "Foo"
        assert article.text == "Bar baz."
        assert article.article_id == "42"

    def test_save_and_load_articles(self, tmp_path: Path) -> None:
        from download import Article, load_articles, save_articles

        articles = [
            Article(title="Alpha", text="Alpha text.", article_id="1"),
            Article(title="Beta", text="Beta text.", article_id="2"),
        ]
        out_file = tmp_path / "articles.jsonl"
        save_articles(articles, out_file)
        assert out_file.exists()

        loaded = load_articles(out_file)
        assert len(loaded) == 2
        assert loaded[0].title == "Alpha"
        assert loaded[1].title == "Beta"

    def test_save_articles_jsonl_format(self, tmp_path: Path) -> None:
        """Each line in the file should be valid JSON."""
        from download import Article, save_articles

        articles = [
            Article(title="A", text="A text.", article_id="1"),
            Article(title="B", text="B text.", article_id="2"),
        ]
        out_file = tmp_path / "articles.jsonl"
        save_articles(articles, out_file)

        lines = out_file.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "title" in obj
            assert "text" in obj
            assert "article_id" in obj

    def test_download_simple_wikipedia_with_mock(self, tmp_path: Path) -> None:
        """download_simple_wikipedia should fetch articles and save them."""
        from download import Article, download_simple_wikipedia

        fake_articles = [
            Article(title="Cat", text="A cat is a small domesticated carnivorous mammal." * 5, article_id="10"),
            Article(title="Dog", text="A dog is a domesticated descendant of the wolf." * 5, article_id="20"),
            Article(title="Fish", text="A fish is an aquatic animal that lives in water." * 5, article_id="30"),
        ]

        with patch("download._fetch_articles", return_value=fake_articles):
            result = download_simple_wikipedia(output_dir=tmp_path, max_articles=3)

        assert len(result) == 3
        assert (tmp_path / "articles.jsonl").exists()

    def test_download_respects_max_articles(self, tmp_path: Path) -> None:
        from download import Article, download_simple_wikipedia

        fake_articles = [
            Article(
                title=f"Article {i}",
                text=f"This is article number {i} with enough text to pass filter." * 5,
                article_id=str(i),
            )
            for i in range(100)
        ]

        with patch("download._fetch_articles", return_value=fake_articles):
            result = download_simple_wikipedia(output_dir=tmp_path, max_articles=5)

        assert len(result) == 5

    def test_download_filters_short_articles(self, tmp_path: Path) -> None:
        from download import Article, download_simple_wikipedia

        fake_articles = [
            Article(title="Short", text="Hi.", article_id="1"),  # too short
            Article(title="Long", text="This is a sufficiently long article with enough words." * 5, article_id="2"),
        ]

        with patch("download._fetch_articles", return_value=fake_articles):
            result = download_simple_wikipedia(
                output_dir=tmp_path,
                max_articles=10,
                min_length=50,
            )

        # Should only keep the long article
        assert len(result) >= 1
        assert all(len(a.text) >= 50 for a in result)


# ---------------------------------------------------------------------------
# ingest.py tests
# ---------------------------------------------------------------------------


class TestIngest:
    """Tests for the Wikipedia ingest module."""

    def test_import(self) -> None:
        import ingest  # noqa: F811

        assert hasattr(ingest, "ingest_articles")
        assert hasattr(ingest, "build_pipeline")

    @pytest.mark.asyncio
    async def test_ingest_articles_returns_stats(self, tmp_path: Path) -> None:
        from download import Article
        from ingest import ingest_articles

        articles = [
            Article(title="Python", text="Python is a programming language.", article_id="1"),
            Article(title="Java", text="Java is another programming language.", article_id="2"),
        ]

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_text = AsyncMock()

        # Make ingest_text return a mock IngestStats-like object
        from embeddy.models import IngestStats

        mock_pipeline.ingest_text.return_value = IngestStats(
            files_processed=1,
            chunks_created=2,
            chunks_embedded=2,
            chunks_stored=2,
            elapsed_seconds=0.01,
        )

        stats = await ingest_articles(articles, pipeline=mock_pipeline)

        assert mock_pipeline.ingest_text.call_count == 2
        assert stats.total_articles == 2
        assert stats.total_chunks_created >= 2
        assert stats.total_chunks_stored >= 2
        assert stats.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_ingest_articles_logs_errors(self, tmp_path: Path) -> None:
        from download import Article
        from ingest import ingest_articles

        articles = [
            Article(title="Bad", text="Will fail.", article_id="1"),
        ]

        mock_pipeline = AsyncMock()
        from embeddy.models import IngestStats, IngestError as IngestErrorModel

        mock_pipeline.ingest_text.return_value = IngestStats(
            files_processed=1,
            chunks_created=0,
            errors=[IngestErrorModel(file_path="Bad", error="test error", error_type="TestError")],
            elapsed_seconds=0.01,
        )

        stats = await ingest_articles(articles, pipeline=mock_pipeline)
        assert stats.total_errors >= 1

    @pytest.mark.asyncio
    async def test_ingest_articles_from_file(self, tmp_path: Path) -> None:
        """Test ingesting from a saved JSONL file."""
        from download import Article, save_articles
        from ingest import ingest_from_file

        articles = [
            Article(title="Test", text="Test article content.", article_id="1"),
        ]
        jsonl_path = tmp_path / "articles.jsonl"
        save_articles(articles, jsonl_path)

        mock_pipeline = AsyncMock()
        from embeddy.models import IngestStats

        mock_pipeline.ingest_text.return_value = IngestStats(
            files_processed=1,
            chunks_created=1,
            chunks_embedded=1,
            chunks_stored=1,
            elapsed_seconds=0.01,
        )

        stats = await ingest_from_file(jsonl_path, pipeline=mock_pipeline)
        assert stats.total_articles == 1

    @pytest.mark.asyncio
    async def test_ingest_with_progress(self, tmp_path: Path) -> None:
        """ingest_articles should call progress_callback if provided."""
        from download import Article
        from ingest import ingest_articles
        from embeddy.models import IngestStats

        articles = [Article(title=f"Art {i}", text=f"Text for article {i}.", article_id=str(i)) for i in range(5)]

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_text.return_value = IngestStats(
            files_processed=1,
            chunks_created=1,
            chunks_embedded=1,
            chunks_stored=1,
            elapsed_seconds=0.01,
        )

        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        await ingest_articles(articles, pipeline=mock_pipeline, progress_callback=on_progress)

        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)


# ---------------------------------------------------------------------------
# search.py tests
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for the Wikipedia search module."""

    def test_import(self) -> None:
        import search  # noqa: F811

        assert hasattr(search, "search_articles")
        assert hasattr(search, "format_results")

    @pytest.mark.asyncio
    async def test_search_articles_delegates_to_service(self) -> None:
        from search import search_articles
        from embeddy.models import SearchResults, SearchResult, SearchMode

        mock_service = AsyncMock()
        mock_service.search.return_value = SearchResults(
            results=[
                SearchResult(
                    chunk_id="c1",
                    content="Python is a language.",
                    score=0.95,
                    source_path="wikipedia:1",
                    metadata={"title": "Python"},
                ),
            ],
            query="python",
            collection="wikipedia",
            total_results=1,
            mode=SearchMode.HYBRID,
            elapsed_ms=15.0,
        )

        results = await search_articles("python", search_service=mock_service, collection="wikipedia")

        mock_service.search.assert_called_once()
        assert results.total_results == 1
        assert results.results[0].content == "Python is a language."

    @pytest.mark.asyncio
    async def test_search_articles_custom_params(self) -> None:
        from search import search_articles
        from embeddy.models import SearchResults, SearchMode

        mock_service = AsyncMock()
        mock_service.search.return_value = SearchResults(
            results=[],
            query="test",
            collection="wiki",
            total_results=0,
            mode=SearchMode.VECTOR,
            elapsed_ms=5.0,
        )

        await search_articles(
            "test",
            search_service=mock_service,
            collection="wiki",
            top_k=5,
            mode=SearchMode.VECTOR,
        )

        call_kwargs = mock_service.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 5 or call_kwargs[1].get("top_k") == 5

    def test_format_results_human_readable(self) -> None:
        from search import format_results
        from embeddy.models import SearchResults, SearchResult, SearchMode

        results = SearchResults(
            results=[
                SearchResult(
                    chunk_id="c1",
                    content="Python is a high-level programming language.",
                    score=0.95,
                    source_path="wikipedia:42",
                    metadata={"title": "Python (programming language)"},
                ),
                SearchResult(
                    chunk_id="c2",
                    content="Java is a popular language.",
                    score=0.85,
                    source_path="wikipedia:99",
                    metadata={"title": "Java (programming language)"},
                ),
            ],
            query="programming languages",
            collection="wikipedia",
            total_results=2,
            mode=SearchMode.HYBRID,
            elapsed_ms=25.0,
        )

        output = format_results(results)
        assert "Python" in output
        assert "Java" in output
        assert "0.95" in output or "95" in output
        assert "programming languages" in output.lower() or "2 result" in output.lower()

    def test_format_results_empty(self) -> None:
        from search import format_results
        from embeddy.models import SearchResults, SearchMode

        results = SearchResults(
            results=[],
            query="nonexistent",
            collection="wikipedia",
            total_results=0,
            mode=SearchMode.HYBRID,
            elapsed_ms=5.0,
        )

        output = format_results(results)
        assert "no result" in output.lower() or "0 result" in output.lower()

    def test_format_results_json(self) -> None:
        from search import format_results_json
        from embeddy.models import SearchResults, SearchResult, SearchMode

        results = SearchResults(
            results=[
                SearchResult(
                    chunk_id="c1",
                    content="Test.",
                    score=0.9,
                    source_path="wikipedia:1",
                    metadata={"title": "Test"},
                ),
            ],
            query="test",
            collection="wikipedia",
            total_results=1,
            mode=SearchMode.HYBRID,
            elapsed_ms=10.0,
        )

        output = format_results_json(results)
        parsed = json.loads(output)
        assert parsed["query"] == "test"
        assert len(parsed["results"]) == 1


# ---------------------------------------------------------------------------
# benchmark.py tests
# ---------------------------------------------------------------------------


class TestBenchmark:
    """Tests for the Wikipedia benchmark module."""

    def test_import(self) -> None:
        import benchmark  # noqa: F811

        assert hasattr(benchmark, "run_ingest_benchmark")
        assert hasattr(benchmark, "run_search_benchmark")
        assert hasattr(benchmark, "BenchmarkConfig")

    def test_benchmark_config_defaults(self) -> None:
        from benchmark import BenchmarkConfig

        config = BenchmarkConfig()
        assert config.num_articles > 0
        assert config.num_queries > 0
        assert config.collection == "wikipedia_bench"

    def test_benchmark_config_custom(self) -> None:
        from benchmark import BenchmarkConfig

        config = BenchmarkConfig(num_articles=500, num_queries=50, collection="custom")
        assert config.num_articles == 500
        assert config.num_queries == 50
        assert config.collection == "custom"

    @pytest.mark.asyncio
    async def test_run_ingest_benchmark(self) -> None:
        from benchmark import BenchmarkConfig, run_ingest_benchmark
        from download import Article
        from embeddy.models import IngestStats

        articles = [
            Article(title=f"Art {i}", text=f"Content for article number {i}." * 10, article_id=str(i))
            for i in range(10)
        ]

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_text.return_value = IngestStats(
            files_processed=1,
            chunks_created=3,
            chunks_embedded=3,
            chunks_stored=3,
            elapsed_seconds=0.01,
        )

        config = BenchmarkConfig(num_articles=10, collection="bench_test")
        result = await run_ingest_benchmark(articles, pipeline=mock_pipeline, config=config)

        assert result.total_articles == 10
        assert result.total_chunks > 0
        assert result.elapsed_seconds >= 0
        assert result.articles_per_second >= 0

    @pytest.mark.asyncio
    async def test_run_search_benchmark(self) -> None:
        from benchmark import BenchmarkConfig, run_search_benchmark
        from embeddy.models import SearchResults, SearchResult, SearchMode

        mock_service = AsyncMock()
        mock_service.search.return_value = SearchResults(
            results=[
                SearchResult(
                    chunk_id="c1",
                    content="result text",
                    score=0.9,
                    source_path="wikipedia:1",
                    metadata={},
                ),
            ],
            query="test",
            collection="bench_test",
            total_results=1,
            mode=SearchMode.HYBRID,
            elapsed_ms=10.0,
        )

        queries = ["what is python", "history of computing", "machine learning basics"]
        config = BenchmarkConfig(num_queries=3, collection="bench_test")
        result = await run_search_benchmark(queries, search_service=mock_service, config=config)

        assert result.total_queries == 9  # 3 queries x 3 default modes
        assert result.avg_latency_ms >= 0
        assert result.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_search_benchmark_records_per_mode(self) -> None:
        from benchmark import BenchmarkConfig, run_search_benchmark
        from embeddy.models import SearchResults, SearchMode

        mock_service = AsyncMock()
        mock_service.search.return_value = SearchResults(
            results=[],
            query="test",
            collection="bench_test",
            total_results=0,
            mode=SearchMode.HYBRID,
            elapsed_ms=5.0,
        )

        queries = ["q1", "q2"]
        config = BenchmarkConfig(
            num_queries=2,
            collection="bench_test",
            search_modes=["vector", "fulltext", "hybrid"],
        )
        result = await run_search_benchmark(queries, search_service=mock_service, config=config)

        # Should have run queries across all 3 modes
        assert mock_service.search.call_count == 6  # 2 queries x 3 modes

    def test_benchmark_result_to_dict(self) -> None:
        from benchmark import IngestBenchmarkResult

        result = IngestBenchmarkResult(
            total_articles=100,
            total_chunks=500,
            elapsed_seconds=10.0,
            articles_per_second=10.0,
            chunks_per_second=50.0,
            errors=0,
        )
        d = result.to_dict()
        assert d["total_articles"] == 100
        assert d["chunks_per_second"] == 50.0

    def test_search_benchmark_result_to_dict(self) -> None:
        from benchmark import SearchBenchmarkResult

        result = SearchBenchmarkResult(
            total_queries=30,
            elapsed_seconds=3.0,
            avg_latency_ms=100.0,
            p50_latency_ms=90.0,
            p95_latency_ms=200.0,
            p99_latency_ms=250.0,
            queries_per_second=10.0,
            per_mode={},
        )
        d = result.to_dict()
        assert d["total_queries"] == 30
        assert d["avg_latency_ms"] == 100.0
