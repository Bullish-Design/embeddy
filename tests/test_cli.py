# tests/test_cli.py
"""Tests for the embeddy CLI (Typer-based).

Tests use Typer's CliRunner with mocked heavy dependencies (Embedder,
VectorStore, Pipeline, SearchService) to verify argument parsing,
output formatting, and correct wiring of CLI commands.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from embeddy.models import IngestStats, SearchMode, SearchResults, SearchResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers: mock factory for the heavy deps
# ---------------------------------------------------------------------------


def _mock_embedder() -> MagicMock:
    """Create a mock Embedder with essential properties."""
    emb = MagicMock()
    emb.model_name = "Qwen/Qwen3-VL-Embedding-2B"
    emb.dimension = 2048
    return emb


def _mock_store() -> MagicMock:
    """Create a mock VectorStore."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.list_collections = AsyncMock(return_value=[])
    store.get_collection = AsyncMock(return_value=None)
    store.collection_stats = AsyncMock(
        return_value={
            "name": "default",
            "chunk_count": 10,
            "source_count": 3,
        }
    )
    return store


def _mock_pipeline() -> MagicMock:
    """Create a mock Pipeline."""
    pipeline = MagicMock()
    pipeline.ingest_text = AsyncMock(
        return_value=IngestStats(
            files_processed=1,
            chunks_created=5,
            chunks_embedded=5,
            chunks_stored=5,
            elapsed_seconds=0.42,
        )
    )
    pipeline.ingest_file = AsyncMock(
        return_value=IngestStats(
            files_processed=1,
            chunks_created=3,
            chunks_embedded=3,
            chunks_stored=3,
            elapsed_seconds=0.31,
        )
    )
    pipeline.ingest_directory = AsyncMock(
        return_value=IngestStats(
            files_processed=4,
            chunks_created=12,
            chunks_embedded=12,
            chunks_stored=12,
            elapsed_seconds=1.23,
        )
    )
    return pipeline


def _mock_search_service() -> MagicMock:
    """Create a mock SearchService."""
    svc = MagicMock()
    svc.search = AsyncMock(
        return_value=SearchResults(
            results=[
                SearchResult(
                    chunk_id="abc123",
                    content="Hello world",
                    score=0.95,
                    source_path="test.py",
                ),
                SearchResult(
                    chunk_id="def456",
                    content="Goodbye world",
                    score=0.82,
                    source_path="other.py",
                ),
            ],
            query="hello",
            collection="default",
            total_results=2,
            mode=SearchMode.HYBRID,
            elapsed_ms=15.5,
        )
    )
    return svc


# ---------------------------------------------------------------------------
# Patch target: we mock _build_deps in cli.main to avoid real model loading
# ---------------------------------------------------------------------------

_BUILD_DEPS_PATH = "embeddy.cli.main._build_deps"


def _fake_build_deps(config):
    """Return mock dependencies instead of real ones."""
    return _mock_embedder(), _mock_store(), _mock_pipeline(), _mock_search_service()


# ---------------------------------------------------------------------------
# Tests: embeddy --version
# ---------------------------------------------------------------------------


class TestVersion:
    """Test the --version flag."""

    def test_version_flag(self) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "embeddy" in result.output.lower()
        # Should contain version number
        import embeddy

        assert embeddy.__version__ in result.output


# ---------------------------------------------------------------------------
# Tests: embeddy info
# ---------------------------------------------------------------------------


class TestInfo:
    """Test the 'info' command."""

    def test_info_default(self) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "embeddy" in result.output.lower()
        # Should show version
        import embeddy

        assert embeddy.__version__ in result.output

    def test_info_shows_config_defaults(self) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        # Should mention model name and dimension defaults
        assert "Qwen" in result.output or "2048" in result.output or "embedder" in result.output.lower()


# ---------------------------------------------------------------------------
# Tests: embeddy serve
# ---------------------------------------------------------------------------


class TestServe:
    """Test the 'serve' command."""

    @patch("embeddy.cli.main.uvicorn")
    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_serve_default(self, mock_build, mock_uvicorn) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        # Default host and port
        assert call_kwargs.kwargs.get("host") or call_kwargs[1].get("host") == "127.0.0.1"

    @patch("embeddy.cli.main.uvicorn")
    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_serve_custom_host_port(self, mock_build, mock_uvicorn) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9090"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert "9090" in str(call_kwargs) or call_kwargs.kwargs.get("port") == 9090

    @patch("embeddy.cli.main.uvicorn")
    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_serve_with_config_file(self, mock_build, mock_uvicorn, tmp_path) -> None:
        from embeddy.cli.main import app

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "server": {"host": "0.0.0.0", "port": 7777},
                }
            )
        )
        result = runner.invoke(app, ["serve", "--config", str(config_file)])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()

    @patch("embeddy.cli.main.uvicorn")
    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_serve_with_db_path(self, mock_build, mock_uvicorn) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["serve", "--db", "/tmp/test.db"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: embeddy ingest
# ---------------------------------------------------------------------------


class TestIngest:
    """Test the 'ingest' subcommand group."""

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_text(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["ingest", "text", "Hello world, this is a test."])
        assert result.exit_code == 0
        assert "chunks_created" in result.output or "5" in result.output

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_text_with_collection(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(
            app,
            [
                "ingest",
                "text",
                "Some text",
                "--collection",
                "my-docs",
            ],
        )
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_text_with_source(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(
            app,
            [
                "ingest",
                "text",
                "Some text",
                "--source",
                "manual-entry",
            ],
        )
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_file(self, mock_build, tmp_path) -> None:
        from embeddy.cli.main import app

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        result = runner.invoke(app, ["ingest", "file", str(test_file)])
        assert result.exit_code == 0
        assert "chunks_created" in result.output or "3" in result.output

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_file_with_collection(self, mock_build, tmp_path) -> None:
        from embeddy.cli.main import app

        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello")

        result = runner.invoke(
            app,
            [
                "ingest",
                "file",
                str(test_file),
                "--collection",
                "docs",
            ],
        )
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_directory(self, mock_build, tmp_path) -> None:
        from embeddy.cli.main import app

        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")

        result = runner.invoke(app, ["ingest", "dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "files_processed" in result.output or "4" in result.output

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_directory_with_patterns(self, mock_build, tmp_path) -> None:
        from embeddy.cli.main import app

        (tmp_path / "a.py").write_text("x = 1")

        result = runner.invoke(
            app,
            [
                "ingest",
                "dir",
                str(tmp_path),
                "--include",
                "*.py",
                "--exclude",
                "*.pyc",
            ],
        )
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_directory_no_recursive(self, mock_build, tmp_path) -> None:
        from embeddy.cli.main import app

        (tmp_path / "a.py").write_text("x = 1")

        result = runner.invoke(
            app,
            [
                "ingest",
                "dir",
                str(tmp_path),
                "--no-recursive",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Tests: embeddy search
# ---------------------------------------------------------------------------


class TestSearch:
    """Test the 'search' command."""

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_basic(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello"])
        assert result.exit_code == 0
        assert "abc123" in result.output or "Hello world" in result.output

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_with_collection(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--collection", "docs"])
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_with_top_k(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--top-k", "5"])
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_with_mode(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--mode", "vector"])
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_with_mode_fulltext(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--mode", "fulltext"])
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_json_output(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--json"])
        assert result.exit_code == 0
        # Should be valid JSON
        parsed = json.loads(result.output)
        assert "results" in parsed

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_with_min_score(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--min-score", "0.5"])
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_shows_scores(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello"])
        assert result.exit_code == 0
        # Should show scores in human-readable output
        assert "0.95" in result.output or "score" in result.output.lower()

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_shows_source(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello"])
        assert result.exit_code == 0
        assert "test.py" in result.output


# ---------------------------------------------------------------------------
# Tests: embeddy ingest stats output
# ---------------------------------------------------------------------------


class TestIngestOutput:
    """Test ingest output formatting."""

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_text_json_output(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["ingest", "text", "Hello", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "chunks_created" in parsed
        assert parsed["chunks_created"] == 5

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_file_json_output(self, mock_build, tmp_path) -> None:
        from embeddy.cli.main import app

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        result = runner.invoke(app, ["ingest", "file", str(test_file), "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "chunks_created" in parsed


# ---------------------------------------------------------------------------
# Tests: CLI config resolution
# ---------------------------------------------------------------------------


class TestConfigResolution:
    """Test config file and option resolution."""

    def test_serve_rejects_invalid_port(self) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["serve", "--port", "abc"])
        assert result.exit_code != 0

    @patch("embeddy.cli.main.uvicorn")
    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_serve_with_log_level(self, mock_build, mock_uvicorn) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["serve", "--log-level", "debug"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert "debug" in str(call_kwargs).lower()


# ---------------------------------------------------------------------------
# Tests: embeddy ingest with DB path
# ---------------------------------------------------------------------------


class TestDBPath:
    """Test --db option propagation."""

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_ingest_with_db(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["ingest", "text", "hello", "--db", "/tmp/test.db"])
        assert result.exit_code == 0

    @patch(_BUILD_DEPS_PATH, side_effect=_fake_build_deps)
    def test_search_with_db(self, mock_build) -> None:
        from embeddy.cli.main import app

        result = runner.invoke(app, ["search", "hello", "--db", "/tmp/test.db"])
        assert result.exit_code == 0
