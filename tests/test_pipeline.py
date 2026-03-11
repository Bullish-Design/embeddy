# tests/test_pipeline.py
"""Tests for the pipeline layer.

TDD: Tests are written before the implementation.
Covers: text ingestion pipeline, file ingestion pipeline, directory ingestion,
        content-hash deduplication, reindex, delete_source, error handling,
        IngestStats reporting.

All tests mock the Embedder and VectorStore to avoid heavy dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from embeddy.config import ChunkConfig
from embeddy.exceptions import IngestError
from embeddy.models import (
    Chunk,
    Collection,
    ContentType,
    Embedding,
    IngestResult,
    IngestStats,
    SourceMetadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_embedder(dimension: int = 128, model_name: str = "test-model"):
    """Create a mock Embedder that returns fake embeddings."""
    embedder = AsyncMock()
    embedder.dimension = dimension
    embedder.model_name = model_name

    async def _encode_documents(texts):
        """Return one embedding per input text."""
        return [
            Embedding(
                vector=[0.1] * dimension,
                model_name=model_name,
                normalized=True,
            )
            for _ in texts
        ]

    embedder.encode.side_effect = _encode_documents
    return embedder


def _make_mock_store():
    """Create a mock VectorStore with async methods."""
    store = AsyncMock()
    store.get_collection = AsyncMock(
        return_value=Collection(
            name="test",
            dimension=128,
            model_name="test-model",
        )
    )
    store.create_collection = AsyncMock(
        return_value=Collection(
            name="test",
            dimension=128,
            model_name="test-model",
        )
    )
    store.add = AsyncMock()
    store.delete_by_source = AsyncMock(return_value=0)
    store.has_content_hash = AsyncMock(return_value=False)
    store.count = AsyncMock(return_value=0)
    return store


# ---------------------------------------------------------------------------
# Pipeline — construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    """Tests for Pipeline instantiation."""

    def test_pipeline_accepts_dependencies(self):
        """Pipeline can be constructed with embedder, store, and config."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
        )
        assert pipeline is not None

    def test_pipeline_default_collection(self):
        """Pipeline defaults to 'default' collection."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store)
        assert pipeline._collection == "default"

    def test_pipeline_custom_chunk_config(self):
        """Pipeline accepts a ChunkConfig override."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        config = ChunkConfig(strategy="paragraph", max_tokens=256)
        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            chunk_config=config,
        )
        assert pipeline._chunk_config.strategy == "paragraph"
        assert pipeline._chunk_config.max_tokens == 256


# ---------------------------------------------------------------------------
# Pipeline — ingest_text
# ---------------------------------------------------------------------------


class TestPipelineIngestText:
    """Tests for ingesting raw text through the pipeline."""

    async def test_ingest_text_returns_stats(self):
        """ingest_text returns an IngestStats."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_text("Hello, world! This is a test document.")

        assert isinstance(stats, IngestStats)
        assert stats.files_processed == 1
        assert stats.chunks_created > 0
        assert stats.chunks_embedded > 0
        assert stats.chunks_stored > 0

    async def test_ingest_text_calls_embedder(self):
        """Embedder.encode is called with the chunk contents."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        await pipeline.ingest_text("Hello, world! Some test text.")
        embedder.encode.assert_called()

    async def test_ingest_text_calls_store_add(self):
        """VectorStore.add is called with chunks and embeddings."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        await pipeline.ingest_text("Hello, world! Test content here.")
        store.add.assert_called()

        # Verify the add call received chunks and embeddings
        call_args = store.add.call_args
        assert call_args[0][0] == "test"  # collection name
        assert len(call_args[0][1]) > 0  # chunks
        assert len(call_args[0][2]) > 0  # embeddings
        assert len(call_args[0][1]) == len(call_args[0][2])  # same count

    async def test_ingest_text_with_explicit_content_type(self):
        """Caller can specify content_type for text ingestion."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_text(
            "def hello():\n    return 'hi'\n",
            content_type=ContentType.PYTHON,
        )
        assert stats.chunks_created > 0

    async def test_ingest_text_with_source(self):
        """ingest_text accepts a source identifier."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_text(
            "Some text content.",
            source="clipboard",
        )
        assert stats.files_processed == 1

    async def test_ingest_empty_text_reports_error(self):
        """Ingesting empty text results in an error in stats."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_text("")
        assert len(stats.errors) > 0
        assert stats.chunks_created == 0

    async def test_ingest_text_elapsed_seconds(self):
        """Stats include elapsed_seconds > 0."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_text("Hello, world!")
        assert stats.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# Pipeline — ingest_file
# ---------------------------------------------------------------------------


class TestPipelineIngestFile:
    """Tests for ingesting files through the pipeline."""

    async def test_ingest_python_file(self, tmp_path: Path):
        """Ingest a .py file through the pipeline."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text(
            textwrap.dedent("""\
            def hello():
                return 'hi'

            def goodbye():
                return 'bye'
        """)
        )

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(py_file)

        assert isinstance(stats, IngestStats)
        assert stats.files_processed == 1
        assert stats.chunks_created > 0
        assert stats.chunks_embedded > 0
        assert stats.chunks_stored > 0

    async def test_ingest_markdown_file(self, tmp_path: Path):
        """Ingest a .md file through the pipeline."""
        from embeddy.pipeline.pipeline import Pipeline

        md_file = tmp_path / "README.md"
        md_file.write_text("# Hello\n\nThis is a paragraph.\n\n## Section\n\nAnother paragraph.\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(md_file)
        assert stats.files_processed == 1
        assert stats.chunks_created > 0

    async def test_ingest_nonexistent_file_reports_error(self):
        """Ingesting a nonexistent file reports an error in stats."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(Path("/nonexistent/file.py"))
        assert len(stats.errors) > 0
        assert stats.files_processed == 0
        assert stats.chunks_created == 0

    async def test_ingest_file_ensures_collection(self, tmp_path: Path):
        """Pipeline creates the collection if it doesn't exist."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.get_collection = AsyncMock(return_value=None)  # Collection doesn't exist

        pipeline = Pipeline(embedder=embedder, store=store, collection="new_collection")

        stats = await pipeline.ingest_file(py_file)
        store.create_collection.assert_called()


# ---------------------------------------------------------------------------
# Pipeline — content-hash dedup
# ---------------------------------------------------------------------------


class TestPipelineDedup:
    """Tests for content-hash based deduplication."""

    async def test_skip_already_ingested_file(self, tmp_path: Path):
        """If a file's content_hash is already in the store, skip it."""
        from embeddy.pipeline.pipeline import Pipeline

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Same content as before.")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.has_content_hash = AsyncMock(return_value=True)  # Already exists

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(txt_file)
        assert stats.chunks_skipped > 0 or stats.files_processed == 0
        # Embedder should not be called when content is skipped
        embedder.encode.assert_not_called()

    async def test_ingest_new_file_not_skipped(self, tmp_path: Path):
        """A file with new content_hash should be fully processed."""
        from embeddy.pipeline.pipeline import Pipeline

        txt_file = tmp_path / "new.txt"
        txt_file.write_text("Brand new content!")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.has_content_hash = AsyncMock(return_value=False)

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(txt_file)
        assert stats.chunks_created > 0
        embedder.encode.assert_called()


# ---------------------------------------------------------------------------
# Pipeline — ingest_directory
# ---------------------------------------------------------------------------


class TestPipelineIngestDirectory:
    """Tests for directory ingestion."""

    async def test_ingest_directory_processes_files(self, tmp_path: Path):
        """ingest_directory processes all matching files in a directory."""
        from embeddy.pipeline.pipeline import Pipeline

        # Create test files
        (tmp_path / "a.py").write_text("def a():\n    pass\n")
        (tmp_path / "b.py").write_text("def b():\n    pass\n")
        (tmp_path / "c.txt").write_text("Some text.\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path)
        assert stats.files_processed == 3

    async def test_ingest_directory_recursive(self, tmp_path: Path):
        """ingest_directory recurses into subdirectories."""
        from embeddy.pipeline.pipeline import Pipeline

        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.py").write_text("x = 1\n")
        (sub / "nested.py").write_text("y = 2\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path, recursive=True)
        assert stats.files_processed == 2

    async def test_ingest_directory_with_include_filter(self, tmp_path: Path):
        """include filter limits which files are processed."""
        from embeddy.pipeline.pipeline import Pipeline

        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.txt").write_text("text\n")
        (tmp_path / "c.md").write_text("# Hello\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path, include=["*.py"])
        assert stats.files_processed == 1

    async def test_ingest_directory_with_exclude_filter(self, tmp_path: Path):
        """exclude filter skips matching files."""
        from embeddy.pipeline.pipeline import Pipeline

        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        (tmp_path / "c.txt").write_text("text\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path, exclude=["*.txt"])
        assert stats.files_processed == 2

    async def test_ingest_directory_nonexistent_reports_error(self):
        """Ingesting a nonexistent directory reports an error."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(Path("/nonexistent/dir"))
        assert len(stats.errors) > 0

    async def test_ingest_directory_non_recursive(self, tmp_path: Path):
        """With recursive=False, only top-level files are processed."""
        from embeddy.pipeline.pipeline import Pipeline

        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.py").write_text("x = 1\n")
        (sub / "nested.py").write_text("y = 2\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path, recursive=False)
        assert stats.files_processed == 1

    async def test_ingest_empty_directory(self, tmp_path: Path):
        """An empty directory produces zero files_processed."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path)
        assert stats.files_processed == 0


# ---------------------------------------------------------------------------
# Pipeline — reindex
# ---------------------------------------------------------------------------


class TestPipelineReindex:
    """Tests for reindexing a file."""

    async def test_reindex_deletes_old_then_ingests(self, tmp_path: Path):
        """reindex_file deletes existing chunks then re-ingests."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    pass\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=5)

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.reindex_file(py_file)
        store.delete_by_source.assert_called_once_with("test", str(py_file))
        assert stats.files_processed == 1
        assert stats.chunks_created > 0

    async def test_reindex_nonexistent_file(self):
        """Reindexing a nonexistent file reports an error."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.reindex_file(Path("/nonexistent/file.py"))
        assert len(stats.errors) > 0


# ---------------------------------------------------------------------------
# Pipeline — delete_source
# ---------------------------------------------------------------------------


class TestPipelineDeleteSource:
    """Tests for deleting all chunks from a source."""

    async def test_delete_source_calls_store(self):
        """delete_source delegates to VectorStore.delete_by_source."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=10)

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        deleted = await pipeline.delete_source("path/to/file.py")
        store.delete_by_source.assert_called_once_with("test", "path/to/file.py")
        assert deleted == 10


# ---------------------------------------------------------------------------
# Pipeline — error handling
# ---------------------------------------------------------------------------


class TestPipelineErrorHandling:
    """Tests for error handling in the pipeline."""

    async def test_embedder_failure_reports_error(self, tmp_path: Path):
        """If the embedder fails, stats report the error."""
        from embeddy.pipeline.pipeline import Pipeline

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some content here.")

        embedder = _make_mock_embedder()
        embedder.encode.side_effect = Exception("GPU exploded")
        store = _make_mock_store()

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(txt_file)
        assert len(stats.errors) > 0
        assert "GPU" in stats.errors[0].error or "exploded" in stats.errors[0].error

    async def test_store_failure_reports_error(self, tmp_path: Path):
        """If the store fails on add, stats report the error."""
        from embeddy.pipeline.pipeline import Pipeline

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some content here.")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.add.side_effect = Exception("DB locked")

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(txt_file)
        assert len(stats.errors) > 0

    async def test_directory_partial_failure(self, tmp_path: Path):
        """If some files fail in directory ingest, others still succeed."""
        from embeddy.pipeline.pipeline import Pipeline

        (tmp_path / "good.txt").write_text("Good content.\n")
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("")  # Empty — will be rejected by ingestor

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_directory(tmp_path)
        # At least the good file should succeed
        assert stats.files_processed >= 1 or len(stats.errors) >= 1


# ---------------------------------------------------------------------------
# Pipeline — on_file_indexed callback hook
# ---------------------------------------------------------------------------


class TestPipelineOnFileIndexedHook:
    """Tests for the on_file_indexed callback hook."""

    def test_hook_defaults_to_none(self):
        """Pipeline defaults on_file_indexed to None (backward compat)."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store)
        assert pipeline._on_file_indexed is None

    async def test_hook_called_after_ingest_file(self, tmp_path: Path):
        """on_file_indexed is called after successful ingest_file."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    pass\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        stats = await pipeline.ingest_file(py_file)
        hook.assert_called_once()
        call_args = hook.call_args
        assert call_args[0][0] == str(py_file)  # source path
        assert isinstance(call_args[0][1], IngestStats)  # stats

    async def test_hook_called_after_ingest_text(self):
        """on_file_indexed is called after successful ingest_text."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        stats = await pipeline.ingest_text("Hello, world! Test content.", source="clipboard")
        hook.assert_called_once()
        call_args = hook.call_args
        assert call_args[0][0] == "clipboard"  # source
        assert isinstance(call_args[0][1], IngestStats)

    async def test_hook_called_after_reindex_file(self, tmp_path: Path):
        """on_file_indexed is called after successful reindex_file."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    pass\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=3)
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        stats = await pipeline.reindex_file(py_file)
        hook.assert_called_once()
        call_args = hook.call_args
        assert call_args[0][0] == str(py_file)
        assert isinstance(call_args[0][1], IngestStats)

    async def test_hook_not_called_on_error(self):
        """on_file_indexed is NOT called when ingestion has errors."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        # Ingest empty text -> will error
        stats = await pipeline.ingest_text("")
        assert len(stats.errors) > 0
        hook.assert_not_called()

    async def test_hook_not_called_on_file_not_found(self):
        """on_file_indexed is NOT called when file doesn't exist."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        stats = await pipeline.ingest_file(Path("/nonexistent/file.py"))
        hook.assert_not_called()

    async def test_async_hook_is_awaited(self, tmp_path: Path):
        """An async on_file_indexed callback is properly awaited."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()

        captured: list[tuple[str, IngestStats]] = []

        async def async_hook(source: str, stats: IngestStats) -> None:
            captured.append((source, stats))

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=async_hook,
        )

        await pipeline.ingest_file(py_file)
        assert len(captured) == 1
        assert captured[0][0] == str(py_file)
        assert isinstance(captured[0][1], IngestStats)

    async def test_sync_hook_works(self, tmp_path: Path):
        """A plain sync on_file_indexed callback works."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()

        captured: list[tuple[str, IngestStats]] = []

        def sync_hook(source: str, stats: IngestStats) -> None:
            captured.append((source, stats))

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=sync_hook,
        )

        await pipeline.ingest_file(py_file)
        assert len(captured) == 1
        assert captured[0][0] == str(py_file)

    async def test_hook_not_called_for_skipped_dedup(self, tmp_path: Path):
        """on_file_indexed is NOT called when file is skipped due to dedup."""
        from embeddy.pipeline.pipeline import Pipeline

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Same content as before.")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.has_content_hash = AsyncMock(return_value=True)
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        stats = await pipeline.ingest_file(txt_file)
        hook.assert_not_called()

    async def test_hook_receives_source_none_for_text_without_source(self):
        """When ingest_text has no source, hook receives None as source."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        hook = MagicMock()

        pipeline = Pipeline(
            embedder=embedder,
            store=store,
            collection="test",
            on_file_indexed=hook,
        )

        stats = await pipeline.ingest_text("Hello, world! Test content.")
        hook.assert_called_once()
        call_args = hook.call_args
        assert call_args[0][0] is None  # no source


# ---------------------------------------------------------------------------
# Pipeline — stats enrichment (collection, content_hash, chunks_removed)
# ---------------------------------------------------------------------------


class TestPipelineStatsEnrichment:
    """Tests for populating new IngestStats fields."""

    async def test_ingest_file_sets_collection(self, tmp_path: Path):
        """ingest_file populates stats.collection from pipeline's collection."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="emb_code")

        stats = await pipeline.ingest_file(py_file)
        assert stats.collection == "emb_code"

    async def test_ingest_text_sets_collection(self):
        """ingest_text populates stats.collection from pipeline's collection."""
        from embeddy.pipeline.pipeline import Pipeline

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="my_col")

        stats = await pipeline.ingest_text("Hello, world!")
        assert stats.collection == "my_col"

    async def test_reindex_sets_collection(self, tmp_path: Path):
        """reindex_file populates stats.collection."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=2)
        pipeline = Pipeline(embedder=embedder, store=store, collection="reindex_col")

        stats = await pipeline.reindex_file(py_file)
        assert stats.collection == "reindex_col"

    async def test_ingest_file_sets_content_hash(self, tmp_path: Path):
        """ingest_file populates stats.content_hash from IngestResult."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(py_file)
        # The Ingestor computes a content_hash from file content
        assert stats.content_hash is not None
        assert len(stats.content_hash) > 0

    async def test_reindex_sets_content_hash(self, tmp_path: Path):
        """reindex_file populates stats.content_hash."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=2)
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.reindex_file(py_file)
        assert stats.content_hash is not None

    async def test_reindex_captures_chunks_removed(self, tmp_path: Path):
        """reindex_file captures chunks_removed from delete_by_source return."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    pass\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=7)

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.reindex_file(py_file)
        assert stats.chunks_removed == 7

    async def test_reindex_chunks_removed_zero_when_nothing_deleted(self, tmp_path: Path):
        """chunks_removed is 0 when delete_by_source returns 0."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        store.delete_by_source = AsyncMock(return_value=0)

        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.reindex_file(py_file)
        assert stats.chunks_removed == 0

    async def test_ingest_file_chunks_removed_stays_zero(self, tmp_path: Path):
        """ingest_file (not reindex) has chunks_removed == 0."""
        from embeddy.pipeline.pipeline import Pipeline

        py_file = tmp_path / "example.py"
        py_file.write_text("x = 1\n")

        embedder = _make_mock_embedder()
        store = _make_mock_store()
        pipeline = Pipeline(embedder=embedder, store=store, collection="test")

        stats = await pipeline.ingest_file(py_file)
        assert stats.chunks_removed == 0
