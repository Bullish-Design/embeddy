"""Chunker factory (get_chunker): content-type routing + config validation."""

from __future__ import annotations

import pytest

from chonkai import get_chunker
from chonkai.chunkers.docling import DoclingChunker
from chonkai.chunkers.markdown import MarkdownChunker
from chonkai.chunkers.paragraph import ParagraphChunker
from chonkai.chunkers.semchunk import SemchunkChunker
from chonkai.chunkers.treesitter import TreesitterChunker


def test_auto_routing_by_content_type() -> None:
    assert isinstance(get_chunker("python"), TreesitterChunker)
    assert isinstance(get_chunker("javascript"), TreesitterChunker)
    assert isinstance(get_chunker("rust"), TreesitterChunker)
    assert isinstance(get_chunker("markdown"), TreesitterChunker)
    assert isinstance(get_chunker("rst"), MarkdownChunker)
    assert isinstance(get_chunker("text"), SemchunkChunker)
    assert isinstance(get_chunker("txt-unknown"), SemchunkChunker)  # default
    assert isinstance(get_chunker(None), SemchunkChunker)
    assert isinstance(get_chunker("pdf"), DoclingChunker)
    assert isinstance(get_chunker("docx"), DoclingChunker)
    assert isinstance(get_chunker("html"), DoclingChunker)


def test_explicit_strategies() -> None:
    assert isinstance(get_chunker("markdown", strategy="paragraph"), ParagraphChunker)
    assert isinstance(get_chunker("python", strategy="markdown"), MarkdownChunker)
    assert isinstance(get_chunker("text", strategy="treesitter"), TreesitterChunker)
    assert isinstance(get_chunker("markdown", strategy="semchunk"), SemchunkChunker)
    assert isinstance(get_chunker("markdown", strategy="docling"), DoclingChunker)


def test_config_passthrough() -> None:
    chunker = get_chunker("python", config={"granularity": "class", "chunk_max_size": 512})
    assert isinstance(chunker, TreesitterChunker)
    chunker = get_chunker("text", config={"overlap": 0.2, "memoize": False})
    assert isinstance(chunker, SemchunkChunker)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="strategy"):
        get_chunker("text", strategy="bogus")


def test_unknown_config_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        get_chunker("python", config={"bogus": 1})
    with pytest.raises(ValueError, match="unknown config keys"):
        get_chunker("text", strategy="markdown", config={"overlap": 1})
