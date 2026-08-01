"""Docling chunker bridge: HybridChunker wrapper (chonkai[docling] extra).

Docling's HybridChunker produces chunks with heading metadata; the last
heading becomes `Chunk.parent` (plan §4 work item 6). The docling load is
lazy — this module is the only chunker-side lazy-loading module. When no
DoclingDocument is attached to the IngestResult
(e.g. markdown text ingested directly), the text is chunked with the
markdown chunker (docling text is exported as markdown).

Line ranges: docling chunk text is located within `ingest.text`; when
docling's normalized text cannot be located, the whole-document range is
used (documented coarse fallback).
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from chonkai.chunkers.base import BaseChunker
from chonkai.chunkers.markdown import MarkdownChunker
from chonkai.models import Chunk, ChunkBudget, IngestResult


class _HybridChunker(Protocol):
    """docling's HybridChunker (duck-typed)."""

    def chunk(self, document: object) -> Any: ...


def _load_hybrid_chunker(**kwargs: Any) -> _HybridChunker:
    """Lazily construct a docling HybridChunker (raises ImportError without
    the `docling` extra). Config kwargs are passed through verbatim."""
    from docling.chunking import HybridChunker  # lazy extra import

    return cast(_HybridChunker, HybridChunker(**kwargs))


def _locate_range(source: str, text: str) -> tuple[int, int] | None:
    """1-based (start, end) line range of the first occurrence of `text`."""
    start = source.find(text)
    if start < 0:
        return None
    end = start + len(text)
    return source.count("\n", 0, start) + 1, source.count("\n", 0, end) + 1


class DoclingChunker(BaseChunker):
    """Chunk a docling-parsed document via docling's HybridChunker.

    `tokenizer`, `max_tokens`, `merge_mode` and `overlap` are passed through
    to HybridChunker (docling defaults apply when None). `max_tokens` wins
    over the per-chunk `budget.max_tokens`.
    """

    def __init__(
        self,
        *,
        tokenizer: object | None = None,
        max_tokens: int | None = None,
        merge_mode: str | None = None,
        overlap: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._merge_mode = merge_mode
        self._overlap = overlap

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        if ingest.document is None:
            return MarkdownChunker().chunk(ingest, budget)

        size = self._max_tokens
        if size is None and budget is not None:
            size = budget.max_tokens
        kwargs: dict[str, Any] = {}
        if self._tokenizer is not None:
            kwargs["tokenizer"] = self._tokenizer
        if size is not None:
            kwargs["max_tokens"] = size
        if self._merge_mode is not None:
            kwargs["merge_mode"] = self._merge_mode
        if self._overlap is not None:
            kwargs["overlap"] = self._overlap

        chunker = _load_hybrid_chunker(**kwargs)
        total_lines = ingest.text.count("\n") + 1
        out: list[Chunk] = []
        for item in chunker.chunk(ingest.document):
            text = item.text
            headings = tuple(item.headings or ())
            parent = headings[-1] if headings else None
            span = _locate_range(ingest.text, text)
            start, end = span if span is not None else (1, total_lines)
            out.append(
                Chunk(
                    content=text,
                    start_line=start,
                    end_line=end,
                    chunk_type="paragraph",
                    parent=parent,
                )
            )
        return out
