"""Paragraph chunker: blank-line-delimited paragraphs, with short-merge.

Phase-2 refinement (plan §4): when a budget with `min_tokens > 0` is
supplied, consecutive short paragraphs are merged until the token floor is
reached. Merging stops early rather than push a merged chunk past
`max_tokens` (a single paragraph that alone exceeds the budget is emitted
whole — the ValidatedChunker token post-split is the contract). Without a
budget (or with `min_tokens == 0`) the Phase-1 behavior is preserved: one
chunk per paragraph.

Line ranges are 1-based, inclusive.
"""

from __future__ import annotations

from collections.abc import Callable

from chonkai.chunkers.base import BaseChunker
from chonkai.models import Chunk, ChunkBudget, IngestResult
from chonkai.tokens import default_token_counter

_Paragraph = tuple[str, int, int]  # (content, start_line, end_line)


class ParagraphChunker(BaseChunker):
    """Split `ingest.text` into paragraphs on blank lines, merging short ones."""

    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:
        self._counter = token_counter or default_token_counter()

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        paragraphs = _split_paragraphs(ingest.text)
        if budget is None or budget.min_tokens <= 0:
            return [_chunk_of(text, start, end) for text, start, end in paragraphs]
        return self._merge(paragraphs, budget)

    def _merge(self, paragraphs: list[_Paragraph], budget: ChunkBudget) -> list[Chunk]:
        out: list[Chunk] = []
        parts: list[_Paragraph] = []
        text = ""
        tokens = 0
        for p_text, p_start, p_end in paragraphs:
            candidate = text + "\n\n" + p_text if parts else p_text
            p_tokens = self._counter(candidate) - tokens
            if parts and tokens + p_tokens > budget.max_tokens:
                out.append(_join(parts))
                parts, text, tokens = [], "", 0
                candidate, p_tokens = p_text, self._counter(p_text)
            if p_tokens > budget.max_tokens:
                # a lone paragraph that already exceeds the budget: keep it
                # whole, ValidatedChunker post-splits it by tokens
                out.append(_chunk_of(p_text, p_start, p_end))
                continue
            parts.append((p_text, p_start, p_end))
            text = candidate
            tokens = self._counter(text)
            if tokens >= budget.min_tokens:
                out.append(_join(parts))
                parts, text, tokens = [], "", 0
        if parts:
            out.append(_join(parts))
        return out


def _split_paragraphs(text: str) -> list[_Paragraph]:
    """Split `text` into (content, start_line, end_line) paragraphs, 1-based."""
    paragraphs: list[_Paragraph] = []
    buf: list[str] = []
    start_line: int | None = None
    lines = text.split("\n")
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if start_line is None:
                start_line = idx
            buf.append(line)
        elif start_line is not None:
            paragraphs.append(("\n".join(buf), start_line, idx - 1))
            buf = []
            start_line = None
    if start_line is not None:
        paragraphs.append(("\n".join(buf), start_line, len(lines)))
    return paragraphs


def _join(parts: list[_Paragraph]) -> Chunk:
    return Chunk(
        content="\n\n".join(p[0] for p in parts),
        start_line=parts[0][1],
        end_line=parts[-1][2],
    )


def _chunk_of(content: str, start: int, end: int) -> Chunk:
    return Chunk(content=content, start_line=start, end_line=end)
