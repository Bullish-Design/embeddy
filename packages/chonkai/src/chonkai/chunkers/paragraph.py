"""Paragraph chunker: blank-line-delimited paragraphs, one chunk each.

Phase-2 refinement (short-merge via ChunkBudget.min_tokens) is not built yet.
Line ranges are 1-based, inclusive.
"""

from __future__ import annotations

from chonkai.chunkers.base import BaseChunker
from chonkai.models import Chunk, ChunkBudget, IngestResult


class ParagraphChunker(BaseChunker):
    """Split `ingest.text` into paragraphs on blank lines."""

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        del budget  # Phase 2: short-merge uses min_tokens
        chunks: list[Chunk] = []
        buf: list[str] = []
        start_line: int | None = None
        for idx, line in enumerate(ingest.text.split("\n"), start=1):
            if line.strip():
                if start_line is None:
                    start_line = idx
                buf.append(line)
            elif start_line is not None:
                chunks.append(
                    Chunk(
                        content="\n".join(buf),
                        start_line=start_line,
                        end_line=idx - 1,
                    )
                )
                buf = []
                start_line = None
        if start_line is not None:
            chunks.append(
                Chunk(
                    content="\n".join(buf),
                    start_line=start_line,
                    end_line=len(ingest.text.split("\n")),
                )
            )
        return chunks
