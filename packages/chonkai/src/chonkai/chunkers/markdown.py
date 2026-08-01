"""Markdown chunker: code-fence-aware headings, heading hierarchy -> parent.

Fixes the H4 finding (verified in the old codebase): `#`-prefixed lines
inside fenced code blocks are NOT headings. Outside fences, a heading line
updates a heading stack; body text under a heading becomes `paragraph`
chunks whose `parent` is the nearest heading (or None in the preamble);
fenced code blocks become a single `code` chunk per block (parent = the
heading in effect). Heading lines themselves become `heading` chunks with
`parent` = the enclosing heading.

The token budget is deliberately ignored here — the ValidatedChunker wrapper
enforces it (with token post-split) at the public boundary.
"""

from __future__ import annotations

import re

from chonkai.chunkers.base import BaseChunker
from chonkai.models import Chunk, ChunkBudget, IngestResult

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(\S.*)$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


class MarkdownChunker(BaseChunker):
    """Split `ingest.text` into headings, paragraphs and fenced code blocks."""

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        del budget  # token budget is ValidatedChunker's contract
        return _chunk_markdown(ingest.text)


def _chunk_markdown(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (heading level, heading text)
    para: list[str] = []
    para_start: int | None = None
    code: list[str] = []
    code_start: int | None = None
    fence_char: str | None = None

    def parent_of() -> str | None:
        return stack[-1][1] if stack else None

    def flush_para(end: int) -> None:
        nonlocal para, para_start
        if para_start is None:
            return
        chunks.append(
            Chunk(
                content="\n".join(para),
                start_line=para_start,
                end_line=end,
                chunk_type="paragraph",
                parent=parent_of(),
            )
        )
        para, para_start = [], None

    def flush_code(end: int) -> None:
        nonlocal code, code_start
        if code_start is None:
            return
        chunks.append(
            Chunk(
                content="\n".join(code),
                start_line=code_start,
                end_line=end,
                chunk_type="code",
                parent=parent_of(),
            )
        )
        code, code_start = [], None

    lines = text.split("\n")
    for idx, line in enumerate(lines, start=1):
        if fence_char is not None:
            code.append(line)
            if _FENCE_RE.match(line) and line.strip().startswith(fence_char):
                fence_char = None
                flush_code(idx)
            continue
        fence = _FENCE_RE.match(line)
        if fence:
            flush_para(idx - 1)
            code = [line]
            code_start = idx
            fence_char = line.strip()[0]
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush_para(idx - 1)
            level = len(heading.group(1))
            title = heading.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            chunks.append(
                Chunk(
                    content=line,
                    start_line=idx,
                    end_line=idx,
                    chunk_type="heading",
                    parent=parent_of(),
                )
            )
            stack.append((level, title))
            continue
        if line.strip():
            if para_start is None:
                para_start = idx
            para.append(line)
        else:
            flush_para(idx - 1)

    flush_para(len(lines))
    flush_code(len(lines))
    return chunks
