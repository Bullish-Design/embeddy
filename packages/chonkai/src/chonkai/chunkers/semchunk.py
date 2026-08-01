"""semchunk-based token-accurate chunker (overlap + offsets).

`chunk_size` is a TOKEN budget: it is wired to `budget.max_tokens`, so the
chunks this chunker produces are token-accurate by construction (semchunk
3.2.5 verified API: `semchunk.chunk(text, chunk_size, token_counter,
memoize=True, offsets=False, overlap=None)` -> `list[str]` or
`(chunks, [(start, end) char offsets])` with `offsets=True`).

Char offsets from semchunk are mapped to 1-based, inclusive line ranges at
the public boundary (offsets may start/end mid-line; the range is the
smallest line span containing the offset slice).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from semchunk.semchunk import chunk as _semchunk_chunk  # stable pinned (<5)

from chonkai.chunkers.base import BaseChunker
from chonkai.models import Chunk, ChunkBudget, IngestResult
from chonkai.tokens import default_token_counter


def _line_of(text: str, offset: int) -> int:
    """1-based line number of a char offset in `text`."""
    return text.count("\n", 0, offset) + 1


class SemchunkChunker(BaseChunker):
    """Token-accurate text chunking via semchunk.

    `overlap` is passed through to semchunk: a float is a fraction of
    `chunk_size`, an int an absolute token count. `token_counter` must match
    the counter used by the ValidatedChunker wrapper for exact budgets.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int] | None = None,
        *,
        overlap: float | int | None = None,
        memoize: bool = True,
    ) -> None:
        self._counter = token_counter or default_token_counter()
        self._overlap = overlap
        self._memoize = memoize

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        text = ingest.text
        if budget is None:
            # no token budget -> no chunk_size constraint; the whole text is
            # one chunk (ValidatedChunker still enforces invariants)
            if not text.strip():
                return []
            return [
                Chunk(
                    content=text,
                    start_line=1,
                    end_line=text.count("\n") + 1,
                    chunk_type="paragraph",
                )
            ]
        chunks, offsets = cast(
            "tuple[list[str], list[tuple[int, int]]]",
            _semchunk_chunk(
                text,
                budget.max_tokens,
                self._counter,
                memoize=self._memoize,
                offsets=True,
                overlap=self._overlap,
            ),
        )
        out: list[Chunk] = []
        for content, (start, end) in zip(chunks, offsets, strict=True):
            out.append(
                Chunk(
                    content=content,
                    start_line=_line_of(text, start),
                    # end_line is the line of the chunk's LAST CHARACTER
                    # (a trailing newline does not open an empty next line)
                    end_line=_line_of(text, end - 1) if end > start else _line_of(text, end),
                    chunk_type="paragraph",
                )
            )
        return out
