"""ValidatedChunker — the single point where chunk invariants are enforced.

Guarantees on every chunk (CONCEPT §3.5, plan §4):
  * content is non-empty
  * start_line / end_line are present and valid (1-based, inclusive)
  * chunk_type is in the known vocabulary
  * token_count <= budget.max_tokens — a chunk that exceeds the budget is
    POST-SPLIT on token windows (replacing the M1 raise-on-violation) so the
    invariant holds even for oversized inputs. Pieces inherit the parent
    chunk's chunk_type/parent/granularity.

`token_counter` is injectable so callers control token accuracy (embeddy
passes a tiktoken/HF-tokenizers counter; tests pass a deterministic one).
The default counter is `chonkai.tokens.default_token_counter` (tiktoken
cl100k_base, deterministic chars/4 fallback for offline machines).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from chonkai.chunkers.base import BaseChunker
from chonkai.models import CHUNK_TYPES, Chunk, ChunkBudget, IngestResult
from chonkai.tokens import default_token_counter


class ChunkValidationError(ValueError):
    """A chunker output violated a documented invariant."""


def split_by_tokens(
    content: str,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[str]:
    """Token-window split of `content` into pieces each <= max_tokens.

    Line boundaries are preferred; a single over-long line is split on word
    boundaries, and an over-long word on char boundaries (a last resort so
    the token invariant holds on adversarial inputs). Empty trailing pieces
    are dropped.
    """
    pieces = _split_text(content, max_tokens, token_counter, ("\n", " ", ""))
    return [p for p in pieces if p] or [content]


def _split_text(
    content: str,
    max_tokens: int,
    token_counter: Callable[[str], int],
    separators: tuple[str, ...],
) -> list[str]:
    if token_counter(content) <= max_tokens or not content:
        return [content]
    separator = separators[0]
    rest = separators[1:]
    if separator == "":
        # char-level greedy accumulation — always terminates
        pieces: list[str] = []
        cur = ""
        for ch in content:
            candidate = cur + ch
            if cur and token_counter(candidate) > max_tokens:
                pieces.append(cur)
                cur = ch
            else:
                cur = candidate
        if cur:
            pieces.append(cur)
        return pieces or [content]
    parts = content.split(separator)
    if len(parts) <= 1:
        return _split_text(content, max_tokens, token_counter, rest)
    pieces = []
    cur = ""
    for part in parts:
        candidate = part if not cur else cur + separator + part
        if cur and token_counter(candidate) > max_tokens:
            pieces.append(cur)
            cur = part
        else:
            cur = candidate
    if cur:
        pieces.append(cur)
    out: list[str] = []
    for piece in pieces:
        out.extend(_split_text(piece, max_tokens, token_counter, rest))
    return out


class ValidatedChunker(BaseChunker):
    """Wraps any chunker and enforces the chunk invariants."""

    def __init__(
        self,
        inner: BaseChunker,
        *,
        budget: ChunkBudget | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._counter = token_counter or default_token_counter()

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        effective = budget if budget is not None else self._budget
        out: list[Chunk] = []
        for chunk in self._inner.chunk(ingest, effective):
            self._validate(chunk)
            out.extend(self._ensure_budget(chunk, effective))
        return out

    def _validate(self, chunk: Chunk) -> None:
        """Structural invariants; raises ChunkValidationError on violation."""
        if not chunk.content.strip():
            raise ChunkValidationError("chunk content must be non-empty")
        if chunk.start_line < 1 or chunk.end_line < chunk.start_line:
            raise ChunkValidationError(f"invalid line range ({chunk.start_line}-{chunk.end_line})")
        if chunk.chunk_type not in CHUNK_TYPES:
            raise ChunkValidationError(
                f"unknown chunk_type {chunk.chunk_type!r}; expected one of {sorted(CHUNK_TYPES)}"
            )

    def _ensure_budget(self, chunk: Chunk, budget: ChunkBudget | None) -> list[Chunk]:
        """Fill token_count; post-split when the chunk exceeds max_tokens."""
        tokens = self._counter(chunk.content)
        if budget is None or tokens <= budget.max_tokens:
            return [replace(chunk, token_count=tokens)]
        pieces = split_by_tokens(chunk.content, budget.max_tokens, self._counter)
        out: list[Chunk] = []
        line = chunk.start_line
        for piece in pieces:
            piece_lines = piece.count("\n") + 1
            out.extend(
                self._ensure_budget(
                    replace(
                        chunk,
                        content=piece,
                        start_line=line,
                        end_line=line + piece_lines - 1,
                    ),
                    budget,
                )
            )
            line += piece_lines
        return out
