"""ValidatedChunker — the single point where chunk invariants are enforced.

Guarantees on every chunk (CONCEPT §3.5, plan Phase 1):
  * content is non-empty
  * start_line / end_line are present and valid (1-based, inclusive)
  * chunk_type is in the known vocabulary
  * token_count <= budget.max_tokens (the token budget; raises on violation
    in Phase 1 — Phase 2 replaces this with split-then-validate)

`token_counter` is injectable so callers control token accuracy (embeddy
passes a tiktoken/HF-tokenizers counter; tests pass a deterministic one).
The default counter uses tiktoken's cl100k_base lazily and falls back to a
chars/4 estimate when tiktoken cannot load (offline/dev machines).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace

from chonkai.chunkers.base import BaseChunker
from chonkai.models import CHUNK_TYPES, Chunk, ChunkBudget, IngestResult


class ChunkValidationError(ValueError):
    """A chunker output violated a documented invariant."""


def _heuristic_token_counter(text: str) -> int:
    return max(1, len(text) // 4)


def default_token_counter() -> Callable[[str], int]:
    """Best-effort tiktoken counter; deterministic chars/4 fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # ImportError, download failure, cache miss
        warnings.warn(
            f"tiktoken unavailable ({exc!r}); falling back to a "
            "chars/4 token estimate (token budget not token-accurate)",
            stacklevel=2,
        )
        return _heuristic_token_counter
    return lambda text: len(enc.encode(text))


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
        return [self._validate(chunk, effective) for chunk in self._inner.chunk(ingest, effective)]

    def _validate(self, chunk: Chunk, budget: ChunkBudget | None) -> Chunk:
        if not chunk.content.strip():
            raise ChunkValidationError("chunk content must be non-empty")
        if chunk.start_line < 1 or chunk.end_line < chunk.start_line:
            raise ChunkValidationError(f"invalid line range ({chunk.start_line}-{chunk.end_line})")
        if chunk.chunk_type not in CHUNK_TYPES:
            raise ChunkValidationError(
                f"unknown chunk_type {chunk.chunk_type!r}; expected one of {sorted(CHUNK_TYPES)}"
            )
        tokens = self._counter(chunk.content)
        if budget is not None and tokens > budget.max_tokens:
            raise ChunkValidationError(
                f"chunk is {tokens} tokens, exceeding budget "
                f"max_tokens={budget.max_tokens} (Phase 2 adds post-split)"
            )
        return replace(chunk, token_count=tokens)
