"""Chunker factory (CONCEPT §4.7): content type -> chunker instance.

Strategy resolution for `strategy="auto"`:
  * docling-routed types (pdf/docx/html/img/tex) -> DoclingChunker
  * the 10 code types + markdown -> TreesitterChunker
  * rst -> MarkdownChunker (no tree-sitter grammar in the pack)
  * anything else (generic text) -> SemchunkChunker

Explicit strategies ("paragraph" | "markdown" | "semchunk" | "treesitter" |
"docling") bypass content-type routing. `config` keys are validated against
the chosen chunker's constructor parameters; unknown keys raise ValueError.
The returned chunker is unwrapped — wrap it in ValidatedChunker to enforce
the chunk invariants.
"""

from __future__ import annotations

from typing import Any

from chonkai.chunkers.base import BaseChunker
from chonkai.chunkers.docling import DoclingChunker
from chonkai.chunkers.markdown import MarkdownChunker
from chonkai.chunkers.paragraph import ParagraphChunker
from chonkai.chunkers.semchunk import SemchunkChunker
from chonkai.chunkers.treesitter import TreesitterChunker
from chonkai.ingest.content_types import CODE_CONTENT_TYPES, DOCLING_ROUTED, ContentType

_STRATEGIES = frozenset({"auto", "paragraph", "markdown", "semchunk", "treesitter", "docling"})

_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "paragraph": frozenset({"token_counter"}),
    "markdown": frozenset(),
    "semchunk": frozenset({"token_counter", "overlap", "memoize"}),
    "treesitter": frozenset({"granularity", "chunk_max_size"}),
    "docling": frozenset({"tokenizer", "max_tokens", "merge_mode", "overlap"}),
}


def get_chunker(
    content_type: str | None,
    strategy: str = "auto",
    config: dict[str, Any] | None = None,
) -> BaseChunker:
    """Return a chunker for `content_type` (with `strategy`/`config` overrides)."""
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown chunker strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}"
        )
    cfg = config or {}

    if strategy == "auto":
        try:
            ct = ContentType(content_type) if content_type is not None else ContentType.TEXT
        except ValueError:
            # unknown content-type string: route to generic text chunking
            ct = ContentType.TEXT
        if ct in DOCLING_ROUTED:
            target = "docling"
        elif ct in CODE_CONTENT_TYPES or ct is ContentType.MARKDOWN:
            target = "treesitter"
        elif ct is ContentType.RST:
            target = "markdown"
        else:
            target = "semchunk"
    else:
        target = strategy

    unknown = set(cfg) - _CONFIG_KEYS[target]
    if unknown:
        raise ValueError(
            f"unknown config keys {sorted(unknown)} for strategy {target!r}; "
            f"expected one of {sorted(_CONFIG_KEYS[target])}"
        )

    if target == "paragraph":
        return ParagraphChunker(**cfg)
    if target == "markdown":
        return MarkdownChunker(**cfg)
    if target == "semchunk":
        return SemchunkChunker(**cfg)
    if target == "treesitter":
        return TreesitterChunker(**cfg)
    return DoclingChunker(**cfg)
