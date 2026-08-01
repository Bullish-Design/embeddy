"""SearchFilters -> SQL pre-filter compilation + FTS5 query sanitization.

Plan §6 / §14, CONCEPT §5.4 (§3.4 fixes M3/H1):

  * `compile_filters` turns ALL SearchFilters fields into WHERE clauses that
    apply BEFORE the KNN/FTS truncation — never post-filter over-fetch.
      - target="vec": constraints on the vec0 AUXILIARY columns, which
        sqlite-vec applies DURING the KNN scan (probed 2026-08-01 on
        sqlite-vec 0.1.9: EQUALS / IN / NOT_EQUALS / range comparisons are
        in-scan; LIKE is rejected with "Only one of EQUALS, GREATER_THAN,
        LESS_THAN_OR_EQUAL, LESS_THAN, GREATER_THAN_OR_EQUAL, NOT_EQUALS").
        source_path_prefix therefore compiles to a byte-range constraint
        (`path >= prefix AND path < prefix + U+10FFFF` — every UTF-8 string
        with byte-prefix `prefix` lies in that half-open range; verified
        with a multi-byte/emoji suffix). metadata_match compiles to the
        structured filterable fields (below); an UNKNOWN field raises
        FilterCompileError — the H1 silent no-op is structurally impossible.
      - target="join": plain SQL against the chunks (c) / sources (s) join
        used by the FTS path, where the full match set is filtered before
        ORDER BY/LIMIT (FTS5 ranks every match; the WHERE applies before
        truncation, so recall is full).
  * `sanitize_fts_query` implements the plan §14 policy: quote-wrap by
    default + documented opt-in raw mode. Default mode extracts \\w+ tokens
    (unicode-aware) and ANDs them as double-quoted phrases — FTS5
    metacharacters never reach the engine. raw=True passes the string
    verbatim; the caller owns FTS5 syntax.

vec0 auxiliary columns store the filterable fields denormalized onto each
vector row (NULL is rejected by sqlite-vec — the store coerces None -> "").
The mapping below is the single source of truth for which metadata fields
exist, and is shared with index/sources.py documentation.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from embeddy.index.base import SearchFilters

FilterTarget = Literal["vec", "join"]

# metadata_match field -> SQL column, per target. These are the structured,
# filterable fields the store knows (CONCEPT §5.4 / StoredChunk docstring:
# chonkai's richer metadata is truncated at the protocol boundary to the
# fields embeddy can filter on — anything else raises rather than silently
# not filtering, the H1 fix).
_METADATA_COLUMNS_VEC: dict[str, str] = {
    "chunk_type": "v.chunk_type",
    "parent": "v.parent",
    "granularity": "v.granularity",
    "content_type": "v.content_type",
    "source_path": "v.source_path",
}
_METADATA_COLUMNS_JOIN: dict[str, str] = {
    "chunk_type": "c.chunk_type",
    "parent": "c.parent",
    "granularity": "c.granularity",
    "content_type": "s.content_type",
    "source_path": "s.path",
}


class FilterCompileError(ValueError):
    """A SearchFilters field cannot be compiled to SQL (e.g. an unknown
    metadata_match field). Raised instead of silently dropping the filter —
    the H1 class of silent no-op is impossible."""


def compile_filters(
    filters: SearchFilters,
    *,
    target: FilterTarget,
) -> tuple[str, list[Any]]:
    """Compile `filters` into a WHERE-fragment + params.

    Returns `("", [])` when no filters are set, else `(" AND ...", params)`.
    `target="vec"` emits sqlite-vec in-scan constraints on the vec0 aux
    columns; `target="join"` emits plain SQL on the chunks/sources join.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if filters.content_types:
        placeholders = ",".join("?" for _ in filters.content_types)
        col = "v.content_type" if target == "vec" else "s.content_type"
        clauses.append(f"{col} IN ({placeholders})")
        params.extend(filters.content_types)

    if filters.source_path_prefix is not None:
        prefix = filters.source_path_prefix
        if target == "vec":
            # sqlite-vec rejects LIKE on aux columns; a prefix is a half-open
            # byte range: every UTF-8 string starting with `prefix` satisfies
            # `prefix <= s < prefix + U+10FFFF` (max valid UTF-8 scalar).
            lo, hi = _prefix_range(prefix)
            clauses.append("v.source_path >= ? AND v.source_path < ?")
            params.extend([lo, hi])
        else:
            clauses.append("s.path LIKE ? ESCAPE '\\'")
            params.append(_escape_like_prefix(prefix) + "%")

    if filters.chunk_types:
        placeholders = ",".join("?" for _ in filters.chunk_types)
        col = "v.chunk_type" if target == "vec" else "c.chunk_type"
        clauses.append(f"{col} IN ({placeholders})")
        params.extend(filters.chunk_types)

    if filters.metadata_match:
        columns = _METADATA_COLUMNS_VEC if target == "vec" else _METADATA_COLUMNS_JOIN
        for field, value in filters.metadata_match:
            col = columns.get(field)
            if col is None:
                known = ", ".join(sorted(_METADATA_COLUMNS_VEC))
                raise FilterCompileError(
                    f"unknown metadata_match field {field!r}; metadata_match "
                    f"compiles against the structured filterable fields: {known}"
                )
            clauses.append(f"{col} = ?")
            params.append(value)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _prefix_range(prefix: str) -> tuple[str, str]:
    """Half-open byte range [prefix, prefix + U+10FFFF) covering exactly the
    UTF-8 strings that start with `prefix`. U+10FFFF is the maximum Unicode
    scalar, so any valid UTF-8 continuation sorts below it (probed with a
    multi-byte emoji suffix)."""
    return prefix, prefix + "\U0010ffff"


def _escape_like_prefix(prefix: str) -> str:
    """Escape LIKE wildcards so the prefix is matched literally (ESCAPE '\\')."""
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# Unescape-free default tokenizer: unicode-aware \\w keeps accented Latin and
# CJK words intact; FTS5's own tokenizer (unicode61) applies porter stemming
# inside the quoted phrases (verified in the Phase-0 spike).
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# FTS5 boolean operators are METACHARACTERS in the plan §14 sense: in default
# mode they are dropped rather than emitted as literal tokens — nobody searches
# for the word "OR", and keeping it would silently add a phantom AND term.
_FTS5_KEYWORDS = frozenset({"or", "and", "not", "near"})


def sanitize_fts_query(query: str, *, raw: bool = False) -> str:
    """Plan §14 FTS5 query policy.

    Default (raw=False): extract unicode word tokens and AND them as
    double-quoted phrases — `alpha OR beta` becomes `"alpha" AND "beta"`
    (literal, no metacharacter interpretation; porter stemming still applies
    inside phrases). FTS5 syntax characters (-, *, NEAR, ...) are dropped,
    never passed through. Returns "" for a query with no tokens (the caller
    treats that as "no results").

    raw=True: return `query` verbatim — the caller owns FTS5 syntax and its
    metacharacter consequences. The store still validates the query by
    executing it (malformed raw queries raise StoreError).
    """
    if raw:
        return query
    tokens = [
        token for token in _FTS_TOKEN_RE.findall(query) if token.lower() not in _FTS5_KEYWORDS
    ]
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens)
