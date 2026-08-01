"""Unit tests for index/filters.py — SearchFilters -> SQL pre-filter
compilation (both targets) + FTS5 query sanitization policy (plan §6/§14).

These are pure-logic tests (no database): they assert the SQL shape and the
parameter binding order, and the documented metacharacter policy. The
end-to-end behavior on a real sqlite-vec/FTS5 store lives in
tests/test_store_phase4.py ([integration]).
"""

from __future__ import annotations

import pytest

from embeddy.index.base import SearchFilters
from embeddy.index.filters import FilterCompileError, compile_filters, sanitize_fts_query


def test_empty_filters_compile_to_noop() -> None:
    for target in ("vec", "join"):
        where, params = compile_filters(SearchFilters(), target=target)  # type: ignore[arg-type]
        assert where == ""
        assert params == []


def test_content_types_target_vec() -> None:
    where, params = compile_filters(SearchFilters(content_types=("markdown", "pdf")), target="vec")
    assert where == " AND v.content_type IN (?,?)"
    assert params == ["markdown", "pdf"]


def test_content_types_target_join() -> None:
    where, params = compile_filters(SearchFilters(content_types=("python",)), target="join")
    assert where == " AND s.content_type IN (?)"
    assert params == ["python"]


def test_chunk_types_both_targets() -> None:
    where, params = compile_filters(
        SearchFilters(chunk_types=("heading", "paragraph")), target="vec"
    )
    assert "v.chunk_type IN (?,?)" in where
    assert params == ["heading", "paragraph"]
    where_j, params_j = compile_filters(SearchFilters(chunk_types=("function",)), target="join")
    assert "c.chunk_type IN (?)" in where_j


def test_source_path_prefix_vec_uses_byte_range() -> None:
    where, params = compile_filters(SearchFilters(source_path_prefix="docs/guide"), target="vec")
    assert "v.source_path >= ? AND v.source_path < ?" in where
    assert params == ["docs/guide", "docs/guide\U0010ffff"]


def test_source_path_prefix_vec_range_excludes_nonprefix() -> None:
    """The half-open byte range must not swallow strings that merely share a
    character prefix (the 'docs-' vs 'docs/' trap)."""
    where, params = compile_filters(SearchFilters(source_path_prefix="docs/"), target="vec")
    lo, hi = params
    assert lo == "docs/"
    # 'docs-other' sorts below 'docs/'; 'docs/a' sorts inside the range.
    assert "docs-other" < lo < hi
    assert "docs/a.md" >= lo and "docs/a.md" < hi
    # any valid UTF-8 continuation (multi-byte) must stay below the bound.
    assert "docs/\U00010000.md" < hi


def test_source_path_prefix_join_escapes_like() -> None:
    where, params = compile_filters(SearchFilters(source_path_prefix="docs/100%_x"), target="join")
    assert "s.path LIKE ? ESCAPE" in where
    assert params == ["docs/100\\%\\_x%"]


def test_metadata_match_vec_maps_structured_fields() -> None:
    where, params = compile_filters(
        SearchFilters(
            metadata_match=(
                ("parent", "intro"),
                ("granularity", "function"),
            )
        ),
        target="vec",
    )
    assert "v.parent = ?" in where
    assert "v.granularity = ?" in where
    assert params == ["intro", "function"]


def test_metadata_match_join_maps_to_chunks_sources() -> None:
    where, params = compile_filters(
        SearchFilters(metadata_match=(("content_type", "markdown"), ("source_path", "docs/a"))),
        target="join",
    )
    assert "s.content_type = ?" in where
    assert "s.path = ?" in where
    assert params == ["markdown", "docs/a"]


def test_metadata_match_unknown_field_raises_not_silent_noop() -> None:
    """H1 regression: an unfilterable metadata field is an explicit error,
    never a silently-ignored filter."""
    for target in ("vec", "join"):
        with pytest.raises(FilterCompileError, match="unknown metadata_match field"):
            compile_filters(
                SearchFilters(metadata_match=(("symbols_defined", "foo"),)),
                target=target,  # type: ignore[arg-type]
            )


def test_compile_filters_all_fields_combined() -> None:
    filters = SearchFilters(
        content_types=("md",),
        source_path_prefix="docs/",
        chunk_types=("paragraph",),
        metadata_match=(("parent", "intro"),),
    )
    where, params = compile_filters(filters, target="vec")
    assert where.startswith(" AND ")
    assert len(params) == 1 + 2 + 1 + 1


# --------------------------------------------------------------------------- #
# sanitize_fts_query — plan §14 policy
# --------------------------------------------------------------------------- #


def test_sanitize_default_wraps_and_and_joins() -> None:
    assert sanitize_fts_query("alpha beta") == '"alpha" AND "beta"'
    assert sanitize_fts_query("tokens") == '"tokens"'


def test_sanitize_default_strips_metacharacters() -> None:
    # FTS5 syntax characters never reach the engine; tokens are literal.
    assert sanitize_fts_query("alpha OR beta") == '"alpha" AND "beta"'
    assert sanitize_fts_query("-alpha") == '"alpha"'
    assert sanitize_fts_query("alpha*") == '"alpha"'
    assert sanitize_fts_query("alpha NEAR/10 beta") == '"alpha" AND "10" AND "beta"'
    assert sanitize_fts_query('"quoted phrase"') == '"quoted" AND "phrase"'


def test_sanitize_default_empty_and_punctuation_only() -> None:
    assert sanitize_fts_query("") == ""
    assert sanitize_fts_query("   ") == ""
    assert sanitize_fts_query("( ) !") == ""


def test_sanitize_default_keeps_unicode_word_chars() -> None:
    # unicode-aware \\w keeps accented words intact (unicode61 tokenizes them
    # as single tokens); porter stemming applies inside the quoted phrase.
    assert sanitize_fts_query("café") == '"café"'


def test_sanitize_raw_passthrough() -> None:
    assert sanitize_fts_query("alpha OR beta", raw=True) == "alpha OR beta"
    assert sanitize_fts_query('"exact phrase"', raw=True) == '"exact phrase"'
    assert sanitize_fts_query("prefix*", raw=True) == "prefix*"


def test_sanitize_raw_empty_passthrough() -> None:
    assert sanitize_fts_query("", raw=True) == ""
