"""Content-type detection (chonkai/ingest/content_types.py)."""

from __future__ import annotations

import pytest

from chonkai.ingest.content_types import (
    CODE_CONTENT_TYPES,
    DEFAULT_CONTENT_TYPE,
    DOCLING_ROUTED,
    ContentType,
    detect_content_type,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.py", ContentType.PYTHON),
        ("a.pyw", ContentType.PYTHON),
        ("a.js", ContentType.JAVASCRIPT),
        ("a.mjs", ContentType.JAVASCRIPT),
        ("a.ts", ContentType.TYPESCRIPT),
        ("a.tsx", ContentType.TYPESCRIPT),
        ("a.rs", ContentType.RUST),
        ("a.go", ContentType.GO),
        ("a.c", ContentType.C),
        ("a.h", ContentType.C),
        ("a.cpp", ContentType.CPP),
        ("a.hpp", ContentType.CPP),
        ("a.java", ContentType.JAVA),
        ("a.rb", ContentType.RUBY),
        ("a.sh", ContentType.BASH),
        ("a.md", ContentType.MARKDOWN),
        ("a.markdown", ContentType.MARKDOWN),
        ("a.rst", ContentType.RST),
        ("a.txt", ContentType.TEXT),
        ("a.pdf", ContentType.PDF),
        ("a.docx", ContentType.DOCX),
        ("a.html", ContentType.HTML),
        ("a.tex", ContentType.LATEX),
        ("a.png", ContentType.IMAGE),
        ("a.jpeg", ContentType.IMAGE),
        ("a.unknown", DEFAULT_CONTENT_TYPE),
    ],
)
def test_detect_content_type(path: str, expected: ContentType) -> None:
    assert detect_content_type(path) == expected


def test_unknown_extension_falls_back_to_text() -> None:
    assert detect_content_type("README") is ContentType.TEXT
    assert detect_content_type("noext") is ContentType.TEXT


def test_extension_match_is_case_insensitive() -> None:
    assert detect_content_type("A.PY") is ContentType.PYTHON
    assert detect_content_type("Doc.MD") is ContentType.MARKDOWN


def test_ten_code_types_cover_the_grammar_registry() -> None:
    assert len(CODE_CONTENT_TYPES) == 10
    expected = {
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "c",
        "cpp",
        "java",
        "ruby",
        "bash",
    }
    assert {c.value for c in CODE_CONTENT_TYPES} == expected


def test_docling_routed_types() -> None:
    assert {c.value for c in DOCLING_ROUTED} == {"pdf", "docx", "html", "latex", "image"}


def test_content_type_is_a_str_enum() -> None:
    # str-Enum values compare equal to plain strings (SourceMetadata stores str)
    assert ContentType.PYTHON == "python"
    assert ContentType("python") is ContentType.PYTHON
