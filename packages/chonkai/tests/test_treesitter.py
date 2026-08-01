"""Tree-sitter chunker: golden tests for all 10 languages + verified facts.

Exact expected chunks per fixture (computed once, verified by inspection;
regression-locked here). Covers the plan §4 facts: decorators attached via
the raw parser, parent populated from the children tree, granularity via
kind filter, broken syntax flagged (no fallback), 1-based line conversion,
C++ degenerate-span repair.
"""

from __future__ import annotations

import pytest
from conftest import ingest_from_file

from chonkai import IngestResult, TreesitterChunker


def _chunks(source: str, content_type: str | None, **cfg):
    ingest = IngestResult.from_text(source, path="x", content_type=content_type)
    return TreesitterChunker(**cfg).chunk(ingest)


def test_python_decorated(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "python_decorated.py", "python"))
    assert [
        (c.chunk_type, c.start_line, c.end_line, c.parent, c.granularity, c.content) for c in chunks
    ] == [
        (
            "class",
            5,
            17,
            None,
            "class",
            '@dataclass\nclass Person:\n    """A person with a name and an age."""\n\n'
            "    name: str\n    age: int\n\n"
            '    def greet(self, greeting="Hi"):\n'
            '        """Return a greeting for this person."""\n'
            '        return f"{greeting}, {self.name}"\n\n'
            "    @staticmethod\n    def from_name(name):\n"
            "        return Person(name, 0)",
        ),
        (
            "method",
            11,
            13,
            "Person",
            "function",
            'def greet(self, greeting="Hi"):\n'
            '        """Return a greeting for this person."""\n'
            '        return f"{greeting}, {self.name}"',
        ),
        (
            "method",
            16,
            17,
            "Person",
            "function",
            "@staticmethod\ndef from_name(name):\n        return Person(name, 0)",
        ),
        (
            "function",
            20,
            23,
            None,
            "function",
            "def top_level(x):\n    if x:\n        return x + 1\n    return x - 1",
        ),
    ]


def test_python_multiline_decorator() -> None:
    source = '@app.route(\n    "/x",\n    methods=["GET"],\n)\ndef h():\n    return 1\n'
    chunks = _chunks(source, "python")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].content.startswith('@app.route(\n    "/x",\n    methods=["GET"],\n)\ndef h():')


def test_python_method_normalization() -> None:
    # kind=Function nested under a class -> chunk_type 'method'
    source = "class A:\n    def m(self):\n        pass\n\ndef top():\n    pass\n"
    chunks = _chunks(source, "python")
    by_type = {}
    for c in chunks:
        by_type.setdefault(c.chunk_type, []).append(c)
    assert [c.parent for c in by_type["method"]] == ["A"]
    assert by_type["function"][0].parent is None


def test_python_broken_syntax(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "python_broken.py", "python"))
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].has_error_nodes is True  # flagged, no paragraph fallback


def test_javascript(fixtures) -> None:
    chunks = TreesitterChunker().chunk(
        ingest_from_file(fixtures / "javascript_sample.js", "javascript")
    )
    assert [c.chunk_type for c in chunks] == ["class", "method", "method", "method", "function"]
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert {c.parent for c in methods} == {"Greeter"}
    cls = chunks[0]
    assert cls.content.startswith("class Greeter {")
    assert cls.granularity == "class"


def test_typescript(fixtures) -> None:
    chunks = TreesitterChunker().chunk(
        ingest_from_file(fixtures / "typescript_sample.ts", "typescript")
    )
    assert [c.chunk_type for c in chunks] == ["interface", "class", "method", "method", "function"]
    interface = chunks[0]
    assert interface.chunk_type == "interface"
    assert interface.content.startswith("interface Shape {")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert {c.parent for c in methods} == {"Circle"}


def test_rust_decorators_and_parents(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "rust_sample.rs", "rust"))
    assert [c.chunk_type for c in chunks] == [
        "struct",
        "impl",
        "method",
        "method",
        "module",
        "function",
    ]
    struct = chunks[0]
    assert struct.content.startswith("#[derive(Debug, Clone)]\npub struct Point")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert {c.parent for c in methods} == {"Point"}
    mod = [c for c in chunks if c.chunk_type == "module"][0]
    assert mod.content.startswith("#[cfg(test)]\nmod tests")
    helper = [c for c in chunks if c.chunk_type == "function"][0]
    assert helper.parent == "tests"


def test_go(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "go_sample.go", "go"))
    assert [c.chunk_type for c in chunks] == ["function", "method"]
    assert chunks[0].content.startswith("func NewS() *S")
    assert chunks[1].content.startswith("func (s *S) M()")


def test_c(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "c_sample.c", "c"))
    assert [c.chunk_type for c in chunks] == ["function", "function"]
    assert chunks[0].content.startswith("static int helper")
    assert chunks[1].content.startswith("int main")


def test_cpp_span_repair(fixtures) -> None:
    # 1.13.7 emits a degenerate 'class'-keyword span for C++ classes; the raw
    # class_specifier node repairs it (golden regression for the repair).
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "cpp_sample.cpp", "cpp"))
    cls = chunks[0]
    assert cls.chunk_type == "class"
    assert cls.content.startswith("class Greeter {\npublic:")
    assert cls.end_line - cls.start_line + 1 >= 4  # full class body, not 1 line
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert {c.content.splitlines()[0] for c in funcs} == {
        "std::string greet(const std::string& name) {",
        "int clamp(int v, int lo, int hi) {",
    }


def test_java(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "java_sample.java", "java"))
    assert [c.chunk_type for c in chunks] == ["class", "method", "method"]
    assert chunks[0].content.startswith("public class Main {")
    assert {c.parent for c in chunks[1:]} == {"Main"}


def test_ruby(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "ruby_sample.rb", "ruby"))
    assert [c.chunk_type for c in chunks] == ["class", "method", "method", "method"]
    assert chunks[0].content.startswith("class Calculator")
    assert [c.parent for c in chunks[1:]] == ["Calculator", "Calculator", None]


def test_bash(fixtures) -> None:
    chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / "bash_sample.sh", "bash"))
    assert [c.chunk_type for c in chunks] == ["function", "function"]
    assert chunks[0].content.startswith("greet() {")
    assert chunks[1].content.startswith("function add() {")


def test_granularity_filter() -> None:
    source = "class A:\n    def m(self):\n        pass\n\ndef top():\n    pass\n"
    classes = _chunks(source, "python", granularity="class")
    assert [c.chunk_type for c in classes] == ["class"]
    funcs = _chunks(source, "python", granularity="function")
    assert [c.chunk_type for c in funcs] == ["method", "function"]
    mods = _chunks(source, "python", granularity="module")
    assert mods == []


def test_oversized_definition_split() -> None:
    big = (
        "def big(n):\n"
        + "".join(f"    r{i} = process_{i}(n) * {i}\n" for i in range(30))
        + "    return r29\n"
    )
    chunks = _chunks(big, "python", chunk_max_size=256)
    assert len(chunks) > 1
    assert chunks[0].content.startswith("def big(n):")
    assert all(c.chunk_type == "function" for c in chunks)
    # header rides window 1; later windows carry parent=definition name
    assert chunks[0].parent is None
    assert all(c.parent == "big" for c in chunks[1:])
    # windows tile the definition with zero byte loss/dup
    from tree_sitter_language_pack import ProcessConfig, process

    span = process(big, ProcessConfig(language="python", structure=True)).structure[0].span
    slice_text = big[span.start_byte : span.end_byte]
    assert "".join(c.content for c in chunks) == slice_text


def test_python_nested_classes_full_recursion() -> None:
    # methods of nested classes are emitted too (full children recursion)
    source = "class A:\n    class B:\n        def m(self):\n            return 2\n"
    chunks = _chunks(source, "python")
    types = [(c.chunk_type, c.parent) for c in chunks]
    assert ("class", None) in types
    assert ("class", "A") in types
    assert ("method", "B") in types  # kind=Function under class -> method


def test_non_code_content_type_raises() -> None:
    with pytest.raises(ValueError, match="supported content type"):
        _chunks("x", "rst")  # rst has no tree-sitter grammar in the pack
    with pytest.raises(ValueError, match="supported content type"):
        _chunks("x", None)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="granularity"):
        TreesitterChunker(granularity="bogus")
    with pytest.raises(ValueError, match="chunk_max_size"):
        TreesitterChunker(chunk_max_size=0)


def test_file_without_definitions_has_no_chunks() -> None:
    assert _chunks("import os\nX = 1\n", "python") == []
    assert _chunks("", "python") == []


def test_grammar_registry_includes_markdown() -> None:
    from chonkai.chunkers.treesitter import (
        CONTENT_TYPE_LANGUAGE,
        RELEASE_REQUIRED_LANGUAGES,
    )

    assert len(CONTENT_TYPE_LANGUAGE) == 11
    assert set(CONTENT_TYPE_LANGUAGE.values()) == {
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
        "markdown",
    }
    # the M2 release promise: python / bash / markdown. The other grammars
    # stay supported but are not release-gated.
    assert RELEASE_REQUIRED_LANGUAGES == frozenset({"python", "bash", "markdown"})


def test_release_required_languages_chunk_cleanly(fixtures) -> None:
    from chonkai.chunkers.treesitter import RELEASE_REQUIRED_LANGUAGES

    samples = [
        ("python_decorated.py", "python"),
        ("bash_sample.sh", "bash"),
        ("markdown_sample.md", "markdown"),
    ]
    for name, ct in samples:
        chunks = TreesitterChunker().chunk(ingest_from_file(fixtures / name, ct))
        assert chunks, f"{ct} fixture produced no chunks"
        assert all(c.content.strip() for c in chunks)
    assert set(RELEASE_REQUIRED_LANGUAGES) == {"python", "bash", "markdown"}


def test_markdown_golden_via_treesitter(fixtures) -> None:
    """Markdown chunked by the tree-sitter grammar matches the golden fixture
    exactly (headings, paragraphs, code fences, heading-hierarchy parents)."""
    chunks = TreesitterChunker().chunk(
        ingest_from_file(fixtures / "markdown_sample.md", "markdown")
    )
    assert [(c.chunk_type, c.start_line, c.end_line, c.parent, c.content) for c in chunks] == [
        ("heading", 1, 1, None, "# Title"),
        ("paragraph", 3, 3, "Title", "Intro paragraph here."),
        ("heading", 5, 5, "Title", "## Section A"),
        ("paragraph", 7, 7, "Section A", "Some text in section A."),
        (
            "code",
            9,
            13,
            "Section A",
            "```python\n# this looks like a heading but is inside a fence\ndef f():\n    pass\n```",
        ),
        ("paragraph", 15, 15, "Section A", "More text after the code block."),
        ("heading", 17, 17, "Section A", "### Subsection"),
        ("paragraph", 19, 19, "Subsection", "Nested paragraph."),
        ("heading", 21, 21, "Title", "## Section B"),
        ("paragraph", 23, 24, "Section B", "- item one\n- item two"),
    ]


def test_markdown_setext_headings() -> None:
    source = "Title\n======\n\nbody\n\nSub\n---\n\nmore\n"
    chunks = _chunks(source, "markdown")
    headings = [c for c in chunks if c.chunk_type == "heading"]
    assert [c.content for c in headings] == ["Title\n======", "Sub\n---"]
    assert headings[1].parent == "Title"  # setext h2 under setext h1
    body = [c for c in chunks if c.chunk_type == "paragraph"]
    assert [c.parent for c in body] == ["Title", "Sub"]


def test_markdown_falls_back_when_grammar_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    fixtures,
) -> None:
    import chonkai.chunkers.treesitter as ts_module

    monkeypatch.setattr(ts_module, "_parse", lambda source, language: None)
    ingest = ingest_from_file(fixtures / "markdown_sample.md", "markdown")
    with pytest.warns(UserWarning, match="falling back to MarkdownChunker"):
        chunks = TreesitterChunker().chunk(ingest)
    assert chunks
    assert {c.chunk_type for c in chunks} == {"heading", "paragraph", "code"}
