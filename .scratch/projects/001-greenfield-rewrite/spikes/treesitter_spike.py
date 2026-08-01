"""TREE-SITTER SPIKE — per-definition code chunking from the `structure` list.

Empirical validation of CONCEPT §4.4's revised design on
tree-sitter-language-pack 1.13.7, run under the spike venv:

    LSTD=$(find /nix/store -name 'libstdc++.so.6' | head -1)
    LZ=$(find /nix/store -name 'libz.so.1' | head -1)
    LD_LIBRARY_PATH="$(dirname "$LSTD"):$(dirname "$LZ")" /tmp/spike-venv/bin/python \
        .scratch/projects/001-greenfield-rewrite/spikes/treesitter_spike.py

What this spike proves (and what it does NOT):
  H3  decorators stay attached, parent populated, granularity real.
  H5  an oversized single definition is split via chunk_max_size windowing.
  Offline grammar-cache path (10 target grammars are bundled; unknown
  language raises DownloadError).

FINDING (reported in SPIKE_RESULT.md): the `StructureItem.decorators`,
`.visibility`, `.signature` and `.doc_comment` fields are declared but are
NEVER populated by 1.13.7 (verified for python, javascript, typescript, rust).
The item `.span` also excludes the decorator lines. The spike therefore
recovers decorators with the raw-parser fallback (get_parser + node walk on
`decorated_definition`), which works reliably — this is the mitigation the
CONCEPT already names, and it is now empirically confirmed as REQUIRED for
H3's "decorators attached", not just a churn contingency.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import tree_sitter_language_pack as tslp

# --- language facts ----------------------------------------------------------

TARGET_LANGUAGES = [
    "python", "javascript", "typescript", "rust",
    "go", "c", "cpp", "java", "ruby", "bash",
]

# StructureKind values observed in 1.13.7.
FUNCTION_KINDS = {"function", "method"}
CLASS_KINDS = {"class", "struct", "trait", "interface", "enum", "impl"}
MODULE_KINDS = {"module", "namespace"}
SKIP_KINDS = {"other"}

# Per-definition chunk types (the typed vocabulary the ValidatedChunker
# will enforce in chonkai).
CHUNK_TYPE = {
    "function": "function",
    "method": "method",
    "class": "class",
    "struct": "struct",
    "trait": "trait",
    "interface": "interface",
    "enum": "enum",
    "impl": "impl",
    "module": "module",
}

# --- typed chunk record ------------------------------------------------------


@dataclass(frozen=True)
class CodeChunk:
    """One per-definition chunk. Mirrors the chonkai Chunk fields we need."""

    content: str
    start_line: int            # 0-based, inclusive
    end_line: int              # 0-based, inclusive
    chunk_type: str
    granularity: str           # "function" | "class" | "module"
    parent: str | None         # enclosing definition name, if any
    decorators: tuple[str, ...]
    has_error_nodes: bool
    signature: str | None = None
    part: int = 1              # part index when an oversized definition is split
    parts: int = 1

    @property
    def key(self) -> str:
        return f"{self.chunk_type}:{self.parent or ''}:{self.start_line}"


@dataclass
class _Def:
    """Internal per-definition record derived from a StructureItem."""

    kind: str
    name: str
    span: tslp.Span
    body_span: tslp.Span | None
    children: list["_Def"] = field(default_factory=list)
    parent: str | None = None
    decorators: tuple[str, ...] = ()


# --- decorator recovery (raw-parser fallback) --------------------------------


def _raw_decorators(source: str, language: str) -> dict[int, tuple[str, ...]]:
    """Map definition start-line -> decorator texts, via the raw parser.

    The high-level StructureItem.decorators is never populated in 1.13.7
    (verified for python/js/ts/rust), so decorators are recovered from the
    raw parse tree. Two shapes are handled:

      * Python-style: a `decorated_definition` node wraps `decorator`
        children around the real definition node.
      * Rust-style: `attribute_item` nodes are leading SIBLINGS of the item
        node (`struct_item`, `mod_item`, ...) — there is no wrapper.

    Returns {definition_start_line: ("@app.route(...)", "@auth.require")}.
    """
    out: dict[int, tuple[str, ...]] = {}
    try:
        parser = tslp.get_parser(language)
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return out  # raw parse unavailable -> chunk without decorators
    root = tree.root_node

    rust_items = (
        "struct_item", "enum_item", "function_item", "trait_item",
        "impl_item", "mod_item", "union_item", "type_item", "const_item",
    )

    def walk(node) -> None:
        if node.type == "decorated_definition":
            decos: list[str] = []
            for child in node.children:
                if child.type in ("decorator", "attribute_item"):
                    decos.append(source[child.start_byte:child.end_byte])
            if decos:
                for child in node.children:
                    if child.type in (
                        "function_definition", "class_definition",
                        "type_definition", "method_definition",
                    ):
                        out[child.start_point.row] = tuple(decos)
                        break
        elif node.type in rust_items:
            # Rust attributes sit on the item's PREVIOUS named siblings.
            decos = []
            prev = node.prev_named_sibling
            while prev is not None and prev.type == "attribute_item":
                decos.append(source[prev.start_byte:prev.end_byte])
                prev = prev.prev_named_sibling
            if decos:
                out[node.start_point.row] = tuple(reversed(decos))
        for child in node.children:
            walk(child)

    walk(root)
    return out


# --- structure -> definitions -------------------------------------------------


def _build_defs(source: str, items: Iterable[tslp.StructureItem],
                decorators: dict[int, tuple[str, ...]],
                parent: str | None = None) -> list[_Def]:
    """Flatten the structure tree, keeping nesting for parent resolution."""
    defs: list[_Def] = []
    for it in items:
        # StructureKind is a pyo3 enum: str() -> 'Function' (capitalized),
        # no .value attribute. Normalize to lowercase for the kind sets above.
        kind = str(it.kind).lower()
        name = it.name or f"<anon-{kind}>"
        span = it.span
        d = _Def(
            kind=kind,
            name=name,
            span=span,
            body_span=it.body_span,
            parent=parent,
            decorators=decorators.get(span.start_line, ()),
        )
        d.children = _build_defs(
            source, it.children or [], decorators, parent=name
        )
        defs.append(d)
    return defs


# --- chunk assembly -----------------------------------------------------------


def _slice(source: str, span: tslp.Span) -> str:
    return source[span.start_byte:span.end_byte]


def _granularity(kind: str) -> str | None:
    if kind in FUNCTION_KINDS:
        return "function"
    if kind in CLASS_KINDS:
        return "class"
    if kind in MODULE_KINDS:
        return "module"
    return None


def _split_oversized(source: str, d: _Def, language: str,
                     budget_bytes: int) -> list[tuple[str, int, int]]:
    """Split an oversized single definition via chunk_max_size windowing.

    Re-runs process() on the definition's source slice with a byte budget.
    Returns [(window_text, start_line, end_line), ...]. The first window
    carries the definition header automatically (verified: window 0 is the
    header + first body line, later windows carry context_path=[name]).

    NOTE (reported to SPIKE_RESULT.md): chunk_max_size is a BYTE budget, not a
    token budget. chonkai's ValidatedChunker must still enforce the token
    budget; the byte window is the coarse split, tokens are the contract.
    """
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be > 0")
    slice_text = source[d.span.start_byte:d.span.end_byte]
    sub = tslp.process(
        slice_text,
        tslp.ProcessConfig(language=language, chunk_max_size=budget_bytes),
    )
    if not sub.chunks:
        return [(slice_text, d.span.start_line, d.span.end_line)]
    parts: list[tuple[str, int, int]] = []
    for c in sub.chunks:
        parts.append((
            c.content,
            d.span.start_line + c.start_line,
            d.span.start_line + c.end_line,
        ))
    return parts


def chunk_code(
    source: str,
    language: str,
    *,
    granularity: str = "all",           # "function" | "class" | "module" | "all"
    max_bytes_per_definition: int | None = None,  # None = never split
) -> list[CodeChunk]:
    """Per-definition chunker built on the `structure` list.

    granularity selection = StructureItem.kind filter (implements the dead
    `python_granularity` config from the old codebase).
    """
    decorators = _raw_decorators(source, language)
    result = tslp.process(
        source,
        tslp.ProcessConfig(
            language=language,
            structure=True, symbols=True, docstrings=True,
            chunk_max_size=None,  # we do NOT use the windowed chunk stream
        ),
    )
    if not result.structure:
        return []

    defs = _build_defs(source, result.structure, decorators)
    chunks: list[CodeChunk] = []

    def emit(d: _Def) -> None:
        granularity_of = _granularity(d.kind)
        if granularity_of is None:
            return  # "other" — not a definition we chunk
        if granularity != "all" and granularity_of != granularity:
            return
        header = _slice(source, d.span)
        body = _slice(source, d.body_span) if d.body_span else ""
        decor = tuple(d.decorators)
        chunk_type = CHUNK_TYPE.get(d.kind, d.kind)
        signature = header.splitlines()[0] if header else None
        # Decorators live OUTSIDE the structure span; prepend them so the
        # chunk is self-contained (H3: decorators stay attached).
        if decor:
            header = "\n".join(decor) + "\n" + header

        if max_bytes_per_definition and len(body.encode("utf-8")) > max_bytes_per_definition:
            windows = _split_oversized(
                source, d, language, max_bytes_per_definition
            )
            parts = len(windows)
            for i, (text, s, e) in enumerate(windows, start=1):
                chunks.append(CodeChunk(
                    content=text, start_line=s, end_line=e,
                    chunk_type=chunk_type, granularity=granularity_of,
                    parent=d.parent, decorators=decor,
                    has_error_nodes=False, signature=signature,
                    part=i, parts=parts,
                ))
            return

        chunks.append(CodeChunk(
            content=header, start_line=d.span.start_line, end_line=d.span.end_line,
            chunk_type=chunk_type, granularity=granularity_of,
            parent=d.parent, decorators=decor,
            has_error_nodes=False, signature=signature,
        ))

    for d in defs:
        emit(d)
        for child in d.children:
            emit(child)
    return chunks


def check_error_nodes(source: str, language: str) -> bool:
    """Broken syntax: still parses, metadata.has_error_nodes=True."""
    result = tslp.process(
        source,
        tslp.ProcessConfig(language=language, chunk_max_size=512),
    )
    return bool(result.chunks) and any(
        c.metadata.has_error_nodes for c in result.chunks
    )


# --- corpus --------------------------------------------------------------------

PY_DECORATED = '''\
import os
import sys

@dataclass
class Person:
    """A person with a name and an age."""

    name: str
    age: int

    def greet(self, greeting="Hi"):
        """Return a greeting for this person."""
        return f"{greeting}, {self.name}"

    @staticmethod
    def from_name(name):
        return Person(name, 0)

def top_level(x):
    if x:
        return x + 1
    return x - 1
'''

PY_OVERSIZED = (
    "def big(n):\n"
    + "".join(f"    r{i} = process_{i}(n) * {i}\n" for i in range(30))
    + "    return r29\n"
)

PY_BROKEN = "def broken(:\n    if x\n        return\n    else:\n"

JS_CLASS = """\
class Greeter {
  constructor(name) {
    this.name = name;
  }
  greet(greeting) {
    return greeting + this.name;
  }
  static create(name) {
    return new Greeter(name);
  }
}
function standalone(x) { return x * 2; }
"""

RS_FILE = '''\
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

#[cfg(test)]
mod tests {
    fn helper() {}
}
'''


# --- checks --------------------------------------------------------------------


def check_python_decorated() -> None:
    """H3: decorators attached, class/function granularity, parents."""
    chunks = chunk_code(PY_DECORATED, "python")

    person = next(c for c in chunks if c.chunk_type == "class")
    assert person.decorators == ("@dataclass",), person.decorators
    assert person.granularity == "class" and person.parent is None
    assert person.content.lstrip().startswith("@dataclass"), person.content[:40]
    assert '"""A person with a name and an age."""' in person.content

    # NOTE: Python methods are kind=Function (nested under the class), not
    # kind=Method — only JS/TS emit Method. Extraction still works: each
    # method is its own chunk with parent='Person' (H3: methods extracted,
    # parent populated). Normalizing python-method -> chunk_type='method'
    # is a chonkai-phase decision (kind=function + parent is class/struct).
    greet = next(c for c in chunks
                 if c.signature and c.signature.startswith("def greet"))
    assert greet.parent == "Person"
    assert greet.content.lstrip().startswith("def greet"), greet.content[:40]

    from_name = next(c for c in chunks
                     if c.signature and c.signature.startswith("def from_name"))
    assert from_name.decorators == ("@staticmethod",), from_name.decorators
    assert from_name.content.lstrip().startswith("@staticmethod"), \
        from_name.content[:40]

    top = next(c for c in chunks
               if c.signature and c.signature.startswith("def top_level"))
    assert top.granularity == "function" and top.parent is None

    kinds = sorted({c.chunk_type for c in chunks})
    assert kinds == ["class", "function"], kinds
    print(f"  [ok] python decorated: {len(chunks)} chunks, decorators "
          f"attached, parents populated, granularity selected")


def check_granularity_filter() -> None:
    """Granularity selection = kind filter."""
    chunks = chunk_code(PY_DECORATED, "python", granularity="class")
    assert {c.chunk_type for c in chunks} == {"class"}, chunks
    chunks = chunk_code(PY_DECORATED, "python", granularity="function")
    assert {c.chunk_type for c in chunks} == {"function"}, chunks
    assert any(c.parent == "Person" for c in chunks)  # methods extracted
    chunks = chunk_code(PY_DECORATED, "python", granularity="module")
    assert chunks == [], chunks
    print("  [ok] granularity filter: function/class/module selection works")


def check_oversized_split() -> None:
    """H5: oversized single definition split via chunk_max_size windowing."""
    chunks = chunk_code(PY_OVERSIZED, "python", max_bytes_per_definition=256)
    assert len(chunks) > 1, f"expected a split, got {len(chunks)} chunk(s)"
    first = chunks[0]
    assert first.content.lstrip().startswith("def big(n):"), first.content[:40]
    assert all(c.chunk_type == "function" for c in chunks)
    assert first.parts == len(chunks) and first.part == 1
    total = sum(len(c.content.encode("utf-8")) for c in chunks)
    # The structure span may exclude a trailing newline; compare against the
    # definition slice rather than the raw source.
    span = tslp.process(
        PY_OVERSIZED, tslp.ProcessConfig(language="python", structure=True)
    ).structure[0].span
    slice_text = PY_OVERSIZED[span.start_byte:span.end_byte]
    joined = "".join(c.content for c in chunks)
    assert joined == slice_text, (len(joined), len(slice_text))
    print(f"  [ok] oversized split: 1 definition -> {len(chunks)} parts, "
          f"{total} bytes preserved, header attached to part 1")


def check_javascript() -> None:
    """JS: class with methods (kind=Method), function granularity."""
    chunks = chunk_code(JS_CLASS, "javascript")
    types = {c.chunk_type for c in chunks}
    assert types == {"class", "method", "function"}, types
    greeter = [c for c in chunks if c.chunk_type == "class"][0]
    assert greeter.content.startswith("class Greeter")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert {c.parent for c in methods} == {"Greeter"}, methods
    standalone = [c for c in chunks if c.chunk_type == "function"][0]
    assert standalone.parent is None
    print(f"  [ok] javascript: {len(chunks)} chunks, "
          f"methods={[c.parent for c in methods]}, parent populated")


def check_rust() -> None:
    """Rust: struct + impl(methods) + mod; decorators via raw fallback."""
    chunks = chunk_code(RS_FILE, "rust")
    types = {c.chunk_type for c in chunks}
    assert types == {"struct", "impl", "function", "module"}, types
    point = [c for c in chunks if c.chunk_type == "struct"][0]
    assert point.content.lstrip().startswith("#[derive(Debug, Clone)]"), \
        point.content[:40]
    assert point.decorators == ("#[derive(Debug, Clone)]",), point.decorators
    impls = [c for c in chunks if c.chunk_type == "impl"]
    assert impls and impls[0].parent is None, impls
    methods = [c for c in chunks if c.chunk_type == "function" and c.parent]
    assert "Point" in {m.parent for m in methods}, methods
    assert "tests" in {m.parent for m in methods}, methods  # helper() in mod
    tests_mod = [c for c in chunks if c.chunk_type == "module"][0]
    assert tests_mod.content.lstrip().startswith("#[cfg(test)]"), \
        tests_mod.content[:40]
    print(f"  [ok] rust: {len(chunks)} chunks, struct+impl+module "
          f"granularity, #[derive]/#[cfg] attached via raw fallback")


def check_broken_syntax() -> None:
    """Broken code: parses anyway; metadata.has_error_nodes=True."""
    assert check_error_nodes(PY_BROKEN, "python"), "expected has_error_nodes"
    # And the structure still yields a (partial) definition:
    chunks = chunk_code(PY_BROKEN, "python")
    assert chunks and chunks[0].chunk_type == "function"
    print(f"  [ok] broken syntax: parses with has_error_nodes=True, "
          f"structure still emitted {len(chunks)} chunk(s) — no fallback needed")


def check_grammar_cache() -> None:
    """Offline grammar path: 10 targets bundled; unknown lang -> DownloadError."""
    import tempfile, os

    with tempfile.TemporaryDirectory(prefix="tslp-spike-cache-") as cache:
        tslp.configure(tslp.PackConfig(cache_dir=cache))
        assert tslp.cache_dir() == cache
        assert tslp.downloaded_languages() == []
        missing = [t for t in TARGET_LANGUAGES
                   if t not in tslp.available_languages()]
        assert not missing, f"target grammars not bundled: {missing}"
        parser = tslp.get_parser("python")  # works with an EMPTY cache
        assert parser is not None
        # an unknown language raises DownloadError (the documented runtime fetch)
        try:
            tslp.get_parser("definitely_not_a_language_xyz")
            raise AssertionError("expected DownloadError")
        except tslp.DownloadError:
            pass
    print(f"  [ok] grammar cache: {len(TARGET_LANGUAGES)} targets bundled "
          f"(offline-capable); unknown language -> DownloadError")


def main() -> int:
    print(f"tree-sitter-language-pack {tslp.__version__}; "
          f"languages bundled: {len(tslp.available_languages())}")
    checks = [
        check_python_decorated,
        check_granularity_filter,
        check_oversized_split,
        check_javascript,
        check_rust,
        check_broken_syntax,
        check_grammar_cache,
    ]
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — spike, not library code
            failed += 1
            print(f"  [FAIL] {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'PASS' if failed == 0 else 'FAIL'} — {len(checks) - failed}/"
          f"{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
