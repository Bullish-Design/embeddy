"""Tree-sitter code chunker: per-definition chunks from the `structure` list.

Ports the proven Phase-0 spike (`spikes/treesitter_spike.py`, 7/7 checks)
onto chonkai's Chunk model, building on the `structure` list — NOT on
`process()`'s default chunk stream (default ProcessConfig returns 0 chunks;
process() chunking is size-windowed — CONCEPT §4.4).

Load-bearing verified facts (tree-sitter-language-pack 1.13.7, probed
2026-07-31 and re-probed in this repo):
  * StructureItem.decorators / .visibility / .signature / .doc_comment are
    INERT — never populated. Decorators are recovered from the raw parse
    tree (Python `decorated_definition` wrapper; Rust `attribute_item`
    nodes as leading siblings of the definition) and PREPENDED to the chunk
    content. The raw tree-sitter walker is load-bearing, not a fallback.
  * StructureKind is a pyo3 enum (`str()` -> 'Function', no `.value`);
    ProcessResult is attribute-access only (not a dict); all lines/spans
    are 0-based (converted to 1-based at the public boundary); a structure
    span may exclude the trailing newline (slice start_byte:end_byte).
  * Broken syntax still parses; `has_error_nodes` flags it — NO paragraph
    fallback.
  * `chunk_max_size` is a BYTE budget (the ValidatedChunker token invariant
    is the contract). An oversized single definition is split by re-running
    process() on its byte-range slice; the header rides window 1, later
    windows carry parent=[definition name] (mirroring context_path).
  * Granularity selection (function/class/module) is a StructureItem.kind
    filter — implements the dead `python_granularity` config.
  * All 10 target grammars are BUNDLED in the wheel (offline-safe; the
    other 296 manifest languages are lazy and NOT touched in v1).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import tree_sitter_language_pack as tslp

from chonkai.chunkers.base import BaseChunker
from chonkai.chunkers.markdown import MarkdownChunker
from chonkai.ingest.content_types import CODE_CONTENT_TYPES, ContentType
from chonkai.models import Chunk, ChunkBudget, IngestResult

if TYPE_CHECKING:
    from collections.abc import Iterable

# Grammar registry: chonkai ContentType -> tree-sitter-language-pack language.
# The 10 code grammars are bundled in the wheel (offline-safe). `markdown` is
# a two-grammar pair in tree-sitter-land: the block grammar (`markdown`, the
# one we chunk with) plus `markdown_inline` (an injection grammar for inline
# elements — not needed for block chunking; verified: process(structure=True)
# emits no structure items for markdown, so the markdown path walks the raw
# parse tree instead). `markdown` is a MANIFEST grammar: the first get_parser
# call downloads it into the language-pack cache dir (configurable via
# tslp.configure(PackConfig(cache_dir=...))); afterwards it is offline-safe.
CONTENT_TYPE_LANGUAGE: dict[ContentType, str] = {
    ContentType.PYTHON: "python",
    ContentType.JAVASCRIPT: "javascript",
    ContentType.TYPESCRIPT: "typescript",
    ContentType.RUST: "rust",
    ContentType.GO: "go",
    ContentType.C: "c",
    ContentType.CPP: "cpp",
    ContentType.JAVA: "java",
    ContentType.RUBY: "ruby",
    ContentType.BASH: "bash",
    ContentType.MARKDOWN: "markdown",
}

# Companion injection grammar (tree-sitter-markdown is a two-grammar pair).
MARKDOWN_INLINE_LANGUAGE = "markdown_inline"

# Languages chonkai PROMISES for the M2 release (plan §4 acceptance). The
# remaining bundled grammars stay supported (registered + tested) but are NOT
# release-gated: their per-language quirks are fixed opportunistically, not on
# the release path.
RELEASE_REQUIRED_LANGUAGES: frozenset[str] = frozenset({"python", "bash", "markdown"})

# The content types TreesitterChunker can chunk (10 code + markdown).
_TREESITTER_CONTENT_TYPES: frozenset[ContentType] = CODE_CONTENT_TYPES | frozenset(
    {ContentType.MARKDOWN}
)

# StructureKind values observed in 1.13.7 (probed for all 10 targets).
FUNCTION_KINDS = frozenset({"function", "method"})
CLASS_KINDS = frozenset({"class", "struct", "trait", "interface", "enum", "impl"})
MODULE_KINDS = frozenset({"module", "namespace"})

_CHUNK_TYPE_BY_KIND: dict[str, str] = {
    "function": "function",
    "method": "method",
    "class": "class",
    "struct": "struct",
    "enum": "enum",
    "trait": "trait",
    "interface": "interface",
    "impl": "impl",
    "module": "module",
    "namespace": "module",
}

GRANULARITIES = frozenset({"function", "class", "module"})

# language-pack 1.13.7 emits a DEGENERATE span for some structure kinds — a
# C++ Class item covers only the `class` keyword (verified by probe). These
# are repaired from the raw parse tree via the node types below.
_SPAN_REPAIR: dict[tuple[str, str], tuple[str, ...]] = {
    ("cpp", "class"): ("class_specifier",),
}


@dataclass
class _Def:
    """Internal per-definition record derived from a StructureItem."""

    kind: str
    name: str
    span: tslp.Span
    body_span: tslp.Span | None
    children: list[_Def] = field(default_factory=list)
    parent: str | None = None
    parent_kind: str | None = None
    decorators: tuple[str, ...] = ()


class TreesitterChunker(BaseChunker):
    """Per-definition code chunker built on the `structure` list; markdown via
    the raw parse tree (the structure API emits nothing for markdown).

    `granularity`: "all" | "function" | "class" | "module" — a
    StructureItem.kind filter (no markdown meaning; "all" only for markdown).
    `chunk_max_size`: BYTE budget; definitions whose body exceeds it are
    split via process() windowing.
    """

    def __init__(
        self,
        *,
        granularity: str = "all",
        chunk_max_size: int | None = None,
    ) -> None:
        if granularity not in GRANULARITIES and granularity != "all":
            raise ValueError(
                f"granularity must be one of all/function/class/module, got {granularity!r}"
            )
        if chunk_max_size is not None and chunk_max_size <= 0:
            raise ValueError(f"chunk_max_size must be > 0, got {chunk_max_size}")
        self._granularity = granularity
        self._chunk_max_size = chunk_max_size

    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]:
        # the token budget is ValidatedChunker's contract (code path below);
        # the markdown fallback passes the budget through to MarkdownChunker
        source = ingest.text
        ct = ingest.source.content_type
        try:
            content_type = ContentType(ct) if ct is not None else None
        except ValueError:
            content_type = None
        language = CONTENT_TYPE_LANGUAGE.get(content_type) if content_type is not None else None
        if language is None:
            raise ValueError(
                f"TreesitterChunker requires a supported content type "
                f"({sorted(c.value for c in _TREESITTER_CONTENT_TYPES)}), got {ct!r}"
            )

        root = _parse(source, language)
        if content_type is ContentType.MARKDOWN:
            if self._granularity != "all":
                return []  # function/class/module granularity has no markdown meaning
            if root is None:
                # grammar not downloaded and offline: fall back to the
                # pure-Python markdown chunker (same chunk shapes, no download)
                warnings.warn(
                    "markdown tree-sitter grammar unavailable (offline first "
                    "run?); falling back to MarkdownChunker",
                    stacklevel=2,
                )
                return MarkdownChunker().chunk(ingest, budget)
            return _chunk_markdown_tree(source, root, _tree_has_errors(root))

        del budget  # the code path never reads it

        decorators: dict[int, tuple[str, ...]] = {}
        has_error = False
        if root is not None:
            decorators = _collect_decorators(source, root)
            has_error = _tree_has_errors(root)

        result = tslp.process(
            source,
            tslp.ProcessConfig(
                language=language,
                structure=True,
                symbols=True,
                docstrings=True,
                chunk_max_size=None,  # we do NOT use the windowed chunk stream
            ),
        )
        if not result.structure:
            return []

        defs = _build_defs(source, result.structure, decorators)
        chunks: list[Chunk] = []

        def emit(d: _Def) -> None:
            granularity_of = _granularity(d.kind)
            if granularity_of is None:
                return  # "other" — not a definition we chunk
            if self._granularity != "all" and granularity_of != self._granularity:
                return
            decor = d.decorators
            span = (
                _repair_span(source, language, d.kind, d.span, root) if root is not None else d.span
            )
            header = source[span.start_byte : span.end_byte]
            body = source[d.body_span.start_byte : d.body_span.end_byte] if d.body_span else ""
            if decor:
                # decorators live OUTSIDE the structure span; prepend them so
                # the chunk is self-contained (H3: decorators stay attached)
                header = "\n".join(decor) + "\n" + header
            chunk_type = _chunk_type_for(d.kind, d.parent_kind)

            if (
                self._chunk_max_size is not None
                and len(body.encode("utf-8")) > self._chunk_max_size
            ):
                windows = _split_oversized(source, d, language, self._chunk_max_size)
                for i, (text, s, e) in enumerate(windows, start=1):
                    chunks.append(
                        Chunk(
                            content=text,
                            start_line=s + 1,
                            end_line=e + 1,
                            chunk_type=chunk_type,
                            parent=d.parent if i == 1 else d.name,
                            granularity=granularity_of,
                            has_error_nodes=has_error,
                        )
                    )
                return

            chunks.append(
                Chunk(
                    content=header,
                    start_line=span.start_line + 1,
                    end_line=span.end_line + 1,
                    chunk_type=chunk_type,
                    parent=d.parent,
                    granularity=granularity_of,
                    has_error_nodes=has_error,
                )
            )

        def emit_tree(items: Iterable[_Def]) -> None:
            for d in items:
                emit(d)
                emit_tree(d.children)

        emit_tree(defs)
        return chunks


# --- raw parse tree (load-bearing: decorator recovery + error detection) ----


def _parse(source: str, language: str) -> Any | None:
    """Parse `source` with the raw tree-sitter parser; None on any failure.

    The raw API (`get_parser` + node walk) is load-bearing (decorator
    recovery + error-node detection), not a churn contingency — it stays a
    thin seam sharing the grammar registry with the process() path.
    """
    try:
        parser = tslp.get_parser(language)
        tree = parser.parse(source.encode("utf-8"))
        return tree.root_node
    except Exception:
        return None


# Markdown heading levels: ATX (`#`) or setext (underline).
_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s*")
_ATX_TRAILING_HASHES_RE = re.compile(r"\s*#+\s*$")
_MD_HEADING_NODES = frozenset({"atx_heading", "setext_heading"})
_MD_CONTAINER_NODES = frozenset({"document", "section"})
_MD_CODE_NODES = frozenset({"fenced_code_block", "indented_code_block"})


def _heading_level_and_title(source: str, node: Any) -> tuple[int, str]:
    """(level, title) for an atx/setext heading node."""
    text = source[node.start_byte : node.end_byte].rstrip("\n")
    atx = _ATX_HEADING_RE.match(text)
    if atx:
        title = _ATX_TRAILING_HASHES_RE.sub("", text[atx.end() :]).strip()
        return len(atx.group(1)), title
    lines = text.splitlines()
    underline = lines[-1] if lines else ""
    return (1 if "=" in underline else 2), lines[0].strip() if lines else ""


def _chunk_markdown_tree(source: str, root: Any, has_error: bool) -> list[Chunk]:
    """Chunk markdown from the raw parse tree (the structure API emits nothing
    for markdown — verified). Mirrors MarkdownChunker's output shape: heading
    chunks, paragraph chunks and code chunks, all with parent = nearest heading.
    Heading levels drive the stack (same as the code-fence-aware text chunker).
    """
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, title)

    def parent_of() -> str | None:
        return stack[-1][1] if stack else None

    def emit(node: Any, chunk_type: str) -> None:
        content = source[node.start_byte : node.end_byte].rstrip("\n")
        if not content.strip():
            return  # whitespace-only blocks (e.g. a tab-only indented code block)
        chunks.append(
            Chunk(
                content=content,
                start_line=node.start_point.row + 1,
                # the node may include a trailing newline; end_line is the
                # line of the LAST CONTENT character (same convention as the
                # semchunk chunker — no phantom trailing line)
                end_line=node.start_point.row + 1 + content.count("\n"),
                chunk_type=chunk_type,
                parent=parent_of(),
                has_error_nodes=has_error,
            )
        )

    def walk(node: Any) -> None:
        t = node.type
        if t in _MD_HEADING_NODES:
            level, title = _heading_level_and_title(source, node)
            while stack and stack[-1][0] >= level:
                stack.pop()
            emit(node, "heading")
            stack.append((level, title))
            return
        if t in _MD_CONTAINER_NODES:
            for child in node.children:
                walk(child)
            return
        if t in _MD_CODE_NODES:
            emit(node, "code")
            return
        # leaf blocks: paragraph, list, block_quote, table, html_block,
        # thematic_break, ... — one chunk with the full block text
        emit(node, "paragraph")

    walk(root)
    return chunks


def _tree_has_errors(root: Any) -> bool:
    """True when the raw parse tree contains ERROR/MISSING nodes."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in ("ERROR", "MISSING"):
            return True
        stack.extend(node.children)
    return False


def _repair_span(
    source: str,
    language: str,
    kind: str,
    span: tslp.Span,
    root: Any,
) -> tslp.Span:
    """Repair a known-degenerate structure span from the raw parse tree.

    In 1.13.7 a C++ Class structure item spans only the `class` keyword; the
    raw `class_specifier` node spans the whole definition. Returns the
    original span when no repair applies (or none is needed).
    """
    node_types = _SPAN_REPAIR.get((language, kind))
    if node_types is None:
        return span
    stack = [root]
    while stack:
        node = stack.pop()
        if (
            node.type in node_types
            and node.start_point.row == span.start_line
            and node.start_point.column == span.start_column
        ):
            return tslp.Span(
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point.row,
                start_column=node.start_point.column,
                end_line=node.end_point.row,
                end_column=node.end_point.column,
            )
        stack.extend(node.children)
    return span


def _collect_decorators(source: str, root: Any) -> dict[int, tuple[str, ...]]:
    """Map definition start-line -> decorator texts, via the raw parser.

    Two shapes are handled (verified 2026-07-31 / re-probed):
      * Python-style: a `decorated_definition` node wraps `decorator`
        children around the real definition node.
      * Rust-style: `attribute_item` nodes are leading SIBLINGS of the item
        node (`struct_item`, `mod_item`, ...) — there is no wrapper.
    Returns {definition_start_line: ("@app.route(...)", "@auth.require")}.
    """
    out: dict[int, tuple[str, ...]] = {}
    rust_items = (
        "struct_item",
        "enum_item",
        "function_item",
        "trait_item",
        "impl_item",
        "mod_item",
        "union_item",
        "type_item",
        "const_item",
    )

    def walk(node: Any) -> None:
        if node.type == "decorated_definition":
            decos: list[str] = []
            for child in node.children:
                if child.type in ("decorator", "attribute_item"):
                    decos.append(source[child.start_byte : child.end_byte])
            if decos:
                for child in node.children:
                    if child.type in (
                        "function_definition",
                        "class_definition",
                        "type_definition",
                        "method_definition",
                    ):
                        out[child.start_point.row] = tuple(decos)
                        break
        elif node.type in rust_items:
            decos = []
            prev = node.prev_named_sibling
            while prev is not None and prev.type == "attribute_item":
                decos.append(source[prev.start_byte : prev.end_byte])
                prev = prev.prev_named_sibling
            if decos:
                out[node.start_point.row] = tuple(reversed(decos))
        for child in node.children:
            walk(child)

    walk(root)
    return out


# --- structure -> definitions ------------------------------------------------


def _build_defs(
    source: str,
    items: Iterable[Any],
    decorators: dict[int, tuple[str, ...]],
    parent: str | None = None,
    parent_kind: str | None = None,
) -> list[_Def]:
    """Flatten the structure tree, keeping nesting for parent resolution.

    `items` is duck-typed: the language-pack ships two diverging
    StructureItem declarations (_native vs options), so we access only the
    runtime fields (kind/name/span/body_span/children) via Any.
    """
    defs: list[_Def] = []
    for it in items:
        kind = str(it.kind).lower()  # StructureKind: str() -> 'Function'
        name = it.name or f"<anon-{kind}>"
        span = it.span
        if span is None:
            continue
        d = _Def(
            kind=kind,
            name=name,
            span=span,
            body_span=it.body_span,
            parent=parent,
            parent_kind=parent_kind,
            decorators=decorators.get(span.start_line, ()),
        )
        d.children = _build_defs(
            source, it.children or [], decorators, parent=name, parent_kind=kind
        )
        defs.append(d)
    return defs


# --- chunk assembly -----------------------------------------------------------


def _granularity(kind: str) -> str | None:
    if kind in FUNCTION_KINDS:
        return "function"
    if kind in CLASS_KINDS:
        return "class"
    if kind in MODULE_KINDS:
        return "module"
    return None


def _chunk_type_for(kind: str, parent_kind: str | None) -> str:
    """Kind -> chunk_type, with the python-method normalization.

    JS/TS emit kind=Method; Python emits kind=Function for methods too (they
    are just nested under the class). Normalize kind=function with a
    class-like parent to chunk_type='method' (SPIKE_RESULT recommendation).
    """
    if kind == "function" and parent_kind in CLASS_KINDS:
        return "method"
    return _CHUNK_TYPE_BY_KIND.get(kind, kind)


def _split_oversized(
    source: str,
    d: _Def,
    language: str,
    budget_bytes: int,
) -> list[tuple[str, int, int]]:
    """Split an oversized single definition via chunk_max_size windowing.

    Re-runs process() on the definition's source slice with a byte budget.
    Returns [(window_text, start_line, end_line), ...] (0-based lines).
    Verified: the first window carries the header automatically; later
    windows tile the slice with zero byte loss/duplication.
    """
    slice_text = source[d.span.start_byte : d.span.end_byte]
    sub = tslp.process(
        slice_text,
        tslp.ProcessConfig(language=language, chunk_max_size=budget_bytes),
    )
    if not sub.chunks:
        return [(slice_text, d.span.start_line, d.span.end_line)]
    return [
        (
            c.content,
            d.span.start_line + c.start_line,
            d.span.start_line + c.end_line,
        )
        for c in sub.chunks
    ]
