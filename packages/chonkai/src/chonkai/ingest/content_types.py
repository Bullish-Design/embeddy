"""Content-type detection: the v1 vocabulary and extension map (plan §4).

Covers all 10 tree-sitter code types, markdown/rst/generic text, and the
docling-routed rich-document types. Unknown extensions fall back to generic
text (UTF-8/latin-1 decoding is safe for any bytes).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


# The 10 code types are the v1 tree-sitter grammar registry (all bundled in
# the wheel — offline-safe). ContentType is a str-Enum so values compare
# equal to plain strings (SourceMetadata.content_type is a str).
class ContentType(str, Enum):
    """The v1 content-type vocabulary (chonkai-internal, string values)."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    C = "c"
    CPP = "cpp"
    JAVA = "java"
    RUBY = "ruby"
    BASH = "bash"
    MARKDOWN = "markdown"
    RST = "rst"
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    LATEX = "latex"
    IMAGE = "image"


CODE_CONTENT_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.PYTHON,
        ContentType.JAVASCRIPT,
        ContentType.TYPESCRIPT,
        ContentType.RUST,
        ContentType.GO,
        ContentType.C,
        ContentType.CPP,
        ContentType.JAVA,
        ContentType.RUBY,
        ContentType.BASH,
    }
)

# Routed to the docling DocumentConverter (chonkai[docling] extra).
DOCLING_ROUTED: frozenset[ContentType] = frozenset(
    {
        ContentType.PDF,
        ContentType.DOCX,
        ContentType.HTML,
        ContentType.LATEX,
        ContentType.IMAGE,
    }
)

DEFAULT_CONTENT_TYPE = ContentType.TEXT

_EXTENSION_MAP: dict[str, ContentType] = {
    ".py": ContentType.PYTHON,
    ".pyw": ContentType.PYTHON,
    ".js": ContentType.JAVASCRIPT,
    ".mjs": ContentType.JAVASCRIPT,
    ".cjs": ContentType.JAVASCRIPT,
    ".ts": ContentType.TYPESCRIPT,
    ".tsx": ContentType.TYPESCRIPT,
    ".mts": ContentType.TYPESCRIPT,
    ".cts": ContentType.TYPESCRIPT,
    ".rs": ContentType.RUST,
    ".go": ContentType.GO,
    ".c": ContentType.C,
    ".h": ContentType.C,
    ".cpp": ContentType.CPP,
    ".cc": ContentType.CPP,
    ".cxx": ContentType.CPP,
    ".hpp": ContentType.CPP,
    ".hh": ContentType.CPP,
    ".hxx": ContentType.CPP,
    ".java": ContentType.JAVA,
    ".rb": ContentType.RUBY,
    ".sh": ContentType.BASH,
    ".bash": ContentType.BASH,
    ".zsh": ContentType.BASH,
    ".md": ContentType.MARKDOWN,
    ".markdown": ContentType.MARKDOWN,
    ".rst": ContentType.RST,
    ".txt": ContentType.TEXT,
    ".pdf": ContentType.PDF,
    ".docx": ContentType.DOCX,
    ".doc": ContentType.DOCX,
    ".html": ContentType.HTML,
    ".htm": ContentType.HTML,
    ".tex": ContentType.LATEX,
    ".png": ContentType.IMAGE,
    ".jpg": ContentType.IMAGE,
    ".jpeg": ContentType.IMAGE,
    ".gif": ContentType.IMAGE,
    ".webp": ContentType.IMAGE,
    ".tiff": ContentType.IMAGE,
    ".bmp": ContentType.IMAGE,
}


def detect_content_type(path: str | Path) -> ContentType:
    """Detect the content type from a file path's extension."""
    return _EXTENSION_MAP.get(Path(path).suffix.lower(), DEFAULT_CONTENT_TYPE)
