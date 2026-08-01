#!/usr/bin/env python3
"""Lazy-import gate (plan §9 work item 2; §12 M6 gate).

Extra-only dependencies must be imported ONLY:
  - lazily, inside a function/method (deferred until first use), or
  - at module level in the two KNOWN module-level extra entry points
    (embeddy.server -> fastapi/starlette; embeddy.cli -> typer), or
  - inside an `if TYPE_CHECKING:` block (type-time only, never at runtime).

Anything else breaks the zero-extras `import embeddy` / `import chonkai`
smoke: a module-level `import fastapi` in a core module makes the core
unimportable without the extra (the C4 class of bug).

An AST-based check (not a grep): grep cannot distinguish a module-level
import from a lazy one, and misreads `if TYPE_CHECKING:` blocks. This is a
pure-stdlib script so CI can run it with the bare runner python.

Usage:

    python3 scripts/check_lazy_imports.py [--root REPO]

Exit 0 = audit clean; exit 1 = violations found (details on stdout).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# root module -> the extra that owns it (used only for the message)
EXTRA_ONLY_MODULES: dict[str, str] = {
    # chonkai extras
    "docling": "chonkai[docling]",
    "tokenizers": "chonkai[tokenizers]",
    # embeddy[local]
    "sentence_transformers": "embeddy[local]",
    "torch": "embeddy[local] (transitively, via sentence-transformers)",
    "transformers": "embeddy[local] (transitively, via sentence-transformers)",
    # embeddy[server]
    "fastapi": "embeddy[server]",
    "starlette": "embeddy[server]",
    "uvicorn": "embeddy[server]",
    "typer": "embeddy[server]",
    # embeddy[client]
    "httpx": "embeddy[client]",
    # embeddy[qdrant]
    "qdrant": "embeddy[qdrant]",
    "qdrant_client": "embeddy[qdrant]",
    # spike-only (benchmarks, manually installed — never a library dep)
    "lancedb": "spike-only (benchmarks)",
    "pyarrow": "spike-only (benchmarks)",
}

# (path suffix, root module) — the module-level extra entry points, by design.
# These files REQUIRE their extra: they are the extra's public entry point.
# The set is a deliberate, documented contract (plan §9 lazy-import audit).
# The suffix is matched against the path after the package dir, e.g.
# packages/embeddy/src/embeddy/server.py -> src/embeddy/server.py.
MODULE_LEVEL_ALLOWED: set[tuple[str, str]] = {
    ("src/embeddy/server.py", "fastapi"),
    ("src/embeddy/server.py", "starlette"),
    ("src/embeddy/cli.py", "typer"),
}

SRC_DIRS = ("packages/chonkai/src", "packages/embeddy/src")


def module_key(rel: Path) -> str:
    """Path of the module relative to its package's src/ dir."""
    # rel is like packages/embeddy/src/embeddy/server.py
    parts = rel.parts
    for i, part in enumerate(parts):
        if part == "src":
            return "/".join(parts[i:])
    return str(rel)


def imported_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Root module names this import statement touches ("" for bare relative)."""
    if isinstance(node, ast.ImportFrom):
        if node.level:  # relative import (from . import x) — internal
            return []
        return [node.module.split(".")[0]] if node.module else []
    return [alias.name.split(".")[0] for alias in node.names]


def is_type_checking_block(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )


def check_file(path: Path, rel: Path) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{rel}: SYNTAX ERROR: {exc}"]
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for root in imported_roots(stmt):
                if (
                    root in EXTRA_ONLY_MODULES
                    and (module_key(rel), root) not in MODULE_LEVEL_ALLOWED
                ):
                    violations.append(
                        f"{rel}:{stmt.lineno}: module-level import of '{root}' "
                        f"(extra: {EXTRA_ONLY_MODULES[root]}) — import lazily "
                        "inside a function, or move to the extra's entry-point "
                        "module, or wrap in `if TYPE_CHECKING:`"
                    )
        elif is_type_checking_block(stmt):
            continue  # imports inside TYPE_CHECKING are type-time only — allowed
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # imports nested in a function/class are lazy — allowed
        else:
            # Any other module-body statement containing imports (a plain
            # `if <not-TYPE_CHECKING>`, `try/except`, `with`, ...) executes at
            # import time — treat its imports as module-level (the exact
            # contract: only entry points + TYPE_CHECKING are exempt).
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for root in imported_roots(node):
                        if root in EXTRA_ONLY_MODULES:
                            violations.append(
                                f"{rel}:{node.lineno}: import of '{root}' "
                                f"(extra: {EXTRA_ONLY_MODULES[root]}) inside a "
                                "module-body statement — import lazily inside a "
                                "function instead"
                            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()

    violations: list[str] = []
    checked = 0
    for src_dir in SRC_DIRS:
        base = root / src_dir
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            violations.extend(check_file(path, rel))
            checked += 1

    if violations:
        print(f"lazy-import audit: {len(violations)} violation(s) across {checked} files:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(
        f"lazy-import audit: clean ({checked} files; module-level extra "
        f"imports restricted to {sorted(MODULE_LEVEL_ALLOWED)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
