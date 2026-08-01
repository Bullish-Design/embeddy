#!/usr/bin/env python3
"""Fresh-venv install matrix check (plan §9 work item 1; §12 M6 gate).

For every row of the extras matrix, create a fresh virtualenv, `pip install`
the local packages (dependency order: chonkai first, then embeddy), and
assert the import surface:

  - zero extras: `import chonkai` / `import embeddy` succeed; the two KNOWN
    module-level extra entry points (`embeddy.server` needs fastapi,
    `embeddy.cli` needs typer) raise ImportError without their extra.
  - each extra row: the install resolves cleanly and the packages import.

Rows (install order matters — embeddy depends on chonkai, which is not on
PyPI yet; installing chonkai first satisfies embeddy's requirement):

    zero-extras     chonkai, embeddy                (both import clean)
    embeddy[server] chonkai, embeddy[server]
    embeddy[client] chonkai, embeddy[client]
    embeddy[local]  chonkai, embeddy[local]
    embeddy[qdrant] chonkai, embeddy[qdrant]
    chonkai[docling]    chonkai[docling]
    chonkai[tokenizers] chonkai[tokenizers]

Usage (needs network for PyPI deps; a real Python 3 >= 3.11):

    python3 scripts/check_install_matrix.py [--repo REPO] [--keep]

Exit 0 = every row green; exit 1 = at least one row failed (row detail on
stderr). `--keep` leaves the venvs behind (under /tmp) for inspection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (row name, packages to install in order, zero_extras_row)
ROWS: list[tuple[str, list[str], bool]] = [
    ("zero-extras", ["chonkai", "embeddy"], True),
    ("embeddy[server]", ["chonkai", "embeddy[server]"], False),
    ("embeddy[client]", ["chonkai", "embeddy[client]"], False),
    ("embeddy[local]", ["chonkai", "embeddy[local]"], False),
    ("embeddy[qdrant]", ["chonkai", "embeddy[qdrant]"], False),
    ("chonkai[docling]", ["chonkai[docling]"], False),
    ("chonkai[tokenizers]", ["chonkai[tokenizers]"], False),
]

# The two KNOWN module-level extra entry points (plan §9 lazy-import audit):
# they import their extra at module level BY DESIGN and must fail cleanly
# (ImportError) when the extra is absent.
MODULE_LEVEL_EXTRA_ENTRY_POINTS = ("embeddy.server", "embeddy.cli")

keep_venvs = False  # set by --keep; read in check_row's finally


def run(cmd: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_zero_extras(python: str) -> list[str]:
    """Assert the zero-extras import contract for a given venv python."""
    failures: list[str] = []

    def _run_py(code: str, what: str) -> None:
        proc = run([python, "-c", code])
        if proc.returncode != 0:
            failures.append(f"{what}: rc={proc.returncode} {proc.stderr.strip()}")

    _run_py("import chonkai; import embeddy", "import chonkai/embeddy (zero extras)")
    # The core must not accidentally pull the extras: importing the entry
    # points without their extra is a hard error (documented contract).
    for mod in MODULE_LEVEL_EXTRA_ENTRY_POINTS:
        proc = run([python, "-c", f"import {mod}"])
        if proc.returncode == 0:
            failures.append(
                f"{mod} imports with ZERO extras — module-level extra entry "
                "point imported despite missing extra"
            )
    # embeddy.client is NOT an entry point: httpx is lazy, so importing the
    # module must succeed with zero extras.
    _run_py("import embeddy.client", "import embeddy.client (httpx is lazy)")
    return failures


def check_row(name: str, packages: list[str], zero_extras: bool, repo: Path) -> list[str]:
    failures: list[str] = []
    print(f"[{name}] installing: {', '.join(packages)}")
    venv_dir = Path(
        tempfile.mkdtemp(prefix=f"embeddy-matrix-{name.replace('[', '').replace(']', '')}-")
    )
    try:
        proc = run([sys.executable, "-m", "venv", str(venv_dir)], timeout=600)
        if proc.returncode != 0:
            return [f"{name}: venv create failed: {proc.stderr.strip()}"]
        python = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"

        # Build local package dirs into installable form: install chonkai from
        # its package dir first (satisfies embeddy's dependency), then embeddy.
        specs: list[str] = []
        for pkg in packages:
            base = pkg.split("[")[0]
            extra = pkg[len(base) :] if "[" in pkg else ""
            specs.append(f"{repo / 'packages' / base}{extra}")

        proc = run([str(pip), "install", "--quiet", "--disable-pip-version-check", *specs])
        if proc.returncode != 0:
            return [f"{name}: pip install failed: {proc.stderr.strip()[-2000:]}"]
        print("  install OK")

        import_expr = (
            "import chonkai, embeddy" if "embeddy" in " ".join(packages) else "import chonkai"
        )
        proc = run([str(python), "-c", import_expr])
        if proc.returncode != 0:
            failures.append(f"{name}: {import_expr} failed: {proc.stderr.strip()}")
        if zero_extras:
            failures.extend(check_zero_extras(str(python)))
        return failures
    finally:
        if not keep_venvs:
            proc = run(["rm", "-rf", str(venv_dir)])
    return failures


def main() -> int:
    global keep_venvs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO), help="repo root (default: script parent)")
    parser.add_argument("--keep", action="store_true", help="keep venvs for inspection")
    parser.add_argument(
        "--rows",
        default=None,
        help="comma-separated subset of rows to run (default: all)",
    )
    args = parser.parse_args()
    keep_venvs = args.keep
    repo = Path(args.repo).resolve()

    rows = ROWS
    if args.rows:
        wanted = set(args.rows.split(","))
        rows = [r for r in ROWS if r[0] in wanted]

    failed = 0
    for name, packages, zero in rows:
        errs = check_row(name, packages, zero, repo)
        if errs:
            failed += 1
            print(f"[{name}] FAIL:")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"[{name}] PASS")
    print("-" * 60)
    if failed:
        print(f"install matrix: {len(rows) - failed}/{len(rows)} rows passed")
        return 1
    print(f"install matrix: all {len(rows)} rows passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
