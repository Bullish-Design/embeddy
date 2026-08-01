# SPIKE RESULT — Tree-sitter per-definition chunking (structure-based)

**Date**: 2026-07-31 · **Scope**: CONCEPT §4.4 revised design, empirical validation
**Toolchain**: tree-sitter-language-pack 1.13.7, tree-sitter 0.26.0, CPython 3.13
**Artifact**: `spikes/treesitter_spike.py` — 7/7 checks pass
**Run**: `LD_LIBRARY_PATH=$(dirname $(find /nix/store -name 'libstdc++.so.6' | head -1)) /tmp/spike-venv/bin/python spikes/treesitter_spike.py`

---

## Verdict

**The `structure`-based design delivers H3 and H5 — with one important
correction: the raw-parser fallback is REQUIRED for decorators, not merely a
churn contingency.** The CONCEPT's claim that "decorators stay attached" in the
`structure` output is wrong for 1.13.7: `StructureItem.decorators` is declared
but never populated (verified for python, javascript, typescript, rust). The
item `.span` also excludes the decorator lines. Everything else in §4.4 held
exactly as revised.

## What was verified (fact-by-fact)

| Claim (CONCEPT §4.4) | Result |
|---|---|
| `ProcessConfig(language=...)` default `chunk_max_size=None` returns ZERO chunks | ✅ Confirmed (python/js/rust all return `[]`) |
| `process()` chunking is size-windowed, not per-definition | ✅ Confirmed: one `def big(n)` (867 B) with `chunk_max_size=512` → 3 windows; a small function collapses into 1 window |
| `symbols_defined`/`context_path`/`node_types`/`has_error_nodes` live on `chunk.metadata`, per-window | ✅ Confirmed (`ChunkContext`); `context_path=['big']` on body windows, `[]` on the header window |
| `structure` = per-definition list with `name/kind/span/body_span/decorators/children/visibility/signature/doc_comment` | ✅ Field set exists; `kind` is a pyo3 enum (`str()` → `'Function'`, `'Method'`, `'Class'`, `'Struct'`, `'Impl'`, `'Module'`, ...) |
| Grammars downloaded at runtime via DownloadManager | ✅ `get_parser('kotlin')` lazily fetched `libtree_sitter_kotlin.so` into the cache dir; unknown language → `DownloadError: Language 'x' not available for download` |
| Broken syntax parses anyway; `metadata.has_error_nodes` flags it | ✅ Confirmed; structure still emits the partial definition — no paragraph fallback needed |

## H3 — decorators / parent / granularity: DELIVERED (with raw-parser assist)

- **Decorators**: `structure.decorators` is **always `[]`** in practice (python,
  js, ts, rust probed). Recovered via the **raw-parser fallback** — two shapes:
  - Python-style `decorated_definition` wrapper → `decorator` children.
  - Rust-style `attribute_item` nodes as leading **siblings** of `struct_item`/
    `mod_item`/`function_item`/... (no wrapper node).
  - Decorator text is prepended to the chunk content and carried on the chunk
    record. `@dataclass`/`@staticmethod`/`#[derive(Debug, Clone)]`/
    `#[cfg(test)]` all attach correctly.
- **Parent**: derived from the `children` tree (there is **no `context_path`
  on `StructureItem`** — it exists only on `chunk.metadata`, per-window).
  Python methods and JS methods get `parent='ClassName'`; Rust `impl` methods
  get `parent='Point'`; `mod tests` functions get `parent='tests'`.
- **Granularity**: `StructureItem.kind` filter works — `function`/`class`/
  `module` selection returns exactly the matching chunk types.
- **Kind → chunk_type nuance**: JS/TS emit `Method`; **Python emits `Function`
  for methods too** (they are simply nested in the class's `children`). The
  spike keeps `chunk_type` kind-honest; chonkai should normalize
  `kind=function ∧ parent is class/struct/impl → chunk_type='method'`. This is
  a Phase-2 decision, not a protocol blocker.

## H5 — oversized single definition: DELIVERED

- Split = re-run `process()` on the definition's **byte-range slice** with
  `chunk_max_size`; the first window carries the header automatically and
  later windows carry `context_path=[name]`.
- Verified: 867-byte `def big(n)` at `chunk_max_size=256` → 5 windows; windows
  tile the span slice with **zero byte loss/duplication** (joined == slice).
- Two operational notes for chonkai:
  1. `chunk_max_size` is a **BYTE budget**, not a token budget. The
     `ValidatedChunker` invariant (tokens ≤ budget) must still be enforced;
     the byte window is the coarse split, tokens are the contract.
  2. The structure `span` can exclude a trailing newline (867-source vs
     866-byte span). Use `span.start_byte:end_byte` slicing, not the raw
     source length.
- Split policy for classes: children (methods) are already separate chunks;
     the class body split applies only to definitions that still exceed budget
     after child extraction.

## Offline grammar cache: DELIVERED

- `available_languages()` = 11 bundled grammars, and **all 10 chonkai target
  languages are among them** (python, javascript, typescript, rust, go, c,
  cpp, java, ruby, bash) → zero-download offline ingest for v1.
- `configure(PackConfig(cache_dir=...))` sets the cache; `downloaded_languages()`
  reflects it; with a fresh empty cache the bundled parsers still load.
- Remaining 296 languages (`manifest_languages()`) fetch lazily at first use;
  a missing cache dir + no network = `DownloadError`. chonkai must pin the
  version and document the cache dir (CONCEPT §9.3 concern stands, but only
  for non-target languages).

## Raw-parser fallback: status

The raw API (`get_parser(language)` → `parser.parse` → node walk) is not just
the churn contingency — it is **already load-bearing** in the spike for
decorators and will remain a thin seam in chonkai. `ProcessConfig(structure=...)`
and the raw parser share the same grammar registry, so both paths work offline
for the 10 target languages.

## Gaps & recommendations

1. **CONCEPT §4.4 edit required**: replace "decorators stay attached" with
   "decorators must be recovered via the raw parser; `structure.decorators` is
   inert in 1.13.7" (see `PRECODE_REPORT.md` for the exact wording change).
2. `visibility`/`signature`/`doc_comment` fields are also never populated —
   derive `signature` from the span's first line (done in the spike); ignore
   the other two (chonkai doesn't need them for v1).
3. `ProcessResult` is a pyo3 class (`r.structure`, `r.chunks`, ...), **not** a
   dict — `r["structure"]` raises `AttributeError`. The `.pyi` stub and the
   dataclass wrappers diverge; the spike uses attribute access (correct).
4. Line numbers everywhere are **0-based**; chonkai's public `start_line`/
   `end_line` should be 1-based (convert at the boundary).
