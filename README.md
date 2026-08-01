# chonkai + embeddy

Two libraries in one uv workspace (IMPLEMENTATION_PLAN §2).

| Package | Purpose | Depends on |
|---------|---------|-----------|
| **chonkai** (`packages/chonkai`) | Document processing: ingest + parse + chunk, with guaranteed invariants | — (never imports embeddy) |
| **embeddy** (`packages/embeddy`) | Embed + store + search + serve; typed, metric-honest retrieval | chonkai |

## Layout

```
pyproject.toml            workspace root: members + shared ruff/ty/pytest config
packages/chonkai/         document processing (src/chonkai)
packages/embeddy/         retrieval workflow (src/embeddy)
tests/                    shared integration/e2e tests
benchmarks/               (later phases)
docs/                     (later phases)
.github/workflows/ci.yml  lint / type / test / one-way-dep / import-smoke
```

## Quickstart

```bash
devenv shell          # or: uv sync
uv run pytest         # unit + keystone e2e suite (e2e tagged [e2e])
uv run ty check       # strict type check
uv run ruff check .   # lint
```

## Status — M2 (Phase 1 keystone + Phase 2 chonkai v1)

- `embeddy.protocol` — typed records (`EmbedInput`, `Vector`, `StoredChunk`,
  `ScoredDocument` w/ `.metric`, `CollectionStats`, `SourceMetadata`,
  `SourceId`) and the `EmbeddingProvider` protocol. No `dict[str, Any]`
  crosses any protocol. Drafted at M1, **frozen at M4**.
- `embeddy.registry` — `ModelSpec` + `resolve_dimension` (exact CONCEPT §5.2
  MRL logic) + minimal default registry (Qwen3-0.6B).
- `embeddy.providers.fake` — dev-only deterministic dim-8 `FakeProvider`.
- `embeddy.index` — `Searchable` protocol **including source ops**, and
  `SqliteStore` (aiosqlite + sqlite-vec cosine + FTS5 porter/unicode61 +
  `PRAGMA user_version` stamp). Phase-1 surface: `create_collection`,
  `upsert_source`, `get_source`, `add`, `search_vector`, `search_fts`,
  `stats`.
- `embeddy.search` — `fuse_rrf` (k=60) / `fuse_weighted` pure functions.
- `chonkai` — `IngestResult` / `Chunk` / `SourceMetadata`, `ParagraphChunker`,
  `ValidatedChunker` (non-empty, token budget, line ranges).
- `chonkai.ingest` — content-type detection, encoding-fallback file reading,
  docling bridge (lazy extra), sha256 hashing.
- `chonkai.chunkers` — paragraph (short-merge), markdown (code-fence-aware,
  heading→`parent`), semchunk (token-accurate), tree-sitter (10 bundled code
  grammars + markdown; decorator recovery via the raw parser; granularity
  filter), docling bridge (lazy extra), `get_chunker` factory.
- Tree-sitter language set: **release-required = python / bash / markdown**
  (`RELEASE_REQUIRED_LANGUAGES`); the other 7 bundled grammars (js/ts/rust/go/
  c/cpp/java/ruby) stay supported but are not release-gated. Markdown is a
  two-grammar pair (`markdown` block + `markdown_inline` injection); the block
  grammar is a manifest download on first use (offline-safe after caching,
  cache dir via `tslp.configure(PackConfig(cache_dir=...))`), with a
  pure-Python fallback chunker when it cannot load.
- `chonkai.validated` — invariants on every chunker output + token post-split.
- Type checking: Ty (strict by default) — replaces mypy.

See `.scratch/projects/001-greenfield-rewrite/` for the concept, plan, and
Phase-0 spike evidence.
