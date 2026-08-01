# chonkai + embeddy

Two libraries in one uv workspace (IMPLEMENTATION_PLAN §2).

| Package | Purpose | Depends on |
|---------|---------|-----------|
| **chonkai** (`packages/chonkai`) | Document processing: ingest + parse + chunk, with guaranteed invariants | — (never imports embeddy) |
| **embeddy** (`packages/embeddy`) | Embed + store + search + serve; typed, metric-honest retrieval | chonkai |

## Layout

```
pyproject.toml            workspace root: members + shared ruff/mypy/pytest config
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
uv run pytest         # keystone e2e suite (tagged [e2e])
uv run mypy           # strict
uv run ruff check .   # lint
```

## Status — M1 (Phase 1 keystone slice)

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

See `.scratch/projects/001-greenfield-rewrite/` for the concept, plan, and
Phase-0 spike evidence.
