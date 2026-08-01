# 001 — Greenfield Rewrite

Planning workspace for the greenfield rewrite of embeddy.

## Status

Concept phase. All decisions below were researched and validated in-session (2026-07-31).
No code written yet.

## Documents

| File | Contents |
|------|----------|
| `CONCEPT.md` | The full blue-sky rewrite concept: architecture, the chonkai/embeddy split, model registry, protocols, packaging, build order, and open questions |
| `IMPLEMENTATION_PLAN.md` | The detailed implementation plan: phases, work items, acceptance criteria, test strategy, milestones, risk register |
| `KICKOFF_PROMPT.md` | Self-contained prompt for a clean session to perform a deep concept + plan review |

## One-paragraph summary

Split the library in two distributions in one repo:

- **chonkai** — document processing (ingest + parse + chunk). Built wholesale on
  Docling, semchunk, and tree-sitter-language-pack. Enforces chunking invariants
  (max-token guarantee, line ranges, no empty chunks) via a validated wrapper.
- **embeddy** — embed + store + search + serve. Model-agnostic embedding adapter
  (sentence-transformers locally, OpenAI-compatible protocol remotely), typed
  metric-correct storage (sqlite-vec + FTS5 default, Qdrant adapter for scale),
  source-level dedup, hybrid fusion (RRF), optional reranking, async pipeline
  with real concurrency, FastAPI server with proper lifecycle.

Key positions: embedding protocol as keystone, one wire protocol
(OpenAI-compatible), sources as first-class entities, MRL-capable-but-not-required
model registry, config surface equals implemented surface, packaging as
architecture (extras, lazy imports).
