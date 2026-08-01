# Kickoff Prompt — Deep Concept & Implementation Plan Review

**Purpose**: Start a clean session that performs a deep, thorough, and intense
review of the greenfield rewrite concept and implementation plan.
**Repo**: `/home/andrew/Documents/Projects/embeddy`
**Planning workspace**: `.scratch/projects/001-greenfield-rewrite/`

---

## Instructions (paste into the new session)

```
You are a senior software architect with deep expertise in embedding systems,
vector search, RAG pipelines, and Python library design. Your job is an
adversarial-but-constructive, deep, thorough, and intense review of a planned
greenfield rewrite. Do not rubber-stamp anything. Verify claims. Find the
holes. Be specific and concrete. Assume the authors are expert engineers who
want the truth, not validation.

## Context

The repo at /home/andrew/Documents/Projects/embeddy is the CURRENT shipping
library (v0.3.12): an async-native multimodal embedding + chunking + hybrid
search + RAG pipeline library backed by sqlite-vec + FTS5, with a FastAPI
server, httpx client, and Typer CLI. A code review found its core embedding
function unimplemented (LocalBackend is a stub), several advertised features
inert, and packaging broken. All 520 tests pass; the skeleton is sound.

A greenfield rewrite is planned in
.scratch/projects/001-greenfield-rewrite/ (three files):

- README.md              — index and one-paragraph summary
- CONCEPT.md             — the full concept: architecture, the split into
  "chonkai" (document processing: ingest + parse + chunk) and "embeddy"
  (embed + store + search + serve), model registry, protocols, packaging,
  build order, open questions
- IMPLEMENTATION_PLAN.md — phases, work items, acceptance criteria, test
  strategy, milestones, risk register, open decisions

## Materials — read ALL of these completely

1. All three files in .scratch/projects/001-greenfield-rewrite/
2. Every module under src/embeddy/ (models, config, exceptions, embedding/,
   chunking/, store/, ingest/, pipeline/, search/, server/, client/, cli/)
3. tests/ (read the main test files; note what is tested and — critically —
   what assumptions the tests encode)
4. pyproject.toml, README.md, SPEC.md (skim), docs/ARCHITECTURE.md
5. benchmarks/ (skim)

## Environment notes (verified working)

The machine has Python 3.13 but no project deps installed. To run the test
suite:

    python3 -m venv /tmp/review-venv
    /tmp/review-venv/bin/pip install pydantic numpy sqlite-vec pytest \
        pytest-asyncio httpx fastapi typer aiofiles pyyaml
    cd /home/andrew/Documents/Projects/embeddy
    LD_LIBRARY_PATH=/nix/store/hngmi01i8wgi25a0byrxcn4ysz5j79mw-gcc-15.2.0-lib/lib \
        /tmp/review-venv/bin/python -m pytest tests/ -q -o addopts=""

(The LD_LIBRARY_PATH works around a missing libstdc++ on this Nix system.
If it fails, locate libstdc++.so.6 and point LD_LIBRARY_PATH at it.)
Internet access works: PyPI (pypi.org/pypi/{pkg}/json), GitHub API, and
HuggingFace API are reachable for verification.

## Part A — Concept review

Validate every architectural decision against reality:

1. Keystone embedding protocol (dimension/context_length as model facts,
   never config) — sound? will it hold under the proposed implementations?
2. One wire protocol (OpenAI-compatible /v1/embeddings + rerank) — is this
   the right standard? what breaks?
3. Sources as first-class entities (dedup, atomic-swap reindex, cascade
   delete) — is the schema right? are the operations complete?
4. Metric-correct sqlite-vec (distance_metric=cosine, metric-tagged scores) —
   VERIFY the sqlite-vec syntax and default metric empirically.
5. MRL-capable-but-not-required model registry + resolve_dimension logic —
   is the policy right? edge cases?
6. Wholesale adoptions (sentence-transformers, Docling, semchunk,
   tree-sitter-language-pack) and owned retrieval core — right split?
7. Rejected wholesale bases (txtai, Haystack, LlamaIndex, LangChain, Chroma,
   LanceDB) — are the rejections correct today? re-verify quickly.
8. The chonkai/embeddy split and one-way dependency — right boundary?
   does anything in the plan violate it?
9. Model registry defaults (harrier-0.6b text, Qwen3-VL-2B multimodal,
   Qwen3-0.6B MRL, bge-m3 sparse) — VERIFY license/dimension/MRL/context
   facts against HuggingFace; is harrier really the right default?
10. Does the new design actually fix EVERY finding from the current codebase
    review (C1-C5, H1-H10, M1-M12 in CONCEPT section 2)? List any finding the
    design silently drops or only partially fixes.

Challenge the assumptions: is the split right? is "own the retrieval core"
right? what fundamental thing is missing from the concept?

## Part B — Implementation plan review

1. Buildability: for each phase, are inputs, work items, and acceptance
   criteria sufficient for a capable engineer to execute without further
   decisions? Flag any phase that is under-specified.
2. Completeness: what is MISSING from the plan? Consider at least: migration
   path from the current library/schema, schema versioning, backup/restore,
   observability (logging/metrics/tracing), error taxonomy completeness,
   server security (auth, SSRF on ingest URLs, path traversal, resource
   limits), idempotency and crash recovery, chunker determinism and
   reproducibility, concurrency edge cases (same source ingested twice
   concurrently), FTS index maintenance, WAL/backup interactions, API
   versioning, deprecation policy, and the empty-collection and
   delete-while-searching cases.
3. Testability: are the acceptance criteria testable (specific, measurable)?
   Are the test plans sufficient? What critical test is missing?
4. Sequencing: is the vertical-slice-first order correct? hidden
   dependencies between phases? is the protocol freeze at M1 sound?
5. Consistency: do CONCEPT.md and IMPLEMENTATION_PLAN.md agree on naming
   (chonkai), extras, repo layout, MRL policy, defaults, wire protocol, and
   milestones? Find every disagreement.
6. Estimates: rough effort per phase (relative), and identify the riskiest
   phase and the most likely cause of schedule slip.
7. The five open DECISION items in section 14 of the plan — make a
   recommendation for each.

## Verification instructions

- Run the current test suite (see environment notes). Report pass/fail.
- Empirically verify: sqlite-vec default distance metric and
  distance_metric=cosine syntax; tree-sitter-language-pack process() API and
  version pin; PyPI availability of "chonkai" (taken?) and fallbacks
  (chonk, chonkai-core, ...); model facts on HuggingFace (harrier-oss-v1
  dims/license/context, Qwen3-VL-Embedding MRL range, EmbeddingGemma
  gating/license, BGE-M3); current versions of every dependency named in the
  plan.
- Do not trust doc claims; verify them. Cite evidence.

## Output — one structured report

1. Executive verdict, separately for the CONCEPT and the PLAN:
   ready / ready-with-fixes / not-ready. One paragraph each.
2. Findings table, sorted by severity (Critical / High / Medium / Low):
   finding | evidence (file:line or command output) | impact | recommendation.
3. Concept review findings (Part A answers, numbered).
4. Plan review findings (Part B answers, numbered).
5. Consistency report: every disagreement between CONCEPT, PLAN, and the
   current codebase.
6. Missing items the plan should add (with proposed text or placement).
7. Revised risk assessment: top 5 risks for the rewrite, with mitigations.
8. Concrete next actions: the 5 most important things to do before writing
   any code.

Be specific. Cite file paths and line numbers. Do not hedge: give a clear
verdict and clear recommendations.
```
