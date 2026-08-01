# 0002 — Benchmark tooling: pytest-benchmark + psutil in the dev group

- **Date**: 2026-08-01
- **Status**: Accepted (recorded at M6, plan §9 work item 4 / §11)
- **Revisit**: if the benchmark layer ever needs to gate releases on timing

## Context

Phase 7 ships a benchmark harness (plan §9 work item 4): a chonkai
chunk-quality harness and an embeddy search/ingest/resource harness. Plan
§11 names the tools: **pytest-benchmark** (timing) + **psutil** (resource
metrics). The question is where those tools live in the dependency graph —
the plan §11 test-plan table puts benchmarks in a "separate job", but does
not say how the tools are installed for that job.

Two options:

1. **Dev group** (the httpx/fastapi precedent: "an embeddy extra but a
   dev/test dependency"). pytest-benchmark + psutil are regular dev
   dependencies of the workspace.
2. **Keep out of the dev group**. The benchmark CI job installs them ad hoc
   (`uv run --group benchmarks` or a plain pip install), so the dev env
   stays smaller.

## Decision

**Add `pytest-benchmark>=4.0,<5` and `psutil>=5.9,<7` to the root dev
group** (option 1), matching the httpx/fastapi/typer precedent exactly.

Rationale:

- **The default suite is unaffected.** pytest `testpaths` is
  `["tests", "eval", "packages/chonkai/tests", "packages/embeddy/tests"]` —
  `benchmarks/` is NOT collected, verified empirically (464 tests collected,
  zero from `benchmarks/`). The ~10–15s M6 gate does not change.
- **`uv run pytest benchmarks/` works out of the box** for any developer —
  no manual `pip install` (contrast: sentence-transformers and lancedb are
  intentionally NOT in the lockfile because they are heavy; pytest-benchmark
  and psutil are small wheels with no heavy transitive deps).
- **The precedent is established**: fastapi/uvicorn/typer/httpx are all
  "extras" of embeddy AND dev dependencies, for the same reason — the dev
  env must be able to exercise the code paths those extras enable.
- **CI parity**: the benchmark job runs `uv sync` + `uv run pytest
  benchmarks/` — the exact same environment as a developer, so CI can never
  pass because of a tool a developer cannot reproduce.

Counter-consideration (rejected): keeping them out of the dev group would
shrink the dev env marginally, but creates a second install path that can
drift from what developers run, and the size saving is negligible
(pytest-benchmark is pure-Python; psutil ships a wheel).

## Guardrails

- Benchmark tests live ONLY under `benchmarks/` (outside testpaths) — the
  "NEVER the default suite" rule is structural, not a marker convention.
  Any benchmark file added to `tests/` would be collected and must not be.
- Benchmarks assert correctness (invariants, stats, result shape, a
  generous RSS bound) — never wall-clock thresholds (timing is CI-noise and
  would make the optional job flaky). Timing output is informational.
- The CI job is `continue-on-error: true` (plan §9: "optional job"): it
  runs every push so numbers are visible, but a timing regression never
  blocks the M6 gate.

## Consequences

- Dev environment grows by two small packages; `uv.lock` gains
  `pytest-benchmark`, `py-cpuinfo` (pytest-benchmark dep), and `psutil`.
- The benchmark harness is runnable locally with
  `uv run pytest benchmarks/ --no-cov`.
