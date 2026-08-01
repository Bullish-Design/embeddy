# Versioning policy — chonkai + embeddy release train

## Scheme

Both packages use **Semantic Versioning** with 0.x semantics:
`MAJOR.MINOR.PATCH`.

- `MAJOR` — 0 for the initial development phase (pre-1.0).
- `MINOR` — a feature/behavior release. **Within 0.x, a MINOR bump may
  include breaking changes** (conventional for pre-1.0 software); call them
  out in the CHANGELOG.
- `PATCH` — bug fixes, docs, and non-behavioral changes.

## Release train

chonkai + embeddy ship **together at the same version**, initially. Both
`packages/*/pyproject.toml` carry the same `version`, and both appear in
one CHANGELOG entry. They may decouple once the split proves itself (plan
§8); the decision record will document that change.

## What triggers a release

A release happens when the **M6-style release gate** is green on `main`:

1. **Install matrix green** — `scripts/check_install_matrix.py` passes every
   row (zero extras + all extras) in fresh venvs.
2. **Docs match** — the docs audit passes: `docs/config.md` lists only
   implemented fields (CONCEPT §3.8), `docs/cli.md` matches the CLI, no
   aspirational features documented.
3. **Benchmarks runnable** — `uv run pytest benchmarks/ --no-cov` passes
   (invariant/regression assertions; timing is informational).
4. **Quality gates green** — the full default pytest suite (incl. the eval
   gate), coverage ≥ 90% on core, ty strict, ruff clean, lazy-import audit,
   one-way-dependency grep.
5. **CHANGELOG updated** — the new version's entry is written before
   tagging.

## Process

- Version bumps are driven by the CHANGELOG: add an entry, bump both
  `packages/*/pyproject.toml` versions, tag `v<version>` (both packages),
  and only then publish to PyPI.
- Breaking changes during 0.x require a `docs/decisions/` record (the M4
  protocol freeze rule: any change to `EmbeddingProvider` / `Searchable` /
  `RerankerProvider` / `search_hybrid` / `IngestPipeline` / the chonkai
  public API / the server-client wire protocol needs one).
- The registry (CONCEPT §6, §9.2) is reviewed per release — the model
  landscape moves; model choice is config, not code.
