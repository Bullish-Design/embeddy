# Retrieval-quality eval gate

The deterministic, offline retrieval eval that protects retrieval quality
from regressions. It runs in the **default pytest suite** (`eval/test_eval.py`
— no `[slow]` tag; it is fast and offline by design) and is the documented
**pre-release gate**.

## What it is

A fixed-corpus harness (CONCEPT §9.6): 20 fictional "acme" documents, 8
queries, qrels, embedded by a deterministic `FakeProvider` (sha256-seeded
signed-hash TF-IDF, dim 8) and retrieved through an in-memory store. It
reports mean **nDCG@10** and **recall@10** across any model/chunker/fusion
change.

The gate is **mechanical**: regression detection + full determinism, not a
measure of absolute retrieval quality. Absolute quality across real models is
gated by the Phase-3 `[slow]` sentence-transformers integration tests.

## The thresholds

| Metric | Gate | Measured baseline | Random control |
|--------|------|-------------------|----------------|
| mean nDCG@10 | **≥ 0.50** | 0.599 | 0.234 |
| mean recall@10 | **≥ 0.80** | 0.875 | 0.438 |

The thresholds are **derived from the fixed corpus** — they sit clearly below
the measured baseline and above the random-provider negative control, so the
gate separates signal from noise. The corpus and thresholds are frozen; do
not re-tune them.

## How to recompute

```bash
uv run python eval/run_eval.py
```

Prints per-query and mean nDCG@10 / recall@10. This is the recompute
mechanism when the corpus or provider LEGITIMATELY changes: re-derive the
gate thresholds from the new means (sit below baseline, above random
control), and do NOT carry over stale numbers.

## What a regression means

A failing gate (below 0.50 / 0.80) means **retrieval got worse on the fixed
corpus** — e.g. a fusion/ranking change, a chunker change that destroyed
content boundaries, or a provider change. The negative-control tests in
`eval/test_eval.py` prove the gate can catch regressions, not just pass:

- an orthogonal random provider must FAIL the gate;
- reversing the top-k ranking (a broken ordering/fusion change) must FAIL
  the gate;
- two full runs must produce IDENTICAL numbers (determinism is part of the
  contract).

If a legitimate change (corpus, provider, query set) is intended, recompute
and re-derive the thresholds via `eval/run_eval.py` — deliberately, with the
evidence in hand — never by nudging the numbers.

## Files

| Path | Role |
|------|------|
| `eval/corpus.py` | the fixed corpus + queries + qrels |
| `eval/fake_provider.py` | deterministic dim-8 provider (dev/test only) |
| `eval/store.py` | in-memory `Searchable` |
| `eval/runner.py` | `evaluate()` — search, score, aggregate |
| `eval/run_eval.py` | recompute CLI (prints means) |
| `eval/test_eval.py` | the gate tests (default suite) |
