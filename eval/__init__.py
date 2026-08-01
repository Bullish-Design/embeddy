"""Retrieval-quality eval harness (CONCEPT §9.6 / plan §7, §12, §9.7).

Ported from the Phase-0 spike (`spikes/eval/` — PRECODE_REPORT deliverable 5):
the M4 MECHANICAL regression gate. Fixed 20-doc "acme" corpus + 8 queries +
qrels, deterministic dim-8 FakeProvider, zero models, fully offline, runs in
the DEFAULT pytest suite (deliberately NOT tagged [slow]).

Scope (honest): the fake-provider gate is MECHANICAL — regression detection +
full determinism, not a measure of absolute retrieval quality. Absolute
quality across real models is gated by the Phase-3 `[slow]`
sentence-transformers integration tests (plan §9.7).

Thresholds: nDCG@10 >= 0.50 / recall@10 >= 0.80, derived from THIS corpus +
provider (baseline ~0.599 / 0.875, random control 0.234 / 0.438). Recompute
mechanism: `uv run python eval/run_eval.py` prints the measured means — when
the corpus or provider legitimately changes, re-derive the gate thresholds in
eval/test_eval.py from the new means (never carry over stale numbers).
"""
