"""Eval harness CLI — print per-query and mean nDCG@10 / recall@10.

RECOMPUTE MECHANISM for the M4 gate thresholds (plan §9.7 / CONCEPT §9.6):
the MEAN values printed here are the measured baseline for THIS corpus +
provider. When the corpus or provider legitimately changes, re-derive the
gate thresholds in eval/test_eval.py from these means — sit clearly below
the measured baseline (and above the random control ~0.234/0.438) — and do
NOT carry over stale numbers.

Run inside `devenv shell`:

    uv run python eval/run_eval.py

(If numpy fails to import in a bare `uv run python` — a known devenv quirk —
prefix the command with the libstdc++/libz LD_LIBRARY_PATH from
PRECODE_REPORT.md; `uv run pytest` is unaffected.)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.corpus import CORPUS, QRELS, QUERIES  # noqa: E402
from eval.fake_provider import FakeProvider  # noqa: E402
from eval.runner import evaluate  # noqa: E402
from eval.store import InMemoryStore  # noqa: E402


def main() -> int:
    results = asyncio.run(
        evaluate(
            FakeProvider(CORPUS),
            InMemoryStore(),
            collection="acme",
            corpus=CORPUS,
            queries=QUERIES,
            qrels=QRELS,
            k=10,
        )
    )
    print(
        f"provider: {FakeProvider.model_name}  corpus: {len(CORPUS)} docs "
        f"queries: {len(QUERIES)}  k=10"
    )
    print(f"{'query':<4} {'nDCG@10':>8} {'recall@10':>9}   top docs")
    print("-" * 72)
    for qid, per in results["per_query"].items():
        top = ", ".join(per["top"][:3])
        print(f"{qid:<4} {per['ndcg']:>8.3f} {per['recall']:>9.3f}   {top}")
    print("-" * 72)
    print(
        f"MEAN nDCG@10 = {results['mean_ndcg']:.4f}   MEAN recall@10 = {results['mean_recall']:.4f}"
    )
    print(
        "(gate: nDCG@10 >= 0.50, recall@10 >= 0.80 — recompute these when "
        "the corpus/provider changes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
