"""Eval harness CLI — print per-query and mean nDCG@10 / recall@10.

Run: LSTD=$(find /nix/store -name 'libstdc++.so.6' | head -1); LZ=$(find /nix/store -name 'libz.so.1' | head -1) \
     LD_LIBRARY_PATH="$(dirname "$LSTD"):$(dirname "$LZ")" /tmp/spike-venv/bin/python \
     spikes/eval/run_eval.py
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
    print(f"provider: {FakeProvider.model_name}  corpus: {len(CORPUS)} docs "
          f"queries: {len(QUERIES)}  k=10")
    print(f"{'query':<4} {'nDCG@10':>8} {'recall@10':>9}   top docs")
    print("-" * 72)
    for qid, per in results["per_query"].items():
        top = ", ".join(per["top"][:3])
        print(f"{qid:<4} {per['ndcg']:>8.3f} {per['recall']:>9.3f}   {top}")
    print("-" * 72)
    print(f"MEAN nDCG@10 = {results['mean_ndcg']:.4f}   "
          f"MEAN recall@10 = {results['mean_recall']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
