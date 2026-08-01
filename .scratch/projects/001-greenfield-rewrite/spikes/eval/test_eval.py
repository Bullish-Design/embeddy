"""M4 regression gate — deterministic retrieval eval with the FakeProvider.

Run: LD_LIBRARY_PATH="$(dirname $(find /nix/store -name 'libstdc++.so.6' | head -1)):$(dirname $(find /nix/store -name 'libz.so.1' | head -1))" \
     /tmp/spike-venv/bin/python -m pytest eval/test_eval.py -v -o addopts=""

The thresholds encode the CURRENT retrieval quality of the harness stack
(FakeProvider + InMemoryStore + wordhash vectors). A model/chunker/fusion
change that regresses mean nDCG@10 or recall@10 below these fails the gate.
The values are deterministic — recompute with `python eval/run_eval.py` when
the corpus legitimately changes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_types import assert_unit_vector  # noqa: E402
from eval.corpus import CORPUS, QRELS, QUERIES  # noqa: E402
from eval.fake_provider import FakeProvider  # noqa: E402
from eval.runner import evaluate, ndcg_at_k, recall_at_k  # noqa: E402
from eval.store import InMemoryStore  # noqa: E402


@pytest.fixture
def results():
    return asyncio.run(
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


def test_provider_vectors_are_unit_norm() -> None:
    vecs = FakeProvider(CORPUS).encode(["alpha beta gamma"])
    for v in vecs:
        assert_unit_vector(v)
        assert v.shape == (8,)


def test_provider_is_deterministic() -> None:
    p = FakeProvider(CORPUS)
    a = p.encode(["rate limit backoff"])
    b = p.encode(["rate limit backoff"])
    assert (a[0] == b[0]).all()


def test_ndcg_at_k_hand_computed() -> None:
    # rel doc at rank 1 and rank 4: DCG = 1 + 1/log2(5) ; IDCG = 1 + 1/log2(3)
    got = ndcg_at_k(
        ["a", "b", "c", "d", "e"],
        {"a", "d"},
        k=10,
    )
    want = (1.0 + 1.0 / 2.321928) / (1.0 + 1.0 / 1.584962)
    assert got == pytest.approx(want, abs=1e-6)


def test_recall_at_k_hand_computed() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=10) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=10) == 1.0
    assert recall_at_k([], {"a"}, k=10) == 0.0


def test_harness_detects_relevant_docs(results) -> None:
    """The harness has signal: most queries surface a relevant doc in their
    top-10 (a regression to garbage ranking collapses this coverage). q4's
    'rate limit' query is a known brittleness of the dim-8 signed hash
    (opposite-sign collisions; see PRECODE_REPORT §5) — covered by the
    aggregate gate instead."""
    surfaced = sum(1 for per in results["per_query"].values()
                   if per["recall"] > 0.0)
    assert surfaced >= 6, f"only {surfaced}/8 queries surfaced a relevant doc"


def test_m4_regression_gate(results) -> None:
    """The M4 gate thresholds (deterministic for the fixed corpus).

    Baseline measured 2026-07-31: mean nDCG@10 = 0.599, mean recall@10 =
    0.875. Thresholds sit at 0.50/0.80 — the random-provider negative
    control scores 0.234/0.438, so the gate cleanly separates signal from
    noise. The fake provider is a MECHANICAL gate (regression detection,
    determinism); absolute retrieval quality is gated separately by the
    Phase-3 [slow] sentence-transformers integration tests.
    """
    assert results["mean_ndcg"] >= 0.50, results
    assert results["mean_recall"] >= 0.80, results


def test_gate_detects_regression() -> None:
    """Negative control: a provider that returns ORTHOGONAL random vectors
    (no lexical signal) must FAIL the gate — proving the gate can catch
    regressions, not just pass."""
    import numpy as np

    class RandomProvider:
        dimension = 8
        context_length = 4096
        model_name = "fake/random"

        def encode(self, inputs, instruction=None):
            rng = np.random.default_rng(0)
            vecs = []
            for _ in inputs:
                v = rng.standard_normal(8).astype(np.float32)
                vecs.append(v / np.linalg.norm(v))
            return vecs

    results = asyncio.run(
        evaluate(
            RandomProvider(),
            InMemoryStore(),
            collection="acme",
            corpus=CORPUS,
            queries=QUERIES,
            qrels=QRELS,
            k=10,
        )
    )
    assert results["mean_ndcg"] < 0.50  # random provider must FAIL the gate
    assert results["mean_recall"] < 0.80
