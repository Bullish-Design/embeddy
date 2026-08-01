"""MRL truncation + L2 re-normalization tests (CONCEPT §5.2, plan §3).

A sliced MRL vector is NOT unit-norm; cosine scoring assumes unit vectors,
so truncation MUST be followed by L2 re-normalization. Ported from the
Phase-0 spike matrix (`spikes/test_protocols.py`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from embeddy import assert_unit_vector, truncate_and_renormalize
from embeddy.protocol.types import normalize_l2


def test_truncation_produces_unit_norm() -> None:
    rng = np.random.default_rng(7)
    vec = normalize_l2(rng.standard_normal(1024).astype(np.float32))
    for dim in (32, 64, 256, 1024):
        cut = truncate_and_renormalize(vec, dim)
        assert cut.shape == (dim,)
        assert_unit_vector(cut)
        assert math.isclose(float(np.linalg.norm(cut)), 1.0, abs_tol=1e-5), (dim,)


def test_truncation_without_renorm_is_not_unit() -> None:
    """Sanity: slicing WITHOUT re-normalization is not unit-norm — proving
    the re-normalization step is load-bearing, not cosmetic. (Verified
    against sentence-transformers 5.6.1 too: `truncate_dim` alone gives
    norm ~0.41 at dim 64.)"""
    rng = np.random.default_rng(11)
    vec = normalize_l2(rng.standard_normal(1024).astype(np.float32))
    raw_slice = vec[:64]
    assert not math.isclose(float(np.linalg.norm(raw_slice)), 1.0, abs_tol=1e-2)


def test_truncation_too_long_raises() -> None:
    vec = normalize_l2(np.ones(64, dtype=np.float32))
    with pytest.raises(ValueError, match="truncate"):
        truncate_and_renormalize(vec, 128)


def test_truncation_preserves_cosine_ranking_consistency() -> None:
    """Coarse-to-fine sanity: on WELL-SEPARATED clusters (the realistic
    retrieval regime) the dim-64 ranking agrees with dim-1024. Random noise
    is not rank-stable under truncation and is not the use case (CONCEPT
    §5.2: smooth quality/cost curve, coarse-to-fine retrieval)."""
    rng = np.random.default_rng(42)
    # 3 centroids with strong low-dim structure, docs = centroid + small noise
    centroids = np.eye(1024)[:3] * 40.0
    centroids[0, 0] = 100.0
    docs: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(3):
        for _ in range(8):
            docs.append(
                normalize_l2((centroids[i] + rng.standard_normal(1024) * 1.0).astype(np.float32))
            )
            labels.append(i)
    query = normalize_l2((centroids[0] * 0.9 + rng.standard_normal(1024)).astype(np.float32))

    def top3(q: np.ndarray, ds: list[np.ndarray]) -> list[int]:
        return sorted(range(len(ds)), key=lambda i: -float(np.dot(q, ds[i])))[:3]

    full = top3(query, docs)
    cut = top3(
        truncate_and_renormalize(query, 64),
        [truncate_and_renormalize(d, 64) for d in docs],
    )
    # Cluster-level agreement (both rankings surface cluster-0 docs):
    assert labels[full[0]] == labels[cut[0]] == 0
    assert {labels[i] for i in full} == {labels[i] for i in cut} == {0}


def test_truncation_noop_at_native_dim() -> None:
    """Truncating to the vector's own length returns the vector unchanged
    (dim == len short-circuits before any slicing)."""
    vec = normalize_l2(np.arange(1.0, 9.0).astype(np.float32))
    out = truncate_and_renormalize(vec, 8)
    assert out.shape == (8,)
    assert np.allclose(out, vec)


def test_truncation_rejects_multidim() -> None:
    with pytest.raises(ValueError, match="1-D"):
        truncate_and_renormalize(np.zeros((2, 2), dtype=np.float32), 2)


def test_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero"):
        truncate_and_renormalize(np.zeros(64, dtype=np.float32), 32)
