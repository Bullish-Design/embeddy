"""resolve_dimension / MRL policy test matrix (CONCEPT §5.2).

Run: LD_LIBRARY_PATH="$(dirname $(find /nix/store -name 'libstdc++.so.6' | head -1)):$(dirname $(find /nix/store -name 'libz.so.1' | head -1))" \
     /tmp/spike-venv/bin/python -m pytest spikes/test_protocols.py -v

This matrix becomes the Phase-3 `resolve_dimension` unit test (plan §5).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from protocols import (
    ModelSpec,
    RegistryError,
    _Registry,
    normalize,
    resolve_dimension,
    truncate_and_renormalize,
)

MRL_SPEC = ModelSpec(
    id="Qwen/Qwen3-Embedding-0.6B",
    native_dimension=1024,
    mrl_range=range(32, 1025),
    context_length=32768,
    instructions={"query": "query_prompt", "document": "document_prompt"},
    license="Apache-2.0",
)

NON_MRL_SPEC = ModelSpec(
    id="microsoft/harrier-oss-v1-0.6b",
    native_dimension=1024,
    mrl_range=None,
    context_length=32768,
    instructions={"query": "query_prompt", "document": "document_prompt"},
    license="MIT",
)


# --- None -> native ------------------------------------------------------------


@pytest.mark.parametrize("spec", [MRL_SPEC, NON_MRL_SPEC])
def test_none_returns_native(spec: ModelSpec) -> None:
    assert resolve_dimension(spec, None) == spec.native_dimension


# --- MRL model ----------------------------------------------------------------


@pytest.mark.parametrize("requested,expected", [
    (32, 32),            # range start
    (64, 64),            # common MRL slice
    (256, 256),
    (1024, 1024),        # native (range end)
])
def test_mrl_in_range(requested: int, expected: int) -> None:
    assert resolve_dimension(MRL_SPEC, requested) == expected


@pytest.mark.parametrize("requested", [31, 1025, 0, -8])
def test_mrl_out_of_range_raises(requested: int) -> None:
    with pytest.raises(ValueError, match="MRL"):
        resolve_dimension(MRL_SPEC, requested)


# --- non-MRL model -------------------------------------------------------------


def test_non_mrl_native_dim_ok() -> None:
    assert resolve_dimension(NON_MRL_SPEC, 1024) == 1024


@pytest.mark.parametrize("requested", [32, 256, 2048])
def test_non_mrl_wrong_dim_raises(requested: int) -> None:
    with pytest.raises(ValueError, match="not MRL-capable"):
        resolve_dimension(NON_MRL_SPEC, requested)


# --- registry / role -> instruction rule --------------------------------------


def test_role_resolution_via_registry() -> None:
    reg = _Registry()
    reg.register(MRL_SPEC)
    assert reg.resolve_instruction(MRL_SPEC.id, "query") == "query_prompt"
    assert reg.resolve_instruction(MRL_SPEC.id, "document") == "document_prompt"


def test_unknown_model_raises() -> None:
    with pytest.raises(RegistryError):
        _Registry().get("no/such-model")


def test_unknown_role_raises() -> None:
    reg = _Registry()
    reg.register(MRL_SPEC)
    with pytest.raises(RegistryError):
        reg.resolve_instruction(MRL_SPEC.id, "summary")  # no such role


def test_spec_invalid_mrl_range_rejected() -> None:
    with pytest.raises(ValueError):
        ModelSpec(
            id="bad", native_dimension=1024, mrl_range=range(32, 1024),
            context_length=100, instructions={},
        )  # native 1024 is NOT in range(32, 1024) (stop exclusive)


# --- MRL truncation + L2 re-normalization -------------------------------------


def test_truncation_produces_unit_norm() -> None:
    rng = np.random.default_rng(7)
    vec = normalize(rng.standard_normal(1024).astype(np.float32))
    for dim in (32, 64, 256, 1024):
        cut = truncate_and_renormalize(vec, dim)
        assert cut.shape == (dim,)
        norm = float(np.linalg.norm(cut))
        assert math.isclose(norm, 1.0, abs_tol=1e-5), (dim, norm)


def test_truncation_without_renorm_is_not_unit() -> None:
    """Sanity: slicing WITHOUT re-normalization is not unit-norm — proving
    the re-normalization step is load-bearing, not cosmetic."""
    rng = np.random.default_rng(11)
    vec = normalize(rng.standard_normal(1024).astype(np.float32))
    raw_slice = vec[:64]
    assert not math.isclose(float(np.linalg.norm(raw_slice)), 1.0, abs_tol=1e-2)


def test_truncation_too_long_raises() -> None:
    vec = normalize(np.ones(64, dtype=np.float32))
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
    docs = []
    labels = []
    for i in range(3):
        for _ in range(8):
            docs.append(normalize(
                (centroids[i] + rng.standard_normal(1024) * 1.0).astype(np.float32)
            ))
            labels.append(i)
    query = normalize((centroids[0] * 0.9 + rng.standard_normal(1024)).astype(np.float32))

    def top3(q, ds):
        return sorted(range(len(ds)), key=lambda i: -float(np.dot(q, ds[i])))[:3]

    full = top3(query, docs)
    cut = top3(truncate_and_renormalize(query, 64),
               [truncate_and_renormalize(d, 64) for d in docs])
    # Cluster-level agreement (both rankings surface cluster-0 docs):
    assert labels[full[0]] == labels[cut[0]] == 0
    assert {labels[i] for i in full} == {labels[i] for i in cut} == {0}
