"""resolve_dimension / registry test matrix (CONCEPT §5.2, plan §3).

Ports the Phase-0 spike's 22-test matrix (`spikes/test_protocols.py`,
22/22) against the real embeddy.registry, plus the CONCEPT §6.1 registry
fact table (verified against HF model cards 2026-08-01).
"""

from __future__ import annotations

import pytest

from embeddy import (
    DEFAULT_MODELS,
    ModelSpec,
    RegistryError,
    get_model,
    resolve_dimension,
    resolve_instruction,
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


@pytest.mark.parametrize(
    "requested,expected",
    [
        (32, 32),  # range start
        (64, 64),  # common MRL slice
        (256, 256),
        (1024, 1024),  # native (range end)
    ],
)
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
    """resolve_instruction is the ONLY role -> string resolver (CONCEPT
    §5.1); it reads the real registry, so the assertions use the registry's
    card-exact strings (not the fixture spec's synthetic ones)."""
    assert resolve_instruction(MRL_SPEC.id, "query") == (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery:"
    )
    assert resolve_instruction(MRL_SPEC.id, "document") == ""


def test_unknown_model_raises() -> None:
    with pytest.raises(RegistryError):
        get_model("no/such-model")


def test_unknown_role_raises() -> None:
    with pytest.raises(RegistryError):
        resolve_instruction(MRL_SPEC.id, "summary")  # no such role


def test_spec_invalid_mrl_range_rejected() -> None:
    with pytest.raises(ValueError):
        ModelSpec(
            id="bad",
            native_dimension=1024,
            mrl_range=range(32, 1024),
            context_length=100,
            instructions={},
        )  # native 1024 is NOT in range(32, 1024) (stop exclusive)


# --- registry fact table (CONCEPT §6.1, verified 2026-08-01) ------------------


def test_default_text_model_is_qwen3() -> None:
    spec = DEFAULT_MODELS["Qwen/Qwen3-Embedding-0.6B"]
    assert spec.native_dimension == 1024
    assert spec.mrl_range == range(32, 1025)
    assert spec.context_length == 32768
    assert spec.license == "Apache-2.0"


def test_qwen3_instructions_are_card_exact() -> None:
    """The registry stores the model card's exact prompt strings (verified
    against config_sentence_transformers.json). Qwen3's document role has
    NO prompt (empty string); the query role carries the "Instruct: ...\n
    Query:" wrapper."""
    spec = get_model("Qwen/Qwen3-Embedding-0.6B")
    assert spec.instructions["query"] == (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery:"
    )
    assert spec.instructions["document"] == ""


def test_qwen3_retrieval_role_shares_query_prompt() -> None:
    spec = get_model("Qwen/Qwen3-Embedding-0.6B")
    assert spec.instructions["retrieval"] == spec.instructions["query"]


def test_harrier_models_are_non_mrl() -> None:
    for model_id, native in [
        ("microsoft/harrier-oss-v1-0.6b", 1024),
        ("microsoft/harrier-oss-v1-270m", 640),
        ("microsoft/harrier-oss-v1-27b", 5376),
    ]:
        spec = get_model(model_id)
        assert spec.mrl_range is None, model_id
        assert spec.native_dimension == native, model_id
        assert spec.license == "MIT", model_id
        # card-exact web_search_query prompt (harrier prompt_name convention)
        assert spec.instructions["query"].endswith("answer the query\nQuery: "), model_id
        assert spec.instructions["document"] == "", model_id


def test_harrier_dimension_policy() -> None:
    """harrier is non-MRL: only its native dimension is valid."""
    assert resolve_dimension(get_model("microsoft/harrier-oss-v1-270m"), 640) == 640
    with pytest.raises(ValueError, match="not MRL-capable"):
        resolve_dimension(get_model("microsoft/harrier-oss-v1-270m"), 320)


def test_qwen3_vl_embedding_2b_facts() -> None:
    spec = get_model("Qwen/Qwen3-VL-Embedding-2B")
    assert spec.native_dimension == 2048
    assert spec.mrl_range == range(64, 2049)
    assert spec.context_length == 32768
    assert spec.license == "Apache-2.0"
    # the card registers one "default" prompt; both semantic roles map to it
    assert spec.instructions["query"] == "Represent the user's input."
    assert spec.instructions["document"] == "Represent the user's input."


def test_bge_m3_facts() -> None:
    spec = get_model("BAAI/bge-m3")
    assert spec.native_dimension == 1024
    assert spec.mrl_range is None  # bge-m3 has no MRL
    assert spec.context_length == 8192  # sentence_bert_config max_seq_length
    assert spec.instructions["query"] == "Represent this sentence for searching relevant passages: "
    assert spec.instructions["document"] == ""
