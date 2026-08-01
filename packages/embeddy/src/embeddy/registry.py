"""Model registry — the single source of model facts, and the MRL policy point.

Ported from the Phase-0 spike (`spikes/protocols.py`). CONCEPT §5.2 / §6:
  * Model facts (dimension, context_length, mrl_range) come from the
    registry, never from config (fixes C2).
  * `resolve_dimension` is the exact CONCEPT §5.2 logic: None -> native;
    non-MRL + wrong dim -> error; MRL in range -> requested.
  * `truncate_and_renormalize` is the ONE truncation point — MRL truncation
    is ALWAYS followed by L2 re-normalization (a sliced vector is not
    unit-norm, and cosine scoring assumes unit vectors).
  * Role -> instruction resolution happens HERE (`resolve_instruction`), in
    the caller; providers only ever receive a resolved string (fixes H2).

Registry facts (CONCEPT §6.1) were re-verified against the HF model cards
on 2026-08-01: dimensions and context lengths match the table; instruction
strings are CARD-EXACT (the `config_sentence_transformers.json` values for
Qwen3-0.6B / harrier-0.6b; `""` where the card registers no prompt for a
role). The provider maps a resolved string back to a registered prompt name
by VALUE, so both the ST `prompt_name` (harrier: web_search_query/...) and
`prompt` (Qwen3: query/document) conventions work from one registry
dictionary (providers/local.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embeddy.protocol.types import Vector


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model fact entry. `mrl_range` is a half-open range over valid MRL
    dims; None means the model is not MRL-capable."""

    id: str
    native_dimension: int
    mrl_range: range | None  # None = not MRL-capable
    context_length: int
    instructions: dict[str, str]  # role -> resolved prompt string
    license: str = ""

    def __post_init__(self) -> None:
        if self.native_dimension < 1:
            raise ValueError("native_dimension must be >= 1")
        if self.mrl_range is not None:
            if not (
                self.mrl_range.step == 1
                and self.mrl_range.start >= 1
                and self.mrl_range.stop > self.mrl_range.start
            ):
                raise ValueError("mrl_range must be an ascending half-open range (step 1)")
            if self.native_dimension not in self.mrl_range:
                raise ValueError("native_dimension must lie inside mrl_range")


class RegistryError(ValueError):
    """Unknown model id / invalid dimension request / missing role."""


def resolve_dimension(spec: ModelSpec, requested: int | None) -> int:
    """CONCEPT §5.2 — exact logic. `requested is None` -> native dimension."""
    if requested is None:
        return spec.native_dimension
    if spec.mrl_range is None:
        if requested != spec.native_dimension:
            raise ValueError(
                f"{spec.id} is not MRL-capable; embedding_dimension must be {spec.native_dimension}"
            )
        return requested
    if requested not in spec.mrl_range:
        raise ValueError(
            f"{spec.id} supports MRL {spec.mrl_range.start}-"
            f"{spec.mrl_range.stop - 1}; got {requested}"
        )
    return requested


def truncate_and_renormalize(vector: Vector, dim: int) -> Vector:
    """MRL truncation followed by L2 re-normalization (CONCEPT §5.2).

    A sliced MRL vector is not unit-norm; cosine scoring assumes unit
    vectors, so the slice must be re-normalized. This is the ONE truncation
    point — the provider post-process calls it keyed on model facts.
    """
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D vector, got shape {arr.shape}")
    if dim > arr.shape[0]:
        raise ValueError(f"cannot truncate {arr.shape[0]}-dim vector to {dim} dims")
    if dim == arr.shape[0]:
        return arr
    return _normalize(arr[:dim])


def _normalize(vec: np.ndarray) -> Vector:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"cannot normalize zero/NaN vector (norm={norm})")
    return (arr / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Full registry (CONCEPT §6.1, verified 2026-08-01 against HF model cards).
# Role keys are the embeddy semantic roles (query/document/retrieval); values
# are the model card's exact prompt strings ("" = no prompt for that role).
# ---------------------------------------------------------------------------

_QWEN3_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

DEFAULT_MODELS: dict[str, ModelSpec] = {
    # Text default (CONCEPT §9.5): proven, ungated, MRL-capable, Apache-2.0.
    # Card-exact prompts (config_sentence_transformers.json): query gets the
    # "Instruct: ...\nQuery:" wrapper, document gets NO prompt ("").
    "Qwen/Qwen3-Embedding-0.6B": ModelSpec(
        id="Qwen/Qwen3-Embedding-0.6B",
        native_dimension=1024,
        mrl_range=range(32, 1025),
        context_length=32768,
        instructions={
            "query": _QWEN3_QUERY,
            "retrieval": _QWEN3_QUERY,
            "document": "",
        },
        license="Apache-2.0",
    ),
    # Multimodal default (CONCEPT §6.1). local-provider only in v1 (CONCEPT
    # §7: images are not expressible over OpenAI /v1/embeddings). The ST card
    # registers a single "default" prompt applied automatically; we register
    # it under both semantic roles so role resolution always yields a string.
    "Qwen/Qwen3-VL-Embedding-2B": ModelSpec(
        id="Qwen/Qwen3-VL-Embedding-2B",
        native_dimension=2048,
        mrl_range=range(64, 2049),
        context_length=32768,
        instructions={
            "query": "Represent the user's input.",
            "retrieval": "Represent the user's input.",
            "document": "Represent the user's input.",
        },
        license="Apache-2.0",
    ),
    # High-quality text option (CONCEPT §6.1/§9.5). NOT the default: newer,
    # single-source-corroborated, non-MRL. ST pooling VERIFIED 2026-08-01
    # (last-token + include_prompt + Normalize — matches the card). The card's
    # ST prompt names are web_search_query/sts_query/bitext_query; the query
    # string is registered under the semantic roles and matched by value in
    # the provider (prompt_name convention), documents get no prompt.
    "microsoft/harrier-oss-v1-0.6b": ModelSpec(
        id="microsoft/harrier-oss-v1-0.6b",
        native_dimension=1024,
        mrl_range=None,
        context_length=32768,
        instructions={
            "query": (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            ),
            "retrieval": (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            ),
            "document": "",
        },
        license="MIT",
    ),
    "microsoft/harrier-oss-v1-270m": ModelSpec(
        id="microsoft/harrier-oss-v1-270m",
        native_dimension=640,
        mrl_range=None,
        context_length=32768,
        instructions={
            "query": (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            ),
            "retrieval": (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            ),
            "document": "",
        },
        license="MIT",
    ),
    "microsoft/harrier-oss-v1-27b": ModelSpec(
        id="microsoft/harrier-oss-v1-27b",
        native_dimension=5376,
        mrl_range=None,
        context_length=32768,
        instructions={
            "query": (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            ),
            "retrieval": (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            ),
            "document": "",
        },
        license="MIT",
    ),
    # Sparse hybrid (Qdrant path, CONCEPT §6.1): dense+sparse, 8192 ctx
    # (verified sentence_bert_config.json max_seq_length). No MRL. The card
    # defines only the query prefix; documents get no prompt. No ST-registered
    # prompt names, so the provider applies the string via `prompt=` directly.
    "BAAI/bge-m3": ModelSpec(
        id="BAAI/bge-m3",
        native_dimension=1024,
        mrl_range=None,
        context_length=8192,
        instructions={
            "query": "Represent this sentence for searching relevant passages: ",
            "retrieval": "Represent this sentence for searching relevant passages: ",
            "document": "",
        },
        license="MIT",
    ),
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return DEFAULT_MODELS[model_id]
    except KeyError:
        raise RegistryError(f"unknown model: {model_id!r}") from None


def resolve_instruction(model_id: str, role: str) -> str:
    """The ONLY place role -> instruction is resolved (CONCEPT §5.1).

    The pipeline/search/server call this and hand the RESULT string to the
    provider. Providers never see a role and can never mix document vs query
    instructions (the H2 bug class).
    """
    spec = get_model(model_id)
    try:
        return spec.instructions[role]
    except KeyError:
        raise RegistryError(f"model {model_id!r} has no instruction for role {role!r}") from None
