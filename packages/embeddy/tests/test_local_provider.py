"""LocalProvider tests — injected fake sentence-transformers model.

The fake implements only the slice of the ST 5.x API LocalProvider uses:
`.prompts` dict, `.encode(inputs, batch_size=, device=, convert_to_numpy=,
**prompt_kwargs)` returning float32 vectors of the native dimension.
Verified ST 5.6.1 API facts (2026-08-01): `prompt_name` looks up
`model.prompts`; `prompt` is used directly and wins over `prompt_name`;
`truncate_dim` alone does NOT re-normalize (our truncate_and_renormalize
post-process is the load-bearing truncation point).
"""

from __future__ import annotations

import sys
from typing import Any, cast

import numpy as np
import pytest

from embeddy import RegistryError
from embeddy.errors import ModelNotLoadedError, ProviderInputError
from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.protocol.types import ImageInput, assert_unit_vector
from embeddy.providers.local import LocalProvider

QWEN3_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


class FakeSTModel:
    """Duck-typed ST model: records encode calls, returns unit vectors."""

    prompts: dict[str, str] = {"query": QWEN3_QUERY, "document": ""}

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    def encode(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        self.calls.append((list(inputs), dict(kwargs)))
        vec = np.full(self._dim, 1.0 / np.sqrt(self._dim), dtype=np.float32)
        return np.stack([vec.copy() for _ in inputs])


def _qwen3_provider(model: FakeSTModel, dimension: int | None = 256) -> LocalProvider:
    return LocalProvider("Qwen/Qwen3-Embedding-0.6B", dimension, model=model)


# --- protocol conformance + registry facts ------------------------------------


def test_conforms_to_embedding_protocol() -> None:
    provider = _qwen3_provider(FakeSTModel())
    assert isinstance(provider, EmbeddingProvider)


def test_dimension_and_context_from_registry() -> None:
    provider = _qwen3_provider(FakeSTModel(), dimension=256)
    assert provider.dimension == 256  # MRL-resolved
    assert provider.native_dimension == 1024
    assert provider.mrl_range == range(32, 1025)
    assert provider.context_length == 32768  # from the registry, never config
    assert provider.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert provider.supports_instructions is True


def test_non_mrl_wrong_dimension_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="not MRL-capable"):
        LocalProvider("microsoft/harrier-oss-v1-0.6b", 512, model=FakeSTModel())


def test_unknown_model_raises() -> None:
    with pytest.raises(RegistryError, match="unknown model"):
        LocalProvider("no/such-model", model=FakeSTModel())


def test_mrl_out_of_range_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="MRL"):
        LocalProvider("Qwen/Qwen3-Embedding-0.6B", 16, model=FakeSTModel())


# --- encode: MRL truncation + re-normalization ---------------------------------


async def test_encode_mrl_truncates_and_renormalizes() -> None:
    model = FakeSTModel(dim=1024)
    provider = _qwen3_provider(model, dimension=64)
    vectors = await provider.encode(["hello", "world"])
    assert len(vectors) == 2
    for v in vectors:
        assert v.shape == (64,), "MRL slice must have the resolved dimension"
        assert v.dtype == np.float32
        assert_unit_vector(v)  # sliced vectors MUST be re-normalized


async def test_encode_native_dimension_passthrough() -> None:
    model = FakeSTModel(dim=1024)
    provider = _qwen3_provider(model, dimension=None)  # native
    vectors = await provider.encode(["hello"])
    assert vectors[0].shape == (1024,)
    assert_unit_vector(vectors[0])


# --- prompt conventions --------------------------------------------------------


async def test_encode_resolved_instruction_uses_prompt_name() -> None:
    """Qwen3 convention: the resolved string matches a registered prompt
    VALUE, so ST's prompt_name lookup applies (harrier/Qwen3 prompt_name
    path)."""
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    await provider.encode(["hello"], instruction=QWEN3_QUERY)
    _, kwargs = model.calls[0]
    assert kwargs.get("prompt_name") == "query"
    assert "prompt" not in kwargs


async def test_encode_unregistered_instruction_uses_prompt_string() -> None:
    """bge-m3 convention: no registered ST prompt matches, so the resolved
    string is passed via `prompt=` directly."""
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    await provider.encode(
        ["hello"], instruction="Represent this sentence for searching relevant passages: "
    )
    _, kwargs = model.calls[0]
    assert kwargs.get("prompt") == "Represent this sentence for searching relevant passages: "
    assert "prompt_name" not in kwargs


@pytest.mark.parametrize("instruction", [None, ""])
async def test_encode_empty_instruction_sends_no_prompt(instruction: str | None) -> None:
    """Qwen3/harrier document role resolves to "" — no prompt kwargs at all."""
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    await provider.encode(["hello"], instruction=instruction)
    _, kwargs = model.calls[0]
    assert "prompt" not in kwargs and "prompt_name" not in kwargs


# --- batching / device / dtype -------------------------------------------------


async def test_encode_passes_batch_and_device() -> None:
    model = FakeSTModel()
    provider = LocalProvider(
        "Qwen/Qwen3-Embedding-0.6B", 256, model=model, batch_size=7, device="cpu"
    )
    await provider.encode(["a", "b", "c"])
    _, kwargs = model.calls[0]
    assert kwargs.get("batch_size") == 7
    assert kwargs.get("device") == "cpu"
    assert kwargs.get("convert_to_numpy") is True


async def test_encode_empty_inputs_returns_empty() -> None:
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    assert await provider.encode([]) == []


# --- inputs --------------------------------------------------------------------


async def test_image_input_converted_to_st_dict() -> None:
    """Multimodal path (local-provider only, CONCEPT §7): ImageInput is
    decoded to the ST {'image': RGB ndarray} shape. PIL is a lazy ST
    dependency (sentence-transformers 5.x does not require it), so the test
    skips when it is not installed."""
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    await provider.encode([ImageInput(data=buf.getvalue(), mime="image/png")])
    inputs, _ = model.calls[0]
    assert isinstance(inputs[0], dict) and "image" in inputs[0]
    assert np.asarray(inputs[0]["image"]).shape == (8, 8, 3)


async def test_bad_image_bytes_raises() -> None:
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    with pytest.raises(ProviderInputError, match="decode"):
        await provider.encode([ImageInput(data=b"not-an-image", mime="image/png")])


async def test_unsupported_input_type_raises() -> None:
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    with pytest.raises(ProviderInputError, match="unsupported input type"):
        await provider.encode(cast(Any, [123]))


# --- lazy import (zero-extras behavior) ----------------------------------------


async def test_encode_without_st_extra_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `embeddy[local]` extra not installed -> ModelNotLoadedError with
    an actionable message (and import embeddy stays clean, tested by the
    zero-extras smoke test)."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    provider = LocalProvider("Qwen/Qwen3-Embedding-0.6B", 256)  # no injected model
    with pytest.raises(ModelNotLoadedError, match="embeddy\\[local\\]"):
        await provider.encode(["hello"])


def test_constructor_is_lazy() -> None:
    """Constructing the provider never touches sentence-transformers."""
    provider = LocalProvider("Qwen/Qwen3-Embedding-0.6B", 256)
    assert provider.dimension == 256  # resolution happens eagerly
    assert provider._model is None  # model load is lazy


async def test_close_releases_model() -> None:
    model = FakeSTModel()
    provider = _qwen3_provider(model)
    provider.close()
    assert provider._model is None
