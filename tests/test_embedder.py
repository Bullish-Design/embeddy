# tests/test_embedder.py
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from embeddify.config import EmbedderConfig, RuntimeConfig
from embeddify.embedder import Embedder
from embeddify.exceptions import EncodingError, ModelLoadError
from embeddify.models import Embedding, EmbeddingResult


class TestEmbedderInitialisation:
    def test_embedder_loads_model_with_valid_config(self, embedder: Embedder, mock_model_path: Path) -> None:
        """Embedder should load a SentenceTransformer instance on initialisation."""
        # The dummy SentenceTransformer stores the path and device attributes.
        assert embedder.config.model_path == str(mock_model_path)
        assert hasattr(embedder, "_model")

        model = embedder._model  # type: ignore[attr-defined]
        # The dummy model exposes ``model_name_or_path`` and ``device`` attributes.
        assert getattr(model, "model_name_or_path") == str(mock_model_path)
        assert getattr(model, "device") == embedder.config.device

    def test_model_name_property_uses_underlying_model(self, embedder: Embedder, mock_model_path: Path) -> None:
        """model_name should reflect the underlying model's identifier when available."""
        assert embedder.model_name == str(mock_model_path)

    def test_device_property_reflects_configured_device(self, embedder: Embedder) -> None:
        """device property should return the device configured for the model."""
        assert embedder.device == embedder.config.device

    def test_from_config_file_constructs_embedder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_config_file should load both model and runtime configuration."""
        # Build a minimal YAML configuration file.
        config_path = tmp_path / "embedder.yaml"
        config_path.write_text(
            """model:
  model_path: "{model_path}"
  device: "cpu"
runtime:
  batch_size: 8
""".format(
                model_path=tmp_path / "model-dir",
            ),
            encoding="utf-8",
        )

        # Ensure the referenced model path exists so EmbedderConfig is valid.
        model_dir = tmp_path / "model-dir"
        model_dir.mkdir()

        embedder = Embedder.from_config_file(str(config_path))

        assert isinstance(embedder, Embedder)
        assert embedder.config.model_path == str(model_dir)
        assert embedder.runtime_config.batch_size == 8

    def test_invalid_model_load_raises_model_load_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Errors from SentenceTransformer initialisation must surface as ModelLoadError."""

        def _failing_constructor(*_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")

        import sentence_transformers

        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            _failing_constructor,
            raising=True,
        )

        with pytest.raises(ModelLoadError) as exc_info:
            Embedder(config=embedder_config, runtime_config=runtime_config)

        message = str(exc_info.value)
        assert "Failed to load SentenceTransformer model" in message
        assert "boom" in message

    def test_cache_starts_empty(self, embedder: Embedder) -> None:
        """Embedder should initialise an empty in-memory cache for future steps."""
        cache = cast(dict[str, Embedding], embedder._cache)  # type: ignore[attr-defined]
        assert cache == {}



class TestEmbedderEncode:
    def test_encode_single_text_returns_embedding_result(self, embedder: Embedder) -> None:
        """Encoding a single string should yield one embedding in the result."""
        result = embedder.encode("hello world")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1

        embedding = result.embeddings[0]
        assert embedding.text == "hello world"
        assert embedding.model_name == embedder.model_name
        assert embedding.normalized == embedder.config.normalize_embeddings

        assert result.model_name == embedder.model_name
        assert result.dimensions == embedding.dimensions

    def test_encode_list_of_texts_produces_matching_embeddings(self, embedder: Embedder) -> None:
        """Encoding a list of texts should yield one embedding per text."""
        texts = ["one", "two", "three"]

        result = embedder.encode(texts)

        assert len(result.embeddings) == len(texts)
        assert result.model_name == embedder.model_name

        dims = {embedding.dimensions for embedding in result.embeddings}
        assert len(dims) == 1  # All embeddings share the same dimensionality.

        for original, embedding in zip(texts, result.embeddings):
            assert embedding.text == original

    def test_encode_empty_list_returns_empty_result(self, embedder: Embedder) -> None:
        """Encoding an empty list should return an empty EmbeddingResult."""
        result = embedder.encode([])

        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == []
        assert result.dimensions == 0

    def test_encode_with_none_entry_raises_encoding_error(self, embedder: Embedder) -> None:
        """None entries must be rejected with a clear EncodingError."""
        with pytest.raises(EncodingError):
            embedder.encode(["valid", None])  # type: ignore[list-item]

    def test_encode_with_empty_string_raises_encoding_error(self, embedder: Embedder) -> None:
        """Empty or whitespace-only strings are not valid inputs."""
        with pytest.raises(EncodingError):
            embedder.encode("   ")

    def test_encode_respects_convert_to_numpy_runtime_flag(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """When convert_to_numpy is enabled vectors should be numpy arrays."""
        runtime_config = RuntimeConfig(convert_to_numpy=True)
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        result = embedder.encode(["hello numpy"])
        assert len(result.embeddings) == 1

        vector = result.embeddings[0].vector
        assert isinstance(vector, np.ndarray)