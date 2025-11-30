from __future__ import annotations

import os
import sys
import types

import pytest
from pydantic import ValidationError as PydanticValidationError

from embeddify.config import EmbedderConfig
from embeddify.exceptions import ValidationError as EmbeddifyValidationError


class TestEmbedderConfigValidation:
    def test_valid_configuration_minimal(self) -> None:
        config = EmbedderConfig(model_path="/models/test-model")

        assert config.model_path == "/models/test-model"
        assert config.device == "cpu"
        assert config.normalize_embeddings is True
        assert config.trust_remote_code is False

    def test_empty_model_path_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_path="")

    def test_whitespace_model_path_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_path="   ")

    def test_invalid_device_string_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_path="/models/test", device="gpu")

        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_path="/models/test", device="cuda:x")

    def test_cuda_device_when_unavailable_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Provide a fake torch module where cuda.is_available() returns False.
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_path="/models/test", device="cuda")


class TestEmbedderConfigFromEnv:
    def test_from_env_successful_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Fake torch reporting CUDA availability so that "cuda" passes validation.
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        monkeypatch.setenv("EMBEDDIFY_MODEL_PATH", "/env/model")
        monkeypatch.setenv("EMBEDDIFY_DEVICE", "cuda")
        monkeypatch.setenv("EMBEDDIFY_NORMALIZE_EMBEDDINGS", "false")
        monkeypatch.setenv("EMBEDDIFY_TRUST_REMOTE_CODE", "true")

        config = EmbedderConfig.from_env()

        assert config.model_path == "/env/model"
        assert config.device == "cuda"
        assert config.normalize_embeddings is False
        assert config.trust_remote_code is True

    def test_from_env_missing_model_path_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDIFY_MODEL_PATH", raising=False)

        with pytest.raises(EmbeddifyValidationError):
            EmbedderConfig.from_env()

    def test_from_env_invalid_boolean_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        monkeypatch.setenv("EMBEDDIFY_MODEL_PATH", "/env/model")
        monkeypatch.setenv("EMBEDDIFY_NORMALIZE_EMBEDDINGS", "not-a-bool")

        with pytest.raises(EmbeddifyValidationError):
            EmbedderConfig.from_env()
