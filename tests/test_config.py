from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from embeddify.config import EmbedderConfig, RuntimeConfig, load_config_file
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


class TestRuntimeConfigValidation:
    def test_default_runtime_config_values(self) -> None:
        config = RuntimeConfig()

        assert config.batch_size == 32
        assert config.show_progress_bar is False
        assert config.enable_cache is False
        assert config.convert_to_numpy is False

    def test_batch_size_less_than_one_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            RuntimeConfig(batch_size=0)

        with pytest.raises(PydanticValidationError):
            RuntimeConfig(batch_size=-1)

    def test_cache_numpy_incompatible_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("WARNING", logger="embeddify.config")

        RuntimeConfig(enable_cache=True, convert_to_numpy=True)

        warnings = [
            record
            for record in caplog.records
            if record.levelname == "WARNING"
            and "enable_cache=True is incompatible with convert_to_numpy=True"
            in record.getMessage()
        ]
        assert warnings, "Expected warning about cache/NumPy incompatibility."


class TestRuntimeConfigFromEnv:
    def test_from_env_uses_defaults_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDIFY_BATCH_SIZE", raising=False)
        monkeypatch.delenv("EMBEDDIFY_SHOW_PROGRESS_BAR", raising=False)
        monkeypatch.delenv("EMBEDDIFY_ENABLE_CACHE", raising=False)
        monkeypatch.delenv("EMBEDDIFY_CONVERT_TO_NUMPY", raising=False)

        config = RuntimeConfig.from_env()

        assert config.batch_size == 32
        assert config.show_progress_bar is False
        assert config.enable_cache is False
        assert config.convert_to_numpy is False

    def test_from_env_parses_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDIFY_BATCH_SIZE", "16")
        monkeypatch.setenv("EMBEDDIFY_SHOW_PROGRESS_BAR", "1")
        monkeypatch.setenv("EMBEDDIFY_ENABLE_CACHE", "true")
        monkeypatch.setenv("EMBEDDIFY_CONVERT_TO_NUMPY", "0")

        config = RuntimeConfig.from_env()

        assert config.batch_size == 16
        assert config.show_progress_bar is True
        assert config.enable_cache is True
        assert config.convert_to_numpy is False

    def test_from_env_invalid_batch_size_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDIFY_BATCH_SIZE", "not-an-int")

        with pytest.raises(EmbeddifyValidationError):
            RuntimeConfig.from_env()

    def test_from_env_invalid_boolean_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDIFY_ENABLE_CACHE", "maybe")

        with pytest.raises(EmbeddifyValidationError):
            RuntimeConfig.from_env()



class TestConfigFileLoading:
    """Tests for loading configuration from YAML/JSON files."""

    def _fixture_path(self, filename: str) -> Path:
        return Path(__file__).parent / "fixtures" / "configs" / filename

    def test_load_yaml_config(self) -> None:
        model_cfg, runtime_cfg = load_config_file(str(self._fixture_path("valid.yaml")))

        assert model_cfg.model_path == "/models/all-MiniLM-L6-v2"
        assert model_cfg.device == "cpu"
        assert model_cfg.normalize_embeddings is True
        assert model_cfg.trust_remote_code is False

        assert runtime_cfg.batch_size == 16
        assert runtime_cfg.show_progress_bar is True
        assert runtime_cfg.enable_cache is True
        assert runtime_cfg.convert_to_numpy is False

    def test_load_json_config(self) -> None:
        model_cfg, runtime_cfg = load_config_file(str(self._fixture_path("valid.json")))

        assert model_cfg.model_path == "/models/all-MiniLM-L6-v2"
        assert runtime_cfg.batch_size == 16

    def test_env_overrides_file_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path = self._fixture_path("valid.yaml")
        monkeypatch.setenv("EMBEDDIFY_MODEL_PATH", "/override/model")
        monkeypatch.setenv("EMBEDDIFY_BATCH_SIZE", "64")

        model_cfg, runtime_cfg = load_config_file(str(path))

        assert model_cfg.model_path == "/override/model"
        assert runtime_cfg.batch_size == 64

    def test_uses_env_config_path_when_no_path_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = self._fixture_path("valid.yaml")
        monkeypatch.setenv("EMBEDDIFY_CONFIG_PATH", str(path))

        model_cfg, runtime_cfg = load_config_file()

        assert model_cfg.model_path == "/models/all-MiniLM-L6-v2"
        assert runtime_cfg.batch_size == 16

    def test_missing_config_file_raises_file_not_found(self) -> None:
        missing = self._fixture_path("does-not-exist.yaml")

        with pytest.raises(FileNotFoundError):
            load_config_file(str(missing))

    def test_malformed_config_raises_validation_error(self) -> None:
        malformed = self._fixture_path("invalid.yaml")

        with pytest.raises(EmbeddifyValidationError):
            load_config_file(str(malformed))
