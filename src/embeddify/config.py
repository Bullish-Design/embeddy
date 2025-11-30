# src/embeddify/config.py
from __future__ import annotations

import os
import re
import logging
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from embeddify.exceptions import ValidationError as EmbeddifyValidationError


_BOOL_TRUE_VALUES: set[str] = {"1", "true", "yes", "on"}
_BOOL_FALSE_VALUES: set[str] = {"0", "false", "no", "off"}


class EmbedderConfig(BaseModel):
    """Configuration for SentenceTransformer model initialization.

    This model validates all parameters required to load a SentenceTransformer
    model. Validation happens before any attempt to load the model, enabling
    fail-fast behaviour and clear error messages.
    """

    model_path: str = Field(
        description="Path to a pre-downloaded Sentence Transformer model."
    )
    device: str = Field(
        default="cpu",
        description="Device identifier: 'cpu', 'cuda', or 'cuda:N'.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize embedding vectors produced by the model.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading models with custom architectures.",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        """Ensure ``model_path`` is a non-empty string.

        Existence of the path is validated later when the model is actually
        loaded; here we only guard against obviously invalid values.
        """
        if not isinstance(value, str):
            raise ValueError("model_path must be a string")
        if not value.strip():
            raise ValueError("model_path must be a non-empty string")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Validate the device string and, for CUDA, check availability.

        Accepted formats are:

        * ``"cpu"``
        * ``"cuda"``
        * ``"cuda:N"`` where ``N`` is a non-negative integer.
        """
        if not isinstance(value, str):
            raise ValueError("device must be a string")

        pattern = r"^(cpu|cuda|cuda:\\d+)$"
        if re.match(pattern, value) is None:
            raise ValueError(
                f"Invalid device value: {value!r}. Expected 'cpu', 'cuda', or 'cuda:N'."
            )

        if value.startswith("cuda"):
            try:
                import torch  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - import failure path
                raise ValueError(
                    "CUDA device requested but PyTorch with CUDA support is not available."
                ) from exc

            if not torch.cuda.is_available():  # type: ignore[attr-defined]
                raise ValueError(
                    "CUDA device requested but CUDA is not available on this system."
                )

        return value

    @classmethod
    def _parse_bool_env(cls, raw: str, env_name: str) -> bool:
        """Parse a boolean value from an environment variable string.

        Values are parsed case-insensitively using a small set of common
        representations. If the value cannot be interpreted as a boolean,
        an Embeddify :class:`ValidationError` is raised.
        """
        lowered = raw.strip().lower()
        if lowered in _BOOL_TRUE_VALUES:
            return True
        if lowered in _BOOL_FALSE_VALUES:
            return False

        raise EmbeddifyValidationError(
            f"Invalid boolean value {raw!r} for environment variable {env_name}."
        )

    @classmethod
    def from_env(cls) -> EmbedderConfig:
        """Construct configuration from environment variables.

        Environment variables:
            EMBEDDIFY_MODEL_PATH (required)
            EMBEDDIFY_DEVICE (optional, defaults to "cpu")
            EMBEDDIFY_NORMALIZE_EMBEDDINGS (optional, defaults to "true")
            EMBEDDIFY_TRUST_REMOTE_CODE (optional, defaults to "false")

        Returns:
            A validated :class:`EmbedderConfig` instance.

        Raises:
            EmbeddifyValidationError: If required variables are missing or
                contain invalid values.
        """
        model_path = os.getenv("EMBEDDIFY_MODEL_PATH")
        if model_path is None or not model_path.strip():
            raise EmbeddifyValidationError(
                "EMBEDDIFY_MODEL_PATH must be set to use EmbedderConfig.from_env()."
            )

        device = os.getenv("EMBEDDIFY_DEVICE", "cpu")

        normalize_raw = os.getenv("EMBEDDIFY_NORMALIZE_EMBEDDINGS", "true")
        trust_raw = os.getenv("EMBEDDIFY_TRUST_REMOTE_CODE", "false")

        normalize = cls._parse_bool_env(
            normalize_raw, "EMBEDDIFY_NORMALIZE_EMBEDDINGS"
        )
        trust_remote = cls._parse_bool_env(trust_raw, "EMBEDDIFY_TRUST_REMOTE_CODE")

        try:
            return cls(
                model_path=model_path,
                device=device,
                normalize_embeddings=normalize,
                trust_remote_code=trust_remote,
            )
        except Exception as exc:
            # Wrap any underlying Pydantic validation error with a domain-level error.
            raise EmbeddifyValidationError(
                f"Invalid Embedder configuration from environment: {exc}"
            ) from exc


class RuntimeConfig(BaseModel):
    """Configuration for runtime execution behaviour.

    This model controls batch processing, progress reporting, caching and output
    format. It is intentionally separate from :class:`EmbedderConfig` so that
    runtime concerns can be tuned independently of model loading.
    """

    batch_size: int = Field(
        default=32,
        description="Number of texts to encode per batch.",
    )
    show_progress_bar: bool = Field(
        default=False,
        description="Whether to display a progress bar for batch operations.",
    )
    enable_cache: bool = Field(
        default=False,
        description="Cache encoded texts to avoid recomputation.",
    )
    convert_to_numpy: bool = Field(
        default=False,
        description="Return embeddings as numpy.ndarray instead of list[float].",
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        """Ensure ``batch_size`` is at least one."""
        if value < 1:
            raise ValueError("batch_size must be at least 1")
        return value

    @model_validator(mode="after")
    def warn_cache_numpy_incompatible(self) -> "RuntimeConfig":
        """Emit warning when cache and numpy output are requested together."""
        if self.enable_cache and self.convert_to_numpy:
            logging.getLogger(__name__).warning(
                "RuntimeConfig: enable_cache=True is incompatible with "
                "convert_to_numpy=True; caching will be disabled."
            )
        return self

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Construct runtime configuration from environment variables.

        Environment variables:
            EMBEDDIFY_BATCH_SIZE (optional)
            EMBEDDIFY_SHOW_PROGRESS_BAR (optional boolean)
            EMBEDDIFY_ENABLE_CACHE (optional boolean)
            EMBEDDIFY_CONVERT_TO_NUMPY (optional boolean)
        """
        kwargs: dict[str, Any] = {}

        batch_raw = os.getenv("EMBEDDIFY_BATCH_SIZE")
        if batch_raw is not None:
            try:
                kwargs["batch_size"] = int(batch_raw)
            except ValueError as exc:  # pragma: no cover - edge parsing path
                raise EmbeddifyValidationError(
                    f"Invalid integer value {batch_raw!r} for EMBEDDIFY_BATCH_SIZE."
                ) from exc

        def _apply_bool(raw: str | None, env_name: str, key: str) -> None:
            if raw is None:
                return
            kwargs[key] = EmbedderConfig._parse_bool_env(raw, env_name)

        _apply_bool(
            os.getenv("EMBEDDIFY_SHOW_PROGRESS_BAR"),
            "EMBEDDIFY_SHOW_PROGRESS_BAR",
            "show_progress_bar",
        )
        _apply_bool(
            os.getenv("EMBEDDIFY_ENABLE_CACHE"),
            "EMBEDDIFY_ENABLE_CACHE",
            "enable_cache",
        )
        _apply_bool(
            os.getenv("EMBEDDIFY_CONVERT_TO_NUMPY"),
            "EMBEDDIFY_CONVERT_TO_NUMPY",
            "convert_to_numpy",
        )

        try:
            return cls(**kwargs)
        except Exception as exc:
            raise EmbeddifyValidationError(
                f"Invalid Runtime configuration from environment: {exc}"
            ) from exc



def load_config_file(path: str | None = None) -> tuple[EmbedderConfig, RuntimeConfig]:
    """Load configuration from a YAML or JSON file.

    The configuration file is expected to use a nested structure with ``model``
    and ``runtime`` sections. Environment variables can override values loaded
    from the file.

    Args:
        path: Optional path to the configuration file. When ``None``, the
            function falls back to the ``EMBEDDIFY_CONFIG_PATH`` environment
            variable.

    Returns:
        A tuple of (:class:`EmbedderConfig`, :class:`RuntimeConfig`).

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        EmbeddifyValidationError: If the configuration cannot be parsed or
            has an unexpected structure.
    """
    config_path_str = path or os.getenv("EMBEDDIFY_CONFIG_PATH")
    if not config_path_str:
        raise EmbeddifyValidationError(
            "No configuration path provided and EMBEDDIFY_CONFIG_PATH is not set."
        )

    config_path = Path(config_path_str)
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))

    try:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(config_path.read_text())
        elif config_path.suffix.lower() == ".json":
            loaded = json.loads(config_path.read_text())
        else:
            # Prefer YAML parsing for unknown extensions; this mirrors typical
            # config-file behaviour but still validates structure below.
            loaded = yaml.safe_load(config_path.read_text())
    except Exception as exc:  # pragma: no cover - parser specific failures
        raise EmbeddifyValidationError(
            f"Failed to parse configuration file {config_path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):
        raise EmbeddifyValidationError(
            "Configuration file must contain a mapping at the top level."
        )

    model_section = loaded.get("model", {})
    runtime_section = loaded.get("runtime", {})

    if not isinstance(model_section, dict) or not isinstance(runtime_section, dict):
        raise EmbeddifyValidationError(
            "Configuration sections 'model' and 'runtime' must be mappings."
        )

    # Map file keys to EmbedderConfig fields.
    model_kwargs: dict[str, Any] = {}
    if "path" in model_section:
        model_kwargs["model_path"] = model_section["path"]
    if "model_path" in model_section:
        model_kwargs["model_path"] = model_section["model_path"]
    if "device" in model_section:
        model_kwargs["device"] = model_section["device"]
    if "normalize_embeddings" in model_section:
        model_kwargs["normalize_embeddings"] = model_section["normalize_embeddings"]
    if "trust_remote_code" in model_section:
        model_kwargs["trust_remote_code"] = model_section["trust_remote_code"]

    # Map file keys to RuntimeConfig fields.
    runtime_kwargs: dict[str, Any] = {}
    for key in ("batch_size", "show_progress_bar", "enable_cache", "convert_to_numpy"):
        if key in runtime_section:
            runtime_kwargs[key] = runtime_section[key]

    # Apply environment overrides for model configuration.
    env_model_path = os.getenv("EMBEDDIFY_MODEL_PATH")
    if env_model_path is not None:
        model_kwargs["model_path"] = env_model_path

    env_device = os.getenv("EMBEDDIFY_DEVICE")
    if env_device is not None:
        model_kwargs["device"] = env_device

    env_normalize = os.getenv("EMBEDDIFY_NORMALIZE_EMBEDDINGS")
    if env_normalize is not None:
        model_kwargs["normalize_embeddings"] = EmbedderConfig._parse_bool_env(
            env_normalize, "EMBEDDIFY_NORMALIZE_EMBEDDINGS"
        )

    env_trust = os.getenv("EMBEDDIFY_TRUST_REMOTE_CODE")
    if env_trust is not None:
        model_kwargs["trust_remote_code"] = EmbedderConfig._parse_bool_env(
            env_trust, "EMBEDDIFY_TRUST_REMOTE_CODE"
        )

    # Apply environment overrides for runtime configuration.
    env_batch = os.getenv("EMBEDDIFY_BATCH_SIZE")
    if env_batch is not None:
        try:
            runtime_kwargs["batch_size"] = int(env_batch)
        except ValueError as exc:  # pragma: no cover - edge parsing path
            raise EmbeddifyValidationError(
                f"Invalid integer value {env_batch!r} for EMBEDDIFY_BATCH_SIZE."
            ) from exc

    def _apply_bool_env(key: str, env_name: str) -> None:
        raw = os.getenv(env_name)
        if raw is None:
            return
        runtime_kwargs[key] = EmbedderConfig._parse_bool_env(raw, env_name)

    _apply_bool_env("show_progress_bar", "EMBEDDIFY_SHOW_PROGRESS_BAR")
    _apply_bool_env("enable_cache", "EMBEDDIFY_ENABLE_CACHE")
    _apply_bool_env("convert_to_numpy", "EMBEDDIFY_CONVERT_TO_NUMPY")

    try:
        model_config = EmbedderConfig(**model_kwargs)
        runtime_config = RuntimeConfig(**runtime_kwargs)
    except Exception as exc:
        raise EmbeddifyValidationError(
            f"Invalid configuration values in {config_path}: {exc}"
        ) from exc

    return model_config, runtime_config
