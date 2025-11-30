# src/embeddify/config.py
from __future__ import annotations

import os
import re
import logging
from typing import Any

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
