# src/embeddify/config.py
from __future__ import annotations

import os
import re

from pydantic import BaseModel, Field, field_validator

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
