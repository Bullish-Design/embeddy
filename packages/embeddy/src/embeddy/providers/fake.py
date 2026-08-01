"""DEV/TEST-ONLY FakeProvider (dimension 8, deterministic). NEVER in production.

Conforms to the EmbeddingProvider protocol shape (dimension, context_length,
model_name, async encode). Each word is sha256-hashed to a single dim of the
8-dim space with a +/-1 sign (feature-hashing); a text's vector is the
L2-normalized sum. Same input -> same vector, across runs and machines.

`instruction` is accepted for protocol compatibility and does not change the
output (a real provider's prompt behavior is a Phase-3 concern).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from embeddy.protocol.types import EmbedInput, Vector, normalize_l2

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class FakeProvider:
    """Deterministic dim-8 feature-hash provider. Dev/test only."""

    dimension = 8
    context_length = 4096
    model_name = "fake/deterministic-dim8"

    async def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[Vector]:
        del instruction  # accepted for protocol compatibility; unused
        return [self._embed(item) for item in inputs]

    def _embed(self, item: EmbedInput) -> Vector:
        text = item if isinstance(item, str) else "<image>"
        acc = np.zeros(self.dimension, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")
            sign = 1.0 if (h >> 8) & 1 else -1.0
            acc[h % self.dimension] += sign
        if float(np.linalg.norm(acc)) == 0.0:
            acc = np.ones(self.dimension, dtype=np.float32)
        return normalize_l2(acc)
