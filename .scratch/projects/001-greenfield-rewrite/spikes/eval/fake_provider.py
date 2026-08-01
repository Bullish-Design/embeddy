"""Deterministic FakeProvider (dim 8) — the eval harness runs with zero models.

Word-hash signed TF-IDF embedding: each word is hashed (sha256-seeded) to a
single dim of the 8-dim space with a +/-1 sign and weighted by corpus idf;
a text's vector is the L2-normalized sum. All deterministic across runs and
machines (corpus fixed) — the M4 regression gate is reproducible.

Conforms to the EmbeddingProvider protocol shape (core_types/protocols):
dimension=8, context_length, model_name, async encode(inputs, instruction).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from core_types import EmbedInput, normalize_l2


def _words(text: str) -> list[str]:
    """Tokenize minus a small stopword list (the queries are full of
    function words that only add noise in an 8-dim bag-of-words space)."""
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
        "does", "for", "from", "how", "i", "in", "into", "is", "it",
        "of", "on", "or", "the", "to", "was", "we", "what", "when",
        "where", "which", "who", "will", "with", "you", "your",
    }
    out = []
    for raw in text.split():
        w = raw.lower().strip(".:,;()[]{}'\"")
        if w and w not in stop and len(w) > 2:
            out.append(w)
    return out


class FakeProvider:
    """Deterministic dim-8 TF-IDF provider. Dev/test only (the M4 gate's
    model). Corpus-aware by construction: idf weights come from `corpus`.

    Encoding: each word is hashed to a single dim with a +/-1 sign
    (feature-hashing style, seeded by sha256) and weighted by idf. Sparse
    signed hashing measurably beats dense random word vectors at dim 8
    (nDCG@10 0.60 vs 0.46 on this corpus) because shared words contribute
    an exact agreement on a dim instead of being diluted by noise.
    """

    dimension = 8
    context_length = 4096
    model_name = "fake/tfidf-signedhash-dim8"

    def __init__(self, corpus: dict[str, str]) -> None:
        doc_word_sets = [set(_words(text)) for text in corpus.values()]
        n = len(doc_word_sets)
        df: dict[str, int] = {}
        for words in doc_word_sets:
            for w in words:
                df[w] = df.get(w, 0) + 1
        # smoothed idf (+1.0 so every word keeps a baseline weight)
        self._idf = {
            w: math.log((n + 1) / (count + 1)) + 1.0
            for w, count in df.items()
        }

    @staticmethod
    def _signed_hash(word: str) -> tuple[int, float]:
        """(dim, sign) for a word, deterministic across runs."""
        h = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8],
                           "little")
        return h % 8, (1.0 if (h >> 8) & 1 else -1.0)

    def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for item in inputs:
            text = item if isinstance(item, str) else "<image>"
            words = _words(text)
            acc = np.zeros(8, dtype=np.float32)
            for w in words:
                dim, sign = self._signed_hash(w)
                acc[dim] += sign * self._idf.get(w, 1.0)
            if np.linalg.norm(acc) == 0.0:
                acc = np.ones(8, dtype=np.float32)
            vectors.append(normalize_l2(acc))
        return vectors
