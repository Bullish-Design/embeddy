"""Token counting for chunk budgets (CONCEPT §4.2, plan Phase 2).

The default counter uses tiktoken's cl100k_base encoding — the same BPE
ranks as the GPT-4-class / text-embedding-3 tokenizers — so chunk budgets
are token-accurate by default. tiktoken's first `get_encoding()` call
downloads the ranks file, so a documented deterministic fallback
(chars/4) covers offline machines.

Arbitrary HuggingFace tokenizers live behind the `chonkai[tokenizers]`
extra and are imported lazily inside `tokenizers_token_counter`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable


def _heuristic_token_counter(text: str) -> int:
    """Deterministic chars/4 estimate (≈4 chars/token for English)."""
    return max(1, len(text) // 4)


def default_token_counter() -> Callable[[str], int]:
    """Best-effort tiktoken cl100k_base counter; deterministic chars/4 fallback.

    Returns a callable `counter(text) -> int`. Falls back (with a warning)
    when tiktoken cannot load its encoding — ImportError on machines without
    the package, or a download/cache failure on offline machines, because
    tiktoken's first `get_encoding("cl100k_base")` fetches the BPE ranks
    file from the network. The fallback is deterministic and documented:
    token budgets are then an estimate, not exact.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # ImportError, download failure, cache miss
        warnings.warn(
            f"tiktoken unavailable ({exc!r}); falling back to a "
            "chars/4 token estimate (token budget not token-accurate)",
            stacklevel=2,
        )
        return _heuristic_token_counter
    return lambda text: len(enc.encode(text))


def tokenizers_token_counter(model: str) -> Callable[[str], int]:
    """Counter for an arbitrary HuggingFace tokenizer (`chonkai[tokenizers]`).

    `tokenizers` is imported lazily — calling this function is the only way
    it loads, and it raises ImportError when the extra is not installed.
    """
    from tokenizers import Tokenizer  # lazy extra import

    tokenizer = Tokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.encode(text).ids)
