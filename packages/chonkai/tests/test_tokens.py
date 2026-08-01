"""Token counting (chonkai/tokens.py): default counter + tokenizers extra."""

from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest

from chonkai.tokens import default_token_counter, tokenizers_token_counter


def test_default_counter_counts_tokens() -> None:
    counter = default_token_counter()
    assert counter("hello world") == 2  # cl100k_base encodes both words
    assert counter("") == 0
    assert counter("a") == 1


def test_default_counter_falls_back_to_heuristic_without_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # import raises ImportError
    with pytest.warns(UserWarning, match="tiktoken unavailable"):
        counter = default_token_counter()
    # chars/4 heuristic: "abcdefgh" -> 2
    assert counter("abcdefgh") == 2
    assert counter("x") == 1


def test_default_counter_falls_back_on_encoding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = cast(Any, types.ModuleType("tiktoken"))

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("cannot download ranks file")

    fake.get_encoding = fail
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    with pytest.warns(UserWarning, match="tiktoken unavailable"):
        counter = default_token_counter()
    assert counter("hello") == 1  # 5 chars // 4


def test_tokenizers_counter_uses_hf_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEncoding:
        ids: list[int] = [1, 2, 3]

    class _FakeTokenizer:
        def encode(self, text: str) -> _FakeEncoding:
            del text
            return _FakeEncoding()

        @classmethod
        def from_pretrained(cls, model: str) -> _FakeTokenizer:
            assert model == "some/model"
            return cls()

    fake = cast(Any, types.ModuleType("tokenizers"))
    fake.Tokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", fake)

    counter = tokenizers_token_counter("some/model")
    assert counter("anything") == 3


def test_tokenizers_counter_import_error_without_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(ImportError):
        tokenizers_token_counter("some/model")


def test_heuristic_is_deterministic() -> None:
    counter = default_token_counter()
    a, b = counter("same text " * 100), counter("same text " * 100)
    assert a == b
