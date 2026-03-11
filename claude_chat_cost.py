# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "rich>=13.1",
#   "tiktoken>=0.4",
# ]
# ///

"""Estimate Claude chat token counts and optional cost per 1k tokens."""

from __future__ import annotations

import argparse
import dataclasses
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import tiktoken
from rich.console import Console
from rich.table import Table

DEFAULT_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "system": "system",
    "assistant": "assistant",
    "claude": "assistant",
    "model": "assistant",
}

ROLE_PATTERN_TEMPLATE = r"^(?P<role>{roles})\s*:\s*(?P<body>.*)$"


@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    text: str


@dataclasses.dataclass
class RoleStats:
    messages: int = 0
    tokens: int = 0

    def observe(self, count: int) -> None:
        self.messages += 1
        self.tokens += count


def _build_role_pattern(aliases: Mapping[str, str]) -> re.Pattern[str]:
    escaped_roles = sorted({re.escape(role) for role in aliases.keys()}, key=len, reverse=True)
    pattern = ROLE_PATTERN_TEMPLATE.format(roles="|".join(escaped_roles))
    return re.compile(pattern, re.IGNORECASE)


def parse_chat_lines(text: str, role_aliases: Mapping[str, str]) -> list[Message]:
    pattern = _build_role_pattern(role_aliases)
    messages: list[Message] = []
    current_role: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_body
        if not current_role or not current_body:
            return
        body_text = "\n".join(current_body).strip()
        if body_text:
            canonical_role = role_aliases.get(current_role, current_role)
            messages.append(Message(role=canonical_role, text=body_text))
        current_body = []

    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            flush()
            detected = match.group("role").lower()
            canonical = role_aliases.get(detected, detected)
            current_role = canonical
            remainder = match.group("body")
            current_body = [remainder] if remainder else []
            continue
        if current_role is None:
            current_role = "system"
        current_body.append(line)

    flush()
    return messages


def format_currency(tokens: int, price_per_1k: float) -> str:
    if price_per_1k <= 0:
        return ""
    dollars = tokens * price_per_1k / 1000
    return f"${dollars:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Claude chat token counts from logs")
    parser.add_argument("path", type=Path, help="Path to the chat log (text or markdown)")
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Token encoding name (default: cl100k_base for Claude).",
    )
    parser.add_argument(
        "--assistant-label",
        action="append",
        default=[],
        help="Additional label prefixes to normalize as assistant output.",
    )
    parser.add_argument(
        "--price-per-1k",
        type=float,
        default=0.0,
        help="Optional per-1000-token cost to estimate spending.",
    )
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="Show per-assistant message token counts.",
    )
    args = parser.parse_args()

    try:
        encoding = tiktoken.get_encoding(args.encoding)
    except KeyError as exc:
        raise SystemExit(f"Unknown encoding '{args.encoding}': {exc}") from exc

    role_aliases = DEFAULT_ROLE_ALIASES.copy()
    for alias in args.assistant_label:
        normalized = alias.strip().lower()
        if normalized:
            role_aliases[normalized] = "assistant"

    raw_text = args.path.read_text(encoding="utf-8")
    messages = parse_chat_lines(raw_text, role_aliases)

    total_tokens = len(encoding.encode(raw_text))
    stats: dict[str, RoleStats] = defaultdict(RoleStats)
    message_tokens: list[tuple[Message, int]] = []

    for message in messages:
        token_count = len(encoding.encode(message.text))
        stats[message.role].observe(token_count)
        if message.role == "assistant":
            message_tokens.append((message, token_count))

    console = Console()

    console.rule("Claude Token Summary")
    console.print(f"Path: {args.path}")
    console.print(f"Encoding: {args.encoding}")
    console.print(f"Total tokens (file): {total_tokens}")
    if args.price_per_1k > 0:
        console.print(f"Total cost estimate: {format_currency(total_tokens, args.price_per_1k)}")

    table = Table(box=None)
    table.add_column("Role", style="bold")
    table.add_column("Messages", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Avg tokens", justify="right")
    if args.price_per_1k > 0:
        table.add_column("Cost", justify="right")

    for role, summary in sorted(stats.items()):
        avg = summary.tokens / summary.messages if summary.messages else 0
        row = [role, str(summary.messages), str(summary.tokens), f"{avg:.1f}"]
        if args.price_per_1k > 0:
            row.append(format_currency(summary.tokens, args.price_per_1k))
        table.add_row(*row)

    console.print(table)

    if args.show_messages and message_tokens:
        detail_table = Table(title="Assistant messages", box=None)
        detail_table.add_column("#", style="bold", justify="right")
        detail_table.add_column("Tokens", justify="right")
        detail_table.add_column("Excerpt")
        for idx, (message, token_count) in enumerate(message_tokens, 1):
            excerpt = textwrap.shorten(message.text.replace("\n", " "), width=60, placeholder="…")
            detail_table.add_row(str(idx), str(token_count), excerpt)
        console.print(detail_table)


if __name__ == "__main__":
    main()
