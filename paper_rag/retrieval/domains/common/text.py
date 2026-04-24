from __future__ import annotations

import re
from typing import Any


def route_tokens(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def flatten_filter_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def normalized_text_key(value: str) -> str:
    return " ".join(route_tokens(value))


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalized_text_key(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result