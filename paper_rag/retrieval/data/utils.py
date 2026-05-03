from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar


T = TypeVar("T")
K = TypeVar("K")


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DASH_TRANSLATION = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "_": "-",
})

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    """把文本切成搜索/BM25 使用的 token。"""
    return [
        token
        for token in TOKEN_RE.findall(normalize_for_bm25(text))
        if token not in STOPWORDS
    ]


def normalize_for_bm25(text: str) -> str:
    """执行 BM25/tokenize 前的大小写和连字符规范化。"""
    return text.lower().translate(DASH_TRANSLATION)


def filter_value_to_list(value: Any) -> list[str]:
    """把 filter value 统一转成非空字符串列表。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def normalized_text(value: str) -> str:
    """生成确定性比较使用的规整文本。"""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def dedupe_by(values: Iterable[T], key_fn: Callable[[T], K | None]) -> list[T]:
    """按 key 函数对列表保序去重。"""
    seen: set[K] = set()
    result: list[T] = []
    for value in values:
        key = key_fn(value)
        if key is None or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def dedupe_alias_matches(matches: Iterable[T]) -> list[T]:
    """按 alias/canonical 对 alias match 保序去重。"""
    return dedupe_by(matches, lambda match: (getattr(match, "alias", ""), getattr(match, "canonical", "")))


def dedupe_text_values_for_search(values: list[str]) -> list[str]:
    """按搜索 token 语义对文本列表保序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = " ".join(tokenize(value))
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result
