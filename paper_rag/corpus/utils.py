"""corpus 层的轻量工具函数。"""

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


def normalize_bm25_token(text: str) -> list[str]:
    """把文本切成 BM25/search 使用的 token，会移除英文停用词。"""
    text = text.lower().translate(DASH_TRANSLATION)
    return [
        token
        for token in TOKEN_RE.findall(text)
        if token not in STOPWORDS
    ]


def normalize_token(value: str) -> str:
    """生成确定性比较用的 token key，不删除停用词。"""
    # 适合 contains/in/tag/alias 这类确定性判断。
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def value_to_text_list(value: Any) -> list[str]:
    """把单值或列表统一整理成非空字符串列表。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def dedupe_by(values: Iterable[T], key_fn: Callable[[T], K | None]) -> list[T]:
    """按外部提供的 key 函数保序去重。"""
    seen: set[K] = set()
    result: list[T] = []
    for value in values:
        key = key_fn(value)
        if key is None or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def dedupe_text(values: Iterable[Any]) -> list[str]:
    """按原始文本去重：只 strip 和大小写折叠，不切词、不删停用词。"""
    texts = (str(value or "").strip() for value in values)
    return dedupe_by(texts, lambda text: text.casefold() if text else None)


def dedupe_bm25_text(values: list[str]) -> list[str]:
    """按 BM25 token 去重：会切词、统一连字符并删除英文停用词。"""
    return dedupe_by(values, lambda value: " ".join(normalize_bm25_token(value)) or None)


def normalize_interval_bound_text(value: Any) -> str:
    """规范化 year interval 的无穷边界文本。"""
    return value.strip().lower() if isinstance(value, str) else ""


def is_negative_infinity(value: Any) -> bool:
    """判断 interval 下界是否为 -inf。"""
    return normalize_interval_bound_text(value) == "-inf"


def is_positive_infinity(value: Any) -> bool:
    """判断 interval 上界是否为 inf/+inf。"""
    return normalize_interval_bound_text(value) in {"inf", "+inf"}


def interval_bound_as_int(value: Any) -> int | None:
    """把区间边界转成 int，失败时返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
