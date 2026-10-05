"""全文检索查询词规范化。"""

from __future__ import annotations

import re
from collections.abc import Iterable


ENGLISH_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "for", "from", "by", "with",
        "and", "or", "is", "are", "was", "were", "be", "been", "being",
        "what", "which", "who", "how", "why", "when", "where",
        "do", "does", "did", "can", "could", "would", "should",
        "about", "into", "as", "at", "it", "this", "that", "these", "those",
        "please", "find", "out", "list", "show", "give", "all", "paper", "papers",
        "library", "related", "relevant",
    }
)


def extract_terms(value: str, *, remove_stopwords: bool = False) -> tuple[list[str], list[str]]:
    """提取 FTS5 查询词，并返回被过滤的停用词。"""

    text = re.sub(r"[-_]+", " ", str(value or "").casefold())
    candidates = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]+", text)
    terms: list[str] = []
    removed: list[str] = []
    for term in candidates:
        if remove_stopwords and term in ENGLISH_STOPWORDS:
            removed.append(term)
            continue
        if len(term) <= 1:
            continue
        terms.append(term)
    if re.search(r"[\u4e00-\u9fff]", text):
        compact = re.sub(r"\s+", "", text)
        if compact and len(compact) > 1 and compact not in terms:
            terms.append(compact)
    return terms, list(dict.fromkeys(removed))


def build_fts_query(
    value: str,
    *,
    remove_stopwords: bool = False,
    phrases: Iterable[str] | None = None,
) -> str:
    """将查询词和完整短语转换为安全的 FTS5 OR 表达式。"""

    terms, _ = extract_terms(value, remove_stopwords=remove_stopwords)
    expressions: list[str] = ['"' + term.replace('"', '""') + '"' for term in terms]
    for phrase in phrases or ():
        phrase_terms, _ = extract_terms(phrase, remove_stopwords=remove_stopwords)
        if not phrase_terms:
            continue
        expression = '"' + " ".join(phrase_terms).replace('"', '""') + '"'
        if expression not in expressions:
            expressions.append(expression)
    return " OR ".join(expressions)


def normalize_lexical_text(value: str) -> tuple[str, list[str]]:
    """生成词法查询文本，并记录被移除的停用词。"""

    terms, removed = extract_terms(value, remove_stopwords=True)
    return " ".join(terms), removed


__all__ = ["ENGLISH_STOPWORDS", "build_fts_query", "extract_terms", "normalize_lexical_text"]
