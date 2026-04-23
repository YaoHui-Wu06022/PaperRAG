from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DASH_TRANSLATION = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "_": "-",
})

# 停用词表保持保守：论文术语经常由普通英文词组成，不能过度过滤。
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


@dataclass(frozen=True)
class BM25Document:
    doc_id: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    score: float
    text: str
    payload: dict[str, Any]


class BM25Index:
    def __init__(self, documents: list[BM25Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.text) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [count_terms(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return []
        scored: list[BM25Hit] = []
        for index, document in enumerate(self.documents):
            score = self._score(query_tokens, index)
            if score > 0:
                scored.append(BM25Hit(document.doc_id, score, document.text, document.payload))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        tokens = self.doc_tokens[doc_index]
        if not tokens:
            return 0.0
        term_freq = self.term_freqs[doc_index]
        score = 0.0
        doc_len = self.doc_lengths[doc_index]
        total_docs = len(self.documents)
        for token in query_tokens:
            freq = term_freq.get(token, 0)
            if freq == 0:
                continue
            df = self.doc_freqs.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
            score += idf * (freq * (self.k1 + 1) / denominator)
        return score


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_for_bm25(text))
        if token not in STOPWORDS
    ]


def normalize_for_bm25(text: str) -> str:
    """统一不会改变词义的视觉变体，减少 BM25 字面匹配噪声。"""
    return text.lower().translate(DASH_TRANSLATION)


def count_terms(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts
