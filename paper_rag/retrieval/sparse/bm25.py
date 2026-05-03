from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..chunk_fusion import RRF_K
from ..data.chunks import ChunkDocument
from ..data.utils import tokenize


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


def count_terms(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def search_bm25_chunks(documents: list[ChunkDocument], queries: list[str], top_k: int) -> list[BM25Hit]:
    """对多个 BM25 query 分别检索，再用 RRF 合并为 chunk 候选。"""
    bm25_documents = [
        BM25Document(
            document.chunk_id,
            f"{document.text}\n{document.embedding_text}",
            {"document": document},
        )
        for document in documents
    ]
    index = BM25Index(bm25_documents)
    by_id: dict[str, dict[str, Any]] = {}
    for query in queries:
        for rank, hit in enumerate(index.search(query, top_k), start=1):
            slot = by_id.setdefault(hit.doc_id, {"hit": hit, "score": 0.0})
            slot["score"] += 1 / (RRF_K + rank)
    fused = [
        BM25Hit(value["hit"].doc_id, value["score"], value["hit"].text, value["hit"].payload)
        for value in by_id.values()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused[:top_k]
