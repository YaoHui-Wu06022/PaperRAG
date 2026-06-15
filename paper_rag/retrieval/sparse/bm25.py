"""本地 BM25 检索与多 query RRF 合并。"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from paper_rag.corpus.chunks import ChunkDocument
from paper_rag.corpus.utils import normalize_bm25_token
from paper_rag.retrieval.chunk_fusion import RRF_K


BM25_INDEX_VERSION = 1


@dataclass(frozen=True)
class BM25Document:
    """BM25 索引中的文档条目。"""

    doc_id: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BM25Hit:
    """一次 BM25 命中的轻量结果。"""

    doc_id: str
    score: float
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BM25ScopeStats:
    """某个 scope 内的 BM25 统计量。"""

    indices: list[int]
    avgdl: float
    doc_freqs: dict[str, int]


class BM25CorpusIndex:
    """预分词的 chunk 级 BM25 索引。"""

    def __init__(
        self,
        documents: list[BM25Document],
        *,
        tokens_by_doc_id: dict[str, list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        token_source = tokens_by_doc_id or {}
        self.doc_tokens = [
            list(token_source.get(document.doc_id) or normalize_bm25_token(document.text))
            for document in documents
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.term_freqs = [count_terms(tokens) for tokens in self.doc_tokens]
        self._scope_stats_cache: dict[tuple[str, ...] | None, BM25ScopeStats] = {}

    @classmethod
    def from_chunks(cls, chunk_documents: list[ChunkDocument]) -> "BM25CorpusIndex":
        """从 ChunkDocument 列表构建索引。"""
        return cls(bm25_documents_from_chunks(chunk_documents))

    @classmethod
    def load(cls, path: Path, chunk_documents: list[ChunkDocument]) -> "BM25CorpusIndex | None":
        """加载派生索引，索引过期时返回 None。"""
        if not path.exists():
            return None
        bm25_documents = bm25_documents_from_chunks(chunk_documents)
        expected_hashes = {document.doc_id: bm25_document_hash(document.text) for document in bm25_documents}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != BM25_INDEX_VERSION:
            return None
        rows = payload.get("documents")
        if not isinstance(rows, list) or len(rows) != len(bm25_documents):
            return None
        tokens_by_doc_id: dict[str, list[str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                return None
            doc_id = str(row.get("doc_id") or "")
            if not doc_id or expected_hashes.get(doc_id) != row.get("text_hash"):
                return None
            tokens = row.get("tokens")
            if not isinstance(tokens, list):
                return None
            tokens_by_doc_id[doc_id] = [str(token) for token in tokens]
        if set(tokens_by_doc_id) != set(expected_hashes):
            return None
        return cls(bm25_documents, tokens_by_doc_id=tokens_by_doc_id)

    def save(self, path: Path) -> None:
        """写出可删除、可重建的 BM25 派生索引。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": BM25_INDEX_VERSION,
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "text_hash": bm25_document_hash(document.text),
                    "tokens": self.doc_tokens[index],
                }
                for index, document in enumerate(self.documents)
            ],
        }
        tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        allowed_chunk_ids: Iterable[str] | None = None,
    ) -> list[BM25Hit]:
        """在指定 scope 内执行单 query BM25 检索。"""
        if top_k <= 0:
            return []
        query_tokens = normalize_bm25_token(query)
        stats = self.scope_stats(allowed_chunk_ids)
        if not query_tokens or not stats.indices:
            return []
        scored: list[BM25Hit] = []
        for index in stats.indices:
            score = self._score(query_tokens, index, stats)
            if score > 0:
                document = self.documents[index]
                scored.append(BM25Hit(document.doc_id, score, document.text, document.payload))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def search_many(
        self,
        queries: list[str],
        top_k: int,
        *,
        allowed_chunk_ids: Iterable[str] | None = None,
    ) -> list[BM25Hit]:
        """多个 BM25 query 分别检索后用 RRF 合并。"""
        if top_k <= 0:
            return []
        allowed_ids = None if allowed_chunk_ids is None else list(allowed_chunk_ids)
        stats = self.scope_stats(allowed_ids)
        if not stats.indices:
            return []
        by_id: dict[str, dict[str, Any]] = {}
        for query in queries:
            for rank, hit in enumerate(self.search(query, top_k, allowed_chunk_ids=allowed_ids), start=1):
                slot = by_id.setdefault(hit.doc_id, {"hit": hit, "score": 0.0})
                slot["score"] += 1 / (RRF_K + rank)
        fused = [
            BM25Hit(value["hit"].doc_id, value["score"], value["hit"].text, value["hit"].payload)
            for value in by_id.values()
        ]
        fused.sort(key=lambda hit: hit.score, reverse=True)
        return fused[:top_k]

    def scope_stats(self, allowed_chunk_ids: Iterable[str] | None) -> BM25ScopeStats:
        """按论文 scope 计算 IDF 所需统计量，复用已分好的 token。"""
        key = None if allowed_chunk_ids is None else tuple(sorted({str(chunk_id) for chunk_id in allowed_chunk_ids}))
        cached = self._scope_stats_cache.get(key)
        if cached is not None:
            return cached
        if allowed_chunk_ids is None:
            indices = list(range(len(self.documents)))
        else:
            allowed = set(key or ())
            indices = [
                index
                for index, document in enumerate(self.documents)
                if document.doc_id in allowed
            ]
        avgdl = sum(self.doc_lengths[index] for index in indices) / len(indices) if indices else 0.0
        doc_freqs: dict[str, int] = {}
        for index in indices:
            for token in set(self.doc_tokens[index]):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        stats = BM25ScopeStats(indices, avgdl, doc_freqs)
        self._scope_stats_cache[key] = stats
        return stats

    def _score(self, query_tokens: list[str], doc_index: int, stats: BM25ScopeStats) -> float:
        tokens = self.doc_tokens[doc_index]
        if not tokens:
            return 0.0
        term_freq = self.term_freqs[doc_index]
        score = 0.0
        doc_len = self.doc_lengths[doc_index]
        total_docs = len(stats.indices)
        for token in query_tokens:
            freq = term_freq.get(token, 0)
            if freq == 0:
                continue
            df = stats.doc_freqs.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (stats.avgdl or 1))
            score += idf * (freq * (self.k1 + 1) / denominator)
        return score


def bm25_documents_from_chunks(chunk_documents: list[ChunkDocument]) -> list[BM25Document]:
    """把 ChunkDocument 投影成 BM25Document。"""
    return [
        BM25Document(
            chunk_document.chunk_id,
            f"{chunk_document.text}\n{chunk_document.embedding_text}",
            {"chunk_document": chunk_document},
        )
        for chunk_document in chunk_documents
    ]


def bm25_document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_terms(tokens: list[str]) -> dict[str, int]:
    """统计 token 词频。"""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts
