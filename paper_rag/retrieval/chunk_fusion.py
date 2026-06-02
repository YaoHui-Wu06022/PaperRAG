"""跨 dense/BM25 的 chunk 融合逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paper_rag.corpus.chunks import ChunkDocument
from paper_rag.retrieval.dense.milvus_store import SearchResult


RRF_K = 60


@dataclass(frozen=True)
class FusedChunk:
    """dense/BM25 融合后的 chunk 候选。"""

    chunk_document: ChunkDocument
    score: float
    sources: dict[str, dict[str, float | int]]
    dense_result: SearchResult | None = None


def fuse_chunk_hits(
    chunk_documents_by_id: dict[str, ChunkDocument],
    dense_results: list[SearchResult],
    bm25_results,
) -> list[FusedChunk]:
    """用 RRF 合并 dense 和 BM25 命中，并按 chunk_id 去重。"""
    by_id: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense_results, start=1):
        chunk_document = chunk_documents_by_id.get(result.chunk_id)
        if chunk_document is None:
            continue
        slot = by_id.setdefault(result.chunk_id, {"chunk_document": chunk_document, "score": 0.0, "sources": {}})
        slot["score"] += 1 / (RRF_K + rank)
        slot["sources"]["dense"] = {"rank": rank, "score": result.score}
        slot["dense_result"] = result
    for rank, hit in enumerate(bm25_results, start=1):
        # BM25 命中已经带回 ChunkDocument，这里只负责和 dense 分数同口径累加。
        chunk_document = hit.payload["chunk_document"]
        slot = by_id.setdefault(hit.doc_id, {"chunk_document": chunk_document, "score": 0.0, "sources": {}})
        slot["score"] += 1 / (RRF_K + rank)
        slot["sources"]["bm25"] = {"rank": rank, "score": hit.score}
    fused = [
        FusedChunk(
            chunk_document=value["chunk_document"],
            score=value["score"],
            sources=value["sources"],
            dense_result=value.get("dense_result"),
        )
        for value in by_id.values()
    ]
    fused.sort(key=lambda candidate: candidate.score, reverse=True)
    return fused
