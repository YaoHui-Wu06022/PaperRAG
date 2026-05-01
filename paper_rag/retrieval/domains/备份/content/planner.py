from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....config import Settings
from ...context import context_unit
from ...data.aliases import alias_match_to_dict
from ...data.chunks import ChunkDocument, load_chunk_documents
from ...data.filters import match_record_filter
from ...data.manifest_lookup import load_active_manifest_records
from ...dense.milvus_store import SearchResult
from ...dense.service import build_embedder, build_store
from ...evidence import to_evidence_papers
from ...sparse.bm25 import BM25Document, BM25Index
from ...top_router import RouteDecision
from ..common.text import flatten_filter_value


RRF_K = 60


@dataclass(frozen=True)
class FusedChunk:
    document: ChunkDocument
    score: float
    sources: dict[str, dict[str, float | int]]
    dense_result: SearchResult | None = None


def plan_body(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
    *,
    embedder=None,
    store=None,
) -> dict[str, Any]:
    if route.parse_status == "parse_failed":
        return {
            **build_content_evidence_base(route, retrieval_source=None),
            "parse_status": "parse_failed",
            "parser_error": route.parser_error,
            "context_units": [],
        }
    retrieval_source = build_content_retrieval_source(route)
    retrieval_query = retrieval_source["text"]
    documents = filter_content_chunks(settings, load_chunk_documents(settings.paper_data_dir), route)
    documents_by_id = {document.chunk_id: document for document in documents}
    try:
        dense_results = search_dense_chunks(settings, retrieval_query, embedder=embedder, store=store)
    except Exception as exc:
        warnings.append(f"dense retrieval failed: {exc}; using BM25 candidates only")
        dense_results = []
    bm25_results = search_bm25_chunks(documents, retrieval_query, settings.plan_bm25_top_k)
    fused = fuse_chunk_hits(documents_by_id, dense_results, bm25_results)
    context_units = [
        context_unit(settings, candidate, settings.plan_block_window)
        for candidate in fused[:settings.plan_final_top_k]
    ]
    if not context_units:
        warnings.append("body route found no dense/BM25 candidates")
    return {
        **build_content_evidence_base(route, retrieval_source=retrieval_source),
        "parse_status": "ok",
        "context_units": context_units,
    }


def build_content_evidence_base(route: RouteDecision, *, retrieval_source: dict[str, Any] | None) -> dict[str, Any]:
    parser_result = route.parser_result or {}
    evidence: dict[str, Any] = {
        "intent": route.intent,
        "anchors": route.anchors,
        "compare_objects": parser_result.get("compare_objects") or [],
        "objects": parser_result.get("objects") or [],
        "filters": route.filters,
        "resolved_papers": to_evidence_papers(route.resolved_papers),
        "alias_matches": [alias_match_to_dict(match) for match in route.alias_matches],
    }
    if retrieval_source is not None:
        evidence["retrieval_source"] = retrieval_source
    return evidence


def build_content_retrieval_source(route: RouteDecision) -> dict[str, Any]:
    parser_result = route.parser_result or {}
    anchors = list(route.anchors)
    compare_objects = list(parser_result.get("compare_objects") or [])
    objects = list(parser_result.get("objects") or [])
    filter_terms = build_content_filter_terms(route.filters)
    paper_titles = [str(paper.get("title") or "") for paper in route.resolved_papers if paper.get("title")]
    parts = [
        f"intent: {route.intent or 'content'}",
        f"objects: {', '.join(objects)}",
        f"compare_objects: {', '.join(compare_objects)}",
        f"anchors: {', '.join(anchors)}",
        f"papers: {', '.join(paper_titles)}",
        f"filters: {', '.join(filter_terms)}",
        f"question: {route.extract_query}",
    ]
    text = "; ".join(part for part in parts if not part.endswith(": "))
    return {
        "text": text,
        "intent": route.intent,
        "anchors": anchors,
        "compare_objects": compare_objects,
        "objects": objects,
        "filters": route.filters,
        "paper_titles": paper_titles,
    }


def build_content_filter_terms(filters: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for filter_item in filters:
        field = filter_item.get("field")
        op = filter_item.get("op")
        value = filter_item.get("value")
        prefix = "not " if filter_item.get("negated") else ""
        values = flatten_filter_value(value)
        if not values:
            values = [str(value)]
        terms.append(f"{prefix}{field} {op} {'/'.join(values)}")
    return terms


def search_dense_chunks(settings: Settings, query: str, *, embedder=None, store=None) -> list[SearchResult]:
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    query_vector = embedder.embed_texts([query])[0]
    return store.search(query_vector, settings.plan_dense_top_k)


def search_bm25_chunks(documents: list[ChunkDocument], query: str, top_k: int):
    bm25_documents = [
        BM25Document(
            document.chunk_id,
            f"{document.text}\n{document.embedding_text}",
            {"document": document},
        )
        for document in documents
    ]
    return BM25Index(bm25_documents).search(query, top_k)


def fuse_chunk_hits(
    documents_by_id: dict[str, ChunkDocument],
    dense_results: list[SearchResult],
    bm25_results,
) -> list[FusedChunk]:
    by_id: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense_results, start=1):
        document = documents_by_id.get(result.chunk_id)
        if document is None:
            continue
        slot = by_id.setdefault(result.chunk_id, {"document": document, "score": 0.0, "sources": {}})
        slot["score"] += 1 / (RRF_K + rank)
        slot["sources"]["dense"] = {"rank": rank, "score": result.score}
        slot["dense_result"] = result
    for rank, hit in enumerate(bm25_results, start=1):
        document = hit.payload["document"]
        slot = by_id.setdefault(hit.doc_id, {"document": document, "score": 0.0, "sources": {}})
        slot["score"] += 1 / (RRF_K + rank)
        slot["sources"]["bm25"] = {"rank": rank, "score": hit.score}
    fused = [
        FusedChunk(
            document=value["document"],
            score=value["score"],
            sources=value["sources"],
            dense_result=value.get("dense_result"),
        )
        for value in by_id.values()
    ]
    fused.sort(key=lambda candidate: candidate.score, reverse=True)
    return fused


def filter_chunks_by_papers(documents: list[ChunkDocument], resolved_papers: list[dict[str, Any]]) -> list[ChunkDocument]:
    if not resolved_papers:
        return documents
    paper_ids = {str(target.get("paper_id")) for target in resolved_papers if target.get("paper_id")}
    return [document for document in documents if document.paper_id in paper_ids]


def filter_content_chunks(settings: Settings, documents: list[ChunkDocument], route: RouteDecision) -> list[ChunkDocument]:
    documents = filter_chunks_by_papers(documents, route.resolved_papers)
    if not route.filters:
        return documents
    matched_paper_ids = {
        Path(str(record.paper_data_path)).name
        for record in load_active_manifest_records(settings)
        if record.paper_data_path and all(match_record_filter(settings, record, filter_item) for filter_item in route.filters)
    }
    return [document for document in documents if document.paper_id in matched_paper_ids]
