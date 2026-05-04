"""content planner：先收缩论文范围，再做 dense/BM25 chunk 检索。"""

from __future__ import annotations

from typing import Any

from ....config import Settings
from ...chunk_fusion import fuse_chunk_hits
from ...data.chunks import filter_chunks_by_paper_records, load_chunk_documents
from ...data.manifest_records import dedupe_paper_records
from ...data.paper_scope import combined_semantic, records_for_scope
from ...dense.service import search_dense_chunks
from ...evidence import build_content_evidence
from ...route import RouteDecision
from ...sparse.bm25 import search_bm25_chunks
from .context import context_unit
from .retrieval_query import build_content_retrieval_query
from .translation import CloudKeywordTranslator, KeywordTranslator


def plan_body(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
    *,
    embedder=None,
    store=None,
    translator: KeywordTranslator | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """执行正文检索计划，并构造 content evidence。"""
    if route.parse_status == "parse_failed":
        return build_content_evidence(
            route,
            status="parse_failed",
            warnings=warnings,
            scope_records=[],
            context_units=[],
            parser_error=route.parser_error,
            debug=debug,
        )

    if route.group_mode in {"per", "or", "and"}:
        # content 的 group 当前只影响论文范围；chunk 检索仍在合并后的候选论文内执行。
        scope_records = dedupe_paper_records([
            record
            for group in route.paper_groups
            for record in records_for_scope(
                settings,
                combined_semantic(route.paper_semantic, group.get("semantic") or ""),
                [*route.filters, *(group.get("filters") or [])],
                route.group_mode,
            )
        ])
        group_results = [
            {
                "semantic": group.get("semantic") or "",
                "filters": group.get("filters") or [],
                "records": records_for_scope(
                    settings,
                    combined_semantic(route.paper_semantic, group.get("semantic") or ""),
                    [*route.filters, *(group.get("filters") or [])],
                    route.group_mode,
                ),
            }
            for group in route.paper_groups
        ]
    else:
        scope_records = records_for_scope(settings, route.paper_semantic, route.filters, route.group_mode)
        group_results = []
    retrieval_query = build_content_retrieval_query(
        settings,
        route,
        warnings,
        translator=translator or CloudKeywordTranslator(),
    )
    documents = filter_chunks_by_paper_records(load_chunk_documents(settings.paper_data_dir), scope_records)
    documents_by_id = {document.chunk_id: document for document in documents}
    if not scope_records:
        warnings.append("content route found no matching paper scope records")
    if not documents:
        warnings.append("content route found no matching chunks")
        dense_results = []
    else:
        try:
            dense_results = search_dense_chunks(settings, retrieval_query["dense_query"], embedder=embedder, store=store)
        except Exception as exc:
            warnings.append(f"dense retrieval failed: {exc}; using BM25 candidates only")
            dense_results = []
    bm25_results = search_bm25_chunks(documents, retrieval_query["bm25_queries"], settings.plan_bm25_top_k)
    # dense 与 BM25 各自召回后，用 RRF 在 chunk 粒度做最终排序。
    fused = fuse_chunk_hits(documents_by_id, dense_results, bm25_results)
    context_units = [
        context_unit(settings, candidate, settings.plan_block_window)
        for candidate in fused[:settings.plan_final_top_k]
    ]
    if not context_units:
        warnings.append("body route found no dense/BM25 candidates")
    return build_content_evidence(
        route,
        status="ok",
        warnings=warnings,
        scope_records=scope_records,
        context_units=context_units,
        retrieval_query=retrieval_query,
        group_results=group_results or None,
        debug=debug,
    )
