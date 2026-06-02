"""content planner：先收缩论文范围，再做 dense/BM25 chunk 检索。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.corpus.records import paper_record_keys
from paper_rag.corpus.scope import resolve_scope_records
from paper_rag.retrieval.chunk_fusion import fuse_chunk_hits
from paper_rag.retrieval.dense.service import search_dense_chunks
from paper_rag.retrieval.evidence import build_content_evidence
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content.context import context_unit
from paper_rag.retrieval.routes.content.retrieval_query import build_content_retrieval_query
from paper_rag.retrieval.routes.content.translation import CloudKeywordTranslator, KeywordTranslatorProtocol
from paper_rag.retrieval.timing import Timings


def plan_body(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
    *,
    embedder=None,
    store=None,
    translator: KeywordTranslatorProtocol | None = None,
    debug: bool = False,
    corpus: CorpusContext | None = None,
    timings: Timings | None = None,
) -> dict[str, Any]:
    """执行正文检索计划，并构造 content evidence。"""
    corpus = corpus or CorpusContext(settings)
    timings = timings or Timings(False)
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

    with timings.measure("scope"):
        scope_records, group_results = resolve_scope_records(
            settings,
            route.paper_semantic,
            route.filters,
            route.paper_groups,
            route.group_mode,
            corpus=corpus,
        )
    if not scope_records:
        warnings.append("content 路由没有匹配到论文范围记录")
        return build_content_evidence(
            route,
            status="ok",
            warnings=warnings,
            scope_records=scope_records,
            context_units=[],
            group_results=group_results or None,
            debug=debug,
        )

    with timings.measure("retrieval_query"):
        retrieval_query = build_content_retrieval_query(
            settings,
            route,
            warnings,
            translator=translator or CloudKeywordTranslator(),
        )

    with timings.measure("load_chunks"):
        chunk_documents = corpus.content_chunks_for_records(scope_records)
    chunk_documents_by_id = {
        chunk_document.chunk_id: chunk_document
        for chunk_document in chunk_documents
    }
    if not chunk_documents:
        warnings.append("content 路由没有匹配到正文 chunk")
        context_units: list[dict[str, Any]] = []
    else:
        dense_results = []
        try:
            with timings.measure("dense"):
                dense_results = search_dense_chunks(
                    settings,
                    retrieval_query["dense_query"],
                    paper_ids=paper_record_keys(scope_records),
                    embedder=embedder,
                    store=store,
                )
        except Exception as exc:
            warnings.append(f"Dense 检索失败：{exc}；仅使用 BM25 候选")
        with timings.measure("bm25"):
            bm25_results = corpus.bm25_index.search_many(
                retrieval_query["bm25_queries"],
                settings.plan_bm25_top_k,
                allowed_chunk_ids=[chunk_document.chunk_id for chunk_document in chunk_documents],
            )
        with timings.measure("fusion_context"):
            fused = fuse_chunk_hits(chunk_documents_by_id, dense_results, bm25_results)
            context_units = [
                context_unit(
                    settings,
                    candidate,
                    settings.plan_block_window,
                    include_expanded_blocks=debug,
                )
                for candidate in fused[:settings.plan_final_top_k]
            ]

    if not context_units:
        warnings.append("正文检索没有命中 Dense/BM25 候选")
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
