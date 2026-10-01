"""content planner：先收缩论文范围，再做 dense/BM25 chunk 检索。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.corpus.records import paper_record_keys
from paper_rag.corpus.scope import append_scope_fallback_warnings, resolve_scope_records
from paper_rag.retrieval.chunk_fusion import fuse_chunk_hits
from paper_rag.retrieval.dense.service import search_dense_chunks
from paper_rag.retrieval.evidence import build_content_evidence
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content.context import context_unit
from paper_rag.retrieval.routes.content.retrieval_query import build_content_retrieval_query
from paper_rag.retrieval.routes.content.translation import CloudKeywordTranslator, KeywordTranslatorProtocol
from paper_rag.retrieval.timing import Timings


GROUP_WARNING_LIMIT = 3
GROUP_CONTEXTS_PER_GROUP = 3


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
    retrieval_stats = new_retrieval_stats()
    if route.parse_status == "parse_failed":
        return build_content_evidence(
            route,
            status="parse_failed",
            warnings=warnings,
            scope_records=[],
            context_units=[],
            retrieval=retrieval_summary(retrieval_stats, context_units=[], scope_records=[]),
            parser_error=route.parser_error,
            debug=debug,
        )

    append_scope_fallback_warnings(warnings, route.filters, route.paper_groups)
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
            retrieval=retrieval_summary(retrieval_stats, context_units=[], scope_records=[]),
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

    if route.group_mode in {"per", "and"} and group_results:
        context_units = []
        for group in group_results:
            group_contexts = retrieve_context_units(
                settings,
                retrieval_query,
                group.get("records") or [],
                warnings,
                corpus=corpus,
                timings=timings,
                embedder=embedder,
                store=store,
                retrieval_stats=retrieval_stats,
            )
            group["context_units"] = group_contexts
            group["exists"] = bool(group_contexts)
            if not group_contexts:
                warnings.append(f"content {route.group_mode} 分组没有命中正文证据：{format_group_scope(group)}")
        final_limit = max(settings.plan_final_top_k, len(group_results) * GROUP_CONTEXTS_PER_GROUP)
        context_units = merge_group_contexts(group_results, final_limit)
        if route.group_mode == "and" and route.intent == "exists" and any(not group.get("exists") for group in group_results):
            warnings.append("content and 判断存在缺失分组证据，不能确认所有分组都满足条件")
    else:
        context_units = retrieve_context_units(
            settings,
            retrieval_query,
            scope_records,
            warnings,
            corpus=corpus,
            timings=timings,
            embedder=embedder,
            store=store,
            retrieval_stats=retrieval_stats,
        )

    if not context_units:
        warnings.append("正文检索没有命中 Dense/BM25 候选")
    retrieval = retrieval_summary(retrieval_stats, context_units=context_units, scope_records=scope_records)
    return build_content_evidence(
        route,
        status="ok",
        warnings=warnings,
        scope_records=scope_records,
        context_units=context_units,
        retrieval_query=retrieval_query,
        retrieval=retrieval,
        group_results=group_results or None,
        debug=debug,
    )


def retrieve_context_units(
    settings: Settings,
    retrieval_query: dict[str, Any],
    scope_records: list[dict[str, Any]],
    warnings: list[str],
    *,
    corpus: CorpusContext,
    timings: Timings,
    embedder=None,
    store=None,
    retrieval_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """在一组论文 scope 内执行一次 Dense/BM25/RRF/context 扩展。"""
    with timings.measure("load_chunks"):
        chunk_documents = corpus.content_chunks_for_records(scope_records)
    if not chunk_documents:
        return []
    chunk_documents_by_id = {
        chunk_document.chunk_id: chunk_document
        for chunk_document in chunk_documents
    }
    stats = retrieval_stats if retrieval_stats is not None else new_retrieval_stats()
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
        stats["dense_available"] = False
        warnings.append(f"Dense 检索失败：{exc}；仅使用 BM25 候选")
    bm25_results = []
    if retrieval_query["bm25_queries"]:
        try:
            with timings.measure("bm25"):
                bm25_results = corpus.bm25_index.search_many(
                    retrieval_query["bm25_queries"],
                    settings.plan_bm25_top_k,
                    allowed_chunk_ids=[chunk_document.chunk_id for chunk_document in chunk_documents],
                )
        except Exception as exc:
            stats["bm25_available"] = False
            warnings.append(f"BM25 检索失败：{exc}")
    stats["dense_hits"] += len(dense_results)
    stats["bm25_hits"] += len(bm25_results)
    with timings.measure("fusion_context"):
        fused = fuse_chunk_hits(chunk_documents_by_id, dense_results, bm25_results)
        stats["fused_hits"] += len(fused)
        fused = filter_required_term_candidates(fused, retrieval_query.get("required_terms") or [])
        if retrieval_query.get("required_terms") and not fused:
            stats["required_terms_covered"] = False
        for candidate in fused:
            stats["unique_papers"].add(candidate.chunk_document.paper_id)
        return [
            context_unit(
                settings,
                candidate,
                settings.plan_block_window,
                include_expanded_blocks=True,
            )
            for candidate in fused[:settings.plan_final_top_k]
        ]


def new_retrieval_stats() -> dict[str, Any]:
    return {
        "dense_available": True,
        "bm25_available": True,
        "dense_hits": 0,
        "bm25_hits": 0,
        "fused_hits": 0,
        "unique_papers": set(),
        "required_terms_covered": True,
    }


def retrieval_summary(
    stats: dict[str, Any],
    *,
    context_units: list[dict[str, Any]],
    scope_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_hits = int(stats.get("dense_hits", 0))
    bm25_hits = int(stats.get("bm25_hits", 0))
    fused_hits = int(stats.get("fused_hits", 0))
    if not scope_records:
        status = "insufficient"
    elif not context_units and not stats.get("dense_available", True) and not stats.get("bm25_available", True):
        status = "backend_unavailable"
    elif not context_units or not stats.get("required_terms_covered", True):
        status = "insufficient"
    else:
        status = "sufficient"
    return {
        "status": status,
        "dense_hits": dense_hits,
        "bm25_hits": bm25_hits,
        "fused_hits": fused_hits,
        "unique_papers": len(stats.get("unique_papers") or set()),
        "required_terms_covered": bool(stats.get("required_terms_covered", True)),
    }


def filter_required_term_candidates(candidates: list[Any], required_terms: list[Any]) -> list[Any]:
    terms = [str(term or "").strip().casefold() for term in required_terms if str(term or "").strip()]
    if not terms:
        return candidates
    return [candidate for candidate in candidates if candidate_matches_required_terms(candidate, terms)]


def candidate_matches_required_terms(candidate: Any, terms: list[str]) -> bool:
    chunk_document = candidate.chunk_document
    haystack = "\n".join([
        chunk_document.title,
        chunk_document.section_path_text,
        chunk_document.text,
        chunk_document.embedding_text,
    ]).casefold()
    return all(term in haystack for term in terms)


def merge_group_contexts(group_results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """按组轮询合并 contexts，避免某个 group 独占最终 top-k。"""
    if limit <= 0:
        return []
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    groups = [list(group.get("context_units") or []) for group in group_results]
    rank = 0
    while len(merged) < limit and any(rank < len(group) for group in groups):
        for group in groups:
            if rank >= len(group):
                continue
            context = group[rank]
            chunk_id = str(context.get("chunk_id") or "")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            merged.append(context)
            if len(merged) >= limit:
                break
        rank += 1
    return merged


def format_group_scope(group: dict[str, Any]) -> str:
    terms = [str(group.get("semantic") or "").strip()]
    terms.extend(format_filter_value(filter_item) for filter_item in group.get("filters") or [])
    compacted = [term for term in terms if term]
    return "；".join(compacted[:GROUP_WARNING_LIMIT]) or "未命名分组"


def format_filter_value(filter_item: dict[str, Any]) -> str:
    field = str(filter_item.get("field") or "").strip()
    op = str(filter_item.get("op") or "").strip()
    value = filter_item.get("value")
    return f"{field}{op}{value}".strip()
