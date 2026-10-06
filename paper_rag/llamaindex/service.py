"""论文库唯一正文 RAG 服务和 LlamaIndex 索引生命周期。"""

from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any, Iterable

from paper_rag.catalog.service import CatalogIndexNotReady, get_metadata, list_chunks, search_catalog, search_chunks
from paper_rag.answering import ANSWER_CONTEXTS, answer_contract
from paper_rag.config import Settings
from paper_rag.lexical import build_fts_query
from paper_rag.llamaindex.index import IndexService, LlamaIndexError
from paper_rag.llamaindex.nodes import node_metadata
from paper_rag.llamaindex.retrievers import HybridRetriever
from paper_rag.llamaindex.translation import prepare_lexical_query
from paper_rag.presentation import attach_presentation, retrieve_presentation, search_presentation
from paper_rag.routing import RetrieveDecision, RetrieveTask, RouteIntent, classify_retrieve


_BODY_REGIONS = ("abstract", "content", "appendix")
_FILTER_KEYS = {"author", "category", "year", "year_from", "year_to", "state"}
_DEFAULT_REGIONS = ("abstract", "content")
_APPENDIX_QUERY_RE = re.compile(r"(?:\bappendi(?:x|ces)\b|\bsupplement(?:ary|al)?\b|附录)", re.IGNORECASE)
_TECH_ENTITY_RE = re.compile(r"\b(?:[A-Za-z]+[A-Z][A-Za-z0-9.-]*|[A-Z]{2,}[A-Za-z0-9.-]*|[A-Za-z]+[0-9][A-Za-z0-9.-]*)\b")
_TABLE_REF_RE = re.compile(r"(?:\btable\s*|\btab\.?\s*|表\s*)(\d+)", re.IGNORECASE)
_FIGURE_REF_RE = re.compile(r"(?:\bfigure\s*|\bfig\.?\s*|图\s*)(\d+)", re.IGNORECASE)
_GENERIC_ENTITIES = {"attention", "transformer", "transformers", "model", "models", "image", "images", "network", "networks", "method", "methods"}


def search(settings: Settings, query: str, filters: dict[str, Any] | None = None, limit: int = 20) -> dict[str, Any]:
    """只搜索论文元数据，不读取正文。"""
    prepared = prepare_lexical_query(query, settings)
    try:
        records = search_catalog(
            settings,
            prepared.query,
            _normalize_filters(filters),
            _limit(limit, 100),
            fts_query=prepared.fts_query,
        )
    except ValueError as exc:
        return _search_response("invalid_input", {"query": query, "items": [], "count": 0, "query_debug": prepared.debug()}, [*prepared.warnings, str(exc)])
    except CatalogIndexNotReady:
        return _search_response("index_not_ready", {"query": query, "items": [], "count": 0, "query_debug": prepared.debug()}, list(prepared.warnings))
    return _search_response("ok", {"query": query, "items": [record.to_dict() for record in records], "count": len(records), "query_debug": prepared.debug()}, list(prepared.warnings))


def retrieve(settings: Settings, query: str, paper_ids: list[str] | None = None, *, filters: dict[str, Any] | None = None, task: str = "auto", limit: int = 8, mode: str = "hybrid", regions: list[str] | None = None, max_chars: int = 24000) -> dict[str, Any]:
    """执行唯一正文 RAG 入口；JEV 只在 task=auto 时细分正文任务。"""
    normalized_mode = str(mode).casefold()
    if normalized_mode not in {"lexical", "semantic", "hybrid"}:
        return _retrieve_response("invalid_input", _empty_payload(query), ["mode must be lexical, semantic or hybrid"], normalized_mode)
    try:
        selected_regions = _normalize_regions(regions, query)
        normalized_filters = _normalize_filters(filters)
        bounded_limit = _limit(limit, 50)
        bounded_chars = max(1, int(max_chars))
    except (TypeError, ValueError) as exc:
        return _retrieve_response("invalid_input", _empty_payload(query), [str(exc)], normalized_mode)
    try:
        decision = classify_retrieve(settings, query, paper_ids) if str(task).casefold() == "auto" else _explicit_task(task)
    except ValueError as exc:
        return _retrieve_response("invalid_input", _empty_payload(query), [str(exc)], normalized_mode)
    candidate_debug: dict[str, Any] = {}
    try:
        candidates = _resolve_candidates(settings, paper_ids, normalized_filters)
        if candidates is None:
            candidates, candidate_debug = _discover_candidates(settings, query, bounded_limit, decision.task.value, normalized_filters)
        if candidates is not None and not candidates:
            return _retrieve_response("not_found", {**_empty_payload(query), "task": decision.task.value, "retrieval_debug": candidate_debug}, ["no papers match the supplied constraints"], normalized_mode)
        if decision.task is RetrieveTask.COMPARISON and len(candidates or []) < 2:
            return _retrieve_response("invalid_input", {**_empty_payload(query), "task": decision.task.value}, ["comparison requires at least two unique papers"], normalized_mode)
    except CatalogIndexNotReady:
        return _retrieve_response("catalog_not_ready", {**_empty_payload(query), "task": decision.task.value}, [], normalized_mode)

    papers = _paper_records(settings, candidates)
    warnings = [decision.warning] if decision.warning else []
    status = "ok"
    retrieval_debug: dict[str, Any] = {}
    retrieval_debug.update(candidate_debug)
    try:
        if decision.task in {RetrieveTask.SUMMARY, RetrieveTask.COMPARISON}:
            search_limit = max(bounded_limit, min(50, bounded_limit * max(2, len(candidates or ()))))
            ranked_items, retrieve_warnings, stage_debug = _retrieve_items(
                settings,
                query,
                candidates,
                selected_regions,
                normalized_mode,
                search_limit,
                task=decision.task.value,
            )
            warnings.extend(retrieve_warnings)
            retrieval_debug.update(stage_debug)
            items = _ordered_sources(
                settings,
                candidates or (),
                bounded_limit,
                query,
                selected_regions,
                decision.task.value,
                ranked_items,
            )
            status = _warning_status(retrieve_warnings)
        else:
            items, retrieve_warnings, stage_debug = _retrieve_items(settings, query, candidates, selected_regions, normalized_mode, bounded_limit, task=decision.task.value)
            warnings.extend(retrieve_warnings)
            retrieval_debug.update(stage_debug)
            if decision.task is RetrieveTask.REASON:
                items = _reason_items(settings, items, bounded_limit)
            status = _warning_status(retrieve_warnings)
    except CatalogIndexNotReady:
        return _retrieve_response("catalog_not_ready", {**_empty_payload(query), "task": decision.task.value}, [], normalized_mode)
    except LlamaIndexError as exc:
        warnings.extend([f"semantic_unavailable:{type(exc).__name__}", "lexical_fallback"])
        try:
            fallback_limit = max(bounded_limit, min(50, bounded_limit * max(2, len(candidates or ())))) if decision.task in {RetrieveTask.SUMMARY, RetrieveTask.COMPARISON} else bounded_limit
            ranked_items, lexical_warnings, lexical_debug = _retrieve_items(settings, query, candidates, selected_regions, "lexical", fallback_limit, task=decision.task.value)
            warnings.extend(lexical_warnings)
            retrieval_debug.update(lexical_debug)
            if decision.task in {RetrieveTask.SUMMARY, RetrieveTask.COMPARISON}:
                items = _ordered_sources(settings, candidates or (), bounded_limit, query, selected_regions, decision.task.value, ranked_items)
            else:
                items = ranked_items[:bounded_limit]
                if decision.task is RetrieveTask.REASON:
                    items = _reason_items(settings, items, bounded_limit)
        except CatalogIndexNotReady:
            return _retrieve_response("catalog_not_ready", {**_empty_payload(query), "task": decision.task.value}, [], normalized_mode)
        status = _index_error_status(exc)
    if not papers:
        papers = _paper_records(settings, _unique_ids(item.get("paper_id") for item in items))
    _assign_source_ids(items)
    context_text, truncated = _render_context(items, bounded_chars, decision.task is RetrieveTask.COMPARISON)
    answer_context = ANSWER_CONTEXTS.create(
        query=query,
        task=decision.task.value,
        mode=normalized_mode,
        filters=normalized_filters,
        regions=selected_regions,
        items=items,
        context_text=context_text,
        truncated=truncated,
    )
    data = {
        "query": query,
        "task": decision.task.value,
        "routing": decision.to_dict(),
        "papers": [record.to_dict() for record in papers],
        "items": items,
        "count": len(items),
        "context_text": context_text,
        "truncated": truncated,
        "retrieval_debug": retrieval_debug,
        "answer_context_id": answer_context.context_id,
        "answer_contract": answer_contract(),
        "citation_registry": answer_context.citation_registry,
    }
    return _retrieve_response(status, data, warnings, normalized_mode)


def _retrieve_items(settings: Settings, query: str, candidates: list[str] | None, regions: tuple[str, ...], mode: str, limit: int, *, task: str = "fact") -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """执行统一的 lexical/semantic/hybrid 证据召回。"""
    service = _get_index_service(settings)
    index = None if mode == "lexical" else service.load()
    semantic = _build_semantic_retriever(index, candidates, regions, max(settings.llamaindex_semantic_top_k, limit * 5, 50)) if index else None
    retriever = HybridRetriever(settings, semantic, paper_ids=candidates, regions=regions, mode=mode, task=task, lexical_top_k=settings.llamaindex_lexical_top_k, semantic_top_k=settings.llamaindex_semantic_top_k, rrf_k=settings.llamaindex_rrf_k)
    nodes = retriever.retrieve(query)[:limit]
    items: list[dict[str, Any]] = []
    for result in nodes:
        item = node_metadata(result.node)
        item.update({"score": result.score, "semantic_score": result.node.metadata.get("semantic_score"), "lexical_rank": result.node.metadata.get("lexical_rank"), "semantic_rank": result.node.metadata.get("semantic_rank"), "rrf_score": result.node.metadata.get("rrf_score", result.score), "ranking_features": result.node.metadata.get("ranking_features", {}), "page_start_display": _display_page(item.get("page_start")), "page_end_display": _display_page(item.get("page_end")), "evidence_role": "direct"})
        items.append(item)
    return items, list(retriever.warnings), dict(retriever.lexical_debug)


def _build_semantic_retriever(index: Any, paper_ids: list[str] | None, regions: tuple[str, ...], top_k: int) -> Any:
    """为 Milvus 语义检索构造与 lexical 相同的 metadata 约束。"""

    from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

    filters = [MetadataFilter(key="region", value=list(regions), operator=FilterOperator.IN)]
    if paper_ids:
        filters.append(MetadataFilter(key="paper_id", value=list(paper_ids), operator=FilterOperator.IN))
    return index.as_retriever(similarity_top_k=top_k, filters=MetadataFilters(filters=filters, condition=FilterCondition.AND))


def _explicit_task(task: str) -> RetrieveDecision:
    try:
        return RetrieveDecision(RouteIntent.RETRIEVE, RetrieveTask(str(task).casefold()), provider="explicit")
    except ValueError as exc:
        raise ValueError("task must be auto, fact, reason, summary or comparison") from exc


def _resolve_candidates(settings: Settings, paper_ids: list[str] | None, filters: dict[str, Any]) -> list[str] | None:
    requested = _unique_ids(paper_ids or ())
    if not requested and not filters:
        return None
    records = search_catalog(settings, "", filters, 100000)
    by_id = {record.base_id.casefold(): record for record in records}
    by_canonical = {record.canonical_id.casefold(): record for record in records}
    if requested:
        result: list[str] = []
        for value in requested:
            record = by_id.get(value.casefold()) or by_canonical.get(value.casefold()) or get_metadata(settings, value)
            if record and (not filters or record.base_id.casefold() in by_id) and record.base_id not in result:
                result.append(record.base_id)
        return result
    return [record.base_id for record in records]


def _discover_candidates(settings: Settings, query: str, limit: int, task: str, filters: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """按元数据、实体和正文证据分层发现候选论文。"""
    prepared = prepare_lexical_query(query, settings)
    max_candidates = max(2, min(limit, 20))
    records = search_catalog(
        settings,
        prepared.query,
        filters,
        max_candidates,
        fts_query=prepared.fts_query,
    )
    metadata_ids = _unique_ids(record.base_id for record in records)
    allowed_ids: list[str] | None = None
    if filters:
        allowed_ids = [record.base_id for record in search_catalog(settings, "", filters, 100000)]

    entities = _query_entities(query, prepared.core_terms)
    entity_hits: dict[str, list[str]] = {}
    entity_ids: list[str] = []
    entity_record_map: dict[str, Any] = {}
    for entity in entities:
        matched_records = search_catalog(
            settings,
            entity,
            filters,
            max_candidates,
            fts_query=build_fts_query(entity),
        )
        ids = _unique_ids(record.base_id for record in matched_records)
        if ids:
            entity_hits[entity] = ids
            entity_ids.extend(ids)
            entity_record_map.update({record.base_id: record for record in matched_records})
    entity_ids = _unique_ids(entity_ids)

    chunk_items: list[dict[str, Any]] = []
    table_ref = _query_reference(query, _TABLE_REF_RE)
    figure_ref = _query_reference(query, _FIGURE_REF_RE)
    focus_terms = _focus_terms(query, prepared.core_terms, entities)
    if not metadata_ids or task in {RetrieveTask.FACT.value, RetrieveTask.REASON.value} or table_ref or figure_ref:
        chunk_items = search_chunks(
            settings,
            prepared.query,
            allowed_ids,
            max_candidates * 5,
            _candidate_regions(query),
            fts_query=prepared.fts_query,
        )
    chunk_ids = _unique_ids(item.get("paper_id") for item in chunk_items)
    chunk_focus_ids = _unique_ids(
        item.get("paper_id")
        for item in chunk_items
        if _chunk_focus_hit(item, focus_terms, table_ref, figure_ref)
    )
    title_exact_ids = [
        record.base_id
        for record in [*records, *entity_record_map.values()]
        if any(_term_hit(str(getattr(record, "title", "") or ""), entity) for entity in entities)
    ]
    abstract_exact_ids = [
        record.base_id
        for record in [*records, *entity_record_map.values()]
        if any(_term_hit(str(getattr(record, "abstract", "") or ""), entity) for entity in entities)
    ]
    title_exact_ids = _unique_ids(title_exact_ids)
    abstract_exact_ids = _unique_ids(abstract_exact_ids)

    if title_exact_ids:
        selected = list(title_exact_ids)
    elif chunk_focus_ids:
        selected = list(chunk_focus_ids)
    elif entity_ids:
        selected = list(entity_ids)
    else:
        selected = list(metadata_ids) + list(chunk_ids)

    if task == RetrieveTask.COMPARISON.value:
        selected = _unique_ids(title_exact_ids + entity_ids + metadata_ids)
        if len(selected) < 2:
            selected.extend(chunk_focus_ids or chunk_ids)
    elif task == RetrieveTask.SUMMARY.value:
        selected = selected[:1]
    elif task in {RetrieveTask.FACT.value, RetrieveTask.REASON.value} and chunk_focus_ids:
        selected = _unique_ids(chunk_focus_ids + selected)
    selected = _unique_ids(selected)[:max_candidates]
    debug = {
        "candidate_discovery": {
            "metadata_count": len(metadata_ids),
            "chunk_count": len(chunk_ids),
            "entity_hits": entity_hits,
            "fallback_used": not metadata_ids and bool(chunk_items),
            "chunk_search_used": bool(chunk_items),
            "selected_paper_ids": selected,
            "lexical_query": prepared.fts_query,
            "candidate_match_source": {
                paper_id: _candidate_match_source(paper_id, title_exact_ids, abstract_exact_ids, entity_ids, chunk_focus_ids, metadata_ids)
                for paper_id in selected
            },
            "title_exact_hit": title_exact_ids,
            "abstract_exact_hit": abstract_exact_ids,
            "chunk_exact_hit": chunk_focus_ids,
            "table_ref": table_ref,
            "figure_ref": figure_ref,
        }
    }
    return selected, debug


def _paper_records(settings: Settings, paper_ids: Iterable[str] | None) -> list[Any]:
    result: list[Any] = []
    for paper_id in _unique_ids(paper_ids or ()):
        record = get_metadata(settings, paper_id)
        if record:
            result.append(record)
    return result


def _ordered_sources(settings: Settings, paper_ids: Iterable[str], limit: int, query: str, regions: tuple[str, ...], task: str, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为 summary/comparison 平衡论文来源并优先保留相关证据。"""
    ids = _unique_ids(paper_ids)
    desired = max(limit, len(ids))
    per_paper = max(1, math.ceil(desired / max(1, len(ids))))
    ranked_by_paper: dict[str, list[dict[str, Any]]] = {}
    for item in ranked:
        ranked_by_paper.setdefault(str(item.get("paper_id")), []).append(item)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for paper_id in ids:
        chunks = [item for item in list_chunks(settings, paper_id, 10000) if item.get("region") in regions]
        if not chunks:
            continue
        candidates = list(ranked_by_paper.get(paper_id, ()))
        abstract = [item for item in chunks if item.get("region") == "abstract"]
        fallback = abstract[:1] or chunks[:1]
        ordered = _source_order(candidates, task)
        if fallback and not any(item.get("chunk_id") == fallback[0].get("chunk_id") for item in ordered):
            ordered = fallback + ordered
        chosen = ordered[:per_paper] or fallback
        for item in chosen:
            chunk_id = str(item.get("chunk_id"))
            if chunk_id in used:
                continue
            used.add(chunk_id)
            selected.append({"score": None, "semantic_score": None, "lexical_rank": None, "semantic_rank": None, "rrf_score": None, "ranking_features": {}, "page_start_display": _display_page(item.get("page_start")), "page_end_display": _display_page(item.get("page_end")), "query": query, "evidence_role": "direct", **item})
    return selected[:desired]


def _reason_items(settings: Settings, direct: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """为原因类问题补充完整的同章节证据窗口。"""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_text: set[str] = set()
    direct_limit = max(1, math.ceil(limit / 2))
    for item in direct[:direct_limit]:
        chunks = list_chunks(settings, str(item.get("paper_id")), 10000)
        ordinal = int(item.get("ordinal") or 0)
        same_section = [
            chunk for chunk in chunks
            if chunk.get("region") in _BODY_REGIONS
            and chunk.get("section_label") == item.get("section_label")
        ]
        nearby = [chunk for chunk in same_section if abs(int(chunk.get("ordinal") or 0) - ordinal) <= 1]
        window = _build_evidence_window(item, same_section, nearby)
        source_ids = window.get("source_chunk_ids", [])
        if item.get("type") == "image" and window.get("text") and len(result) < limit:
            window_text = _normalize_evidence_text(window.get("text"))
            if window_text not in seen_text:
                result.append(window)
                seen_text.add(window_text)
            seen.update(str(value) for value in source_ids)
            if len(result) >= limit:
                return result[:limit]
            image_id = str(item.get("chunk_id"))
            if image_id not in seen and len(result) < limit:
                result.append(item)
                seen.add(image_id)
        else:
            key = str(item.get("chunk_id"))
            text = window.get("text") or item.get("text")
            text_key = _normalize_evidence_text(text)
            if key not in seen and text_key not in seen_text and len(result) < limit:
                enriched = dict(item)
                enriched.update({"window_id": window.get("window_id"), "source_chunk_ids": source_ids, "continuity_status": window.get("continuity_status"), "text": window.get("text") or item.get("text")})
                result.append(enriched)
                seen.update(str(value) for value in source_ids)
                seen_text.add(text_key)
        if len(result) >= limit:
            return result[:limit]
    return result[:limit]


def _normalize_evidence_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _query_entities(query: str, core_terms: Iterable[str]) -> list[str]:
    """提取论文方法名、缩写和模型名，避免候选发现被问题措辞淹没。"""

    values = [str(term) for term in core_terms]
    values.append(str(query or ""))
    entities: list[str] = []
    seen: set[str] = set()
    for value in values:
        for match in _TECH_ENTITY_RE.findall(value):
            entity = match.strip(".,:;()[]{}")
            key = entity.casefold()
            if len(entity) < 3 or key in _GENERIC_ENTITIES or key in seen:
                continue
            seen.add(key)
            entities.append(entity)
    return entities


def _query_reference(query: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(query or ""))
    return match.group(1) if match else None


def _focus_terms(query: str, core_terms: Iterable[str], entities: Iterable[str]) -> list[str]:
    """提取用于正文候选过滤的非通用短语。"""

    values = [str(value).strip() for value in core_terms if str(value).strip()]
    values.extend(str(value).strip() for value in entities if str(value).strip())
    if not values:
        values.append(str(query or "").strip())
    result: list[str] = []
    for value in values:
        normalized = _normalize_match_text(value)
        if len(normalized) < 4 or normalized in _GENERIC_ENTITIES:
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _chunk_focus_hit(item: dict[str, Any], terms: Iterable[str], table_ref: str | None, figure_ref: str | None) -> bool:
    text = _normalize_match_text(f"{item.get('text', '')} {item.get('retrieval_text', '')}")
    if table_ref and re.search(rf"\btable\s*{re.escape(table_ref)}\b|\btab\.?\s*{re.escape(table_ref)}\b|表\s*{re.escape(table_ref)}", text, re.IGNORECASE):
        return True
    if figure_ref and re.search(rf"\bfigure\s*{re.escape(figure_ref)}\b|\bfig\.?\s*{re.escape(figure_ref)}\b|图\s*{re.escape(figure_ref)}", text, re.IGNORECASE):
        return True
    return any(_term_hit(text, term) for term in terms)


def _term_hit(text: str, term: str) -> bool:
    normalized_text = _normalize_match_text(text)
    normalized_term = _normalize_match_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    plural_pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}s(?![a-z0-9])"
    return bool(re.search(pattern, normalized_text) or re.search(plural_pattern, normalized_text))


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/]+", " ", str(value or "").casefold())).strip()


def _candidate_match_source(
    paper_id: str,
    title_exact_ids: Iterable[str],
    abstract_exact_ids: Iterable[str],
    entity_ids: Iterable[str],
    chunk_focus_ids: Iterable[str],
    metadata_ids: Iterable[str],
) -> str:
    if paper_id in set(title_exact_ids):
        return "title_exact"
    if paper_id in set(abstract_exact_ids):
        return "abstract_exact"
    if paper_id in set(chunk_focus_ids):
        return "chunk_exact"
    if paper_id in set(entity_ids):
        return "entity"
    if paper_id in set(metadata_ids):
        return "metadata"
    return "fallback"


def _candidate_regions(query: str) -> tuple[str, ...]:
    """候选发现默认忽略附录，只有问题明确提及时才加入。"""

    return _DEFAULT_REGIONS + (("appendix",) if _APPENDIX_QUERY_RE.search(str(query or "")) else ())


def _source_order(items: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    """在每篇论文内部优先摘要或已排序的相关证据。"""

    def key(item: dict[str, Any]) -> tuple[int, float]:
        abstract_first = 0 if task in {RetrieveTask.SUMMARY.value, RetrieveTask.COMPARISON.value} and item.get("region") == "abstract" else 1
        score = item.get("score")
        if score is None:
            score = item.get("rrf_score")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = -1.0
        return abstract_first, -numeric_score

    return sorted(items, key=key)


def _build_evidence_window(core: dict[str, Any], same_section: list[dict[str, Any]], nearby: list[dict[str, Any]]) -> dict[str, Any]:
    """把同章节邻接 Chunk 合并成可读证据窗口，不改写 Catalog 原文。"""

    ordered = sorted(nearby, key=lambda item: int(item.get("ordinal") or 0))
    positions = {str(item.get("chunk_id")): index for index, item in enumerate(same_section)}
    first = positions.get(str(ordered[0].get("chunk_id"))) if ordered else None
    last = positions.get(str(ordered[-1].get("chunk_id"))) if ordered else None
    while ordered and _starts_mid_token(str(ordered[0].get("text") or "")) and first is not None and first > 0:
        first -= 1
        ordered.insert(0, same_section[first])
    while ordered and _ends_mid_token(str(ordered[-1].get("text") or "")) and last is not None and last + 1 < len(same_section):
        last += 1
        ordered.append(same_section[last])
    text = _merge_chunk_texts(ordered)
    if _starts_mid_token(text):
        text = _trim_leading_fragment(text)
    source_chunk_ids = [str(item.get("chunk_id")) for item in ordered if item.get("chunk_id")]
    pages = [item.get("page_start") for item in ordered if isinstance(item.get("page_start"), int)]
    end_pages = [item.get("page_end") for item in ordered if isinstance(item.get("page_end"), int)]
    result = dict(core)
    result.update(
        {
            "text": text or core.get("text", ""),
            "type": "text",
            "evidence_role": "context",
            "window_id": f"{core.get('paper_id')}:{core.get('ordinal')}",
            "source_chunk_ids": source_chunk_ids or [str(core.get("chunk_id"))],
            "continuity_status": "complete" if text and not _starts_mid_token(text) else "trimmed_prefix",
            "page_start": min(pages) if pages else core.get("page_start"),
            "page_end": max(end_pages) if end_pages else core.get("page_end"),
            "page_start_display": _display_page(min(pages)) if pages else _display_page(core.get("page_start")),
            "page_end_display": _display_page(max(end_pages)) if end_pages else _display_page(core.get("page_end")),
            "score": None,
            "semantic_score": None,
            "lexical_rank": None,
            "semantic_rank": None,
            "rrf_score": None,
            "ranking_features": {},
        }
    )
    return result


def _merge_chunk_texts(chunks: list[dict[str, Any]]) -> str:
    """合并有重叠的相邻文本，避免窗口中重复整段内容。"""

    merged = ""
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        if not merged:
            merged = text
            continue
        overlap = _text_overlap(merged, text)
        merged += text[overlap:] if overlap else f"\n\n{text}"
    return merged.strip()


def _text_overlap(left: str, right: str) -> int:
    """返回左右文本的最大后缀/前缀重叠长度。"""

    left_folded = left.casefold()
    right_folded = right.casefold()
    maximum = min(240, len(left), len(right))
    for size in range(maximum, 23, -1):
        if left_folded[-size:] == right_folded[:size]:
            return size
    return 0


def _starts_mid_token(text: str) -> bool:
    return bool(re.match(r"^[a-z]", text.strip()))


def _ends_mid_token(text: str) -> bool:
    value = text.rstrip()
    return bool(value and value[-1].isalnum())


def _trim_leading_fragment(text: str) -> str:
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z0-9○])", text)
    if match:
        return text[match.end():].lstrip()
    return text


def _render_context(items: list[dict[str, Any]], max_chars: int, separate: bool) -> tuple[str, bool]:
    """按字符预算渲染带来源标记的客户端上下文。"""
    budget = max(1, int(max_chars))
    groups = _unique_ids(item.get("paper_id") for item in items) if separate else ["_all"]
    cap = max(1, budget // max(1, len(groups))) if separate else budget
    used: dict[str, int] = {}
    rendered: list[str] = []
    truncated = False
    for item in items:
        key = str(item.get("paper_id")) if separate else "_all"
        header = f"[{item.get('source_id')}] {item.get('canonical_id')} p.{item.get('page_start_display')}"
        available = cap - used.get(key, 0)
        required = len(header) + 2
        text = str(item.get("text") or "")
        if available <= required:
            truncated = True
            continue
        take = min(len(text), available - required)
        if take < len(text):
            truncated = True
        snippet = text[:take].rstrip()
        if not snippet:
            truncated = True
            continue
        rendered.append(f"{header}: {snippet}")
        used[key] = used.get(key, 0) + required + len(snippet)
    return "\n\n".join(rendered), truncated


def _assign_source_ids(items: list[dict[str, Any]]) -> None:
    for number, item in enumerate(items, start=1):
        item["source_id"] = f"S{number}"


def _normalize_regions(regions: list[str] | None, query: str) -> tuple[str, ...]:
    selected = tuple(str(value) for value in (regions or _candidate_regions(query)) if str(value))
    if not selected or any(value not in _BODY_REGIONS for value in selected):
        raise ValueError("regions may only contain abstract, content or appendix")
    return tuple(dict.fromkeys(selected))


def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object")
    unknown = sorted(set(filters) - _FILTER_KEYS)
    if unknown:
        raise ValueError(f"unsupported filters: {', '.join(unknown)}")
    result = {key: value for key, value in filters.items() if value not in (None, "")}
    for key in ("year", "year_from", "year_to"):
        if key in result and (not str(result[key]).isdigit() or len(str(result[key])) != 4):
            raise ValueError(f"{key} must be a four digit year")
    if "year_from" in result and "year_to" in result and str(result["year_from"]) > str(result["year_to"]):
        raise ValueError("year_from must not be later than year_to")
    return result


def _unique_ids(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _display_page(value: Any) -> int | None:
    return value + 1 if isinstance(value, int) else None


def _limit(value: Any, maximum: int) -> int:
    return max(1, min(int(value), maximum))


def _index_error_status(error: LlamaIndexError) -> str:
    if "not ready" in str(error).casefold() or "stale" in str(error).casefold():
        return "index_not_ready"
    return "milvus_unavailable" if "Milvus" in str(error) else "embedding_unavailable"


def _warning_status(warnings: Iterable[str]) -> str:
    """将语义层降级 warning 映射为稳定的 MCP 状态。"""

    for warning in warnings:
        if str(warning).startswith("semantic_unavailable:"):
            return "milvus_unavailable" if "Milvus" in str(warning) else "embedding_unavailable"
    return "ok"


def _empty_payload(query: str) -> dict[str, Any]:
    return {"query": query, "task": None, "papers": [], "items": [], "count": 0, "context_text": "", "truncated": False, "retrieval_debug": {}}


def index_status(settings: Settings) -> dict[str, Any]:
    return _envelope("ok", _get_index_service(settings).status())


def rebuild_index(settings: Settings, mode: str = "auto") -> dict[str, Any]:
    try:
        result = _get_index_service(settings).rebuild(mode=mode)
        _get_index_service.cache_clear()
        return _envelope("completed", result)
    except CatalogIndexNotReady:
        return _envelope("catalog_not_ready", {})
    except LlamaIndexError as exc:
        status = str(exc.details.get("status") or "failed")
        return _envelope(status, exc.details, [str(exc)])


def _envelope(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "data": data, "warnings": [item for item in (warnings or []) if item], "read_only": status not in {"completed"}}


def _search_response(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    """为元数据查询附加必须原样输出的确定性答案。"""

    payload = _envelope(status, data, warnings)
    return attach_presentation(payload, search_presentation(payload["data"]))


def _retrieve_response(status: str, data: dict[str, Any], warnings: list[str] | None, mode: str) -> dict[str, Any]:
    """标记正文证据由客户端原样使用还是组织生成。"""

    payload = _envelope(status, data, warnings)
    return attach_presentation(payload, retrieve_presentation(payload["data"], mode))


@lru_cache(maxsize=4)
def _get_index_service(settings: Settings) -> IndexService:
    return IndexService(settings)


__all__ = ["index_status", "rebuild_index", "retrieve", "search"]
