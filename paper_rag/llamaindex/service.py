"""论文库唯一正文 RAG 服务和 LlamaIndex 索引生命周期。"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Iterable

from paper_rag.catalog.service import CatalogIndexNotReady, get_metadata, list_chunks, search_catalog
from paper_rag.config import Settings
from paper_rag.llamaindex.index import IndexService, LlamaIndexError
from paper_rag.llamaindex.nodes import node_metadata
from paper_rag.llamaindex.retrievers import HybridRetriever
from paper_rag.routing import RetrieveDecision, RetrieveTask, RouteIntent, classify_retrieve


_BODY_REGIONS = ("abstract", "content", "appendix")
_FILTER_KEYS = {"author", "category", "year", "year_from", "year_to", "state"}


def search(settings: Settings, query: str, filters: dict[str, Any] | None = None, limit: int = 20) -> dict[str, Any]:
    """只搜索论文元数据，不读取正文。"""
    try:
        records = search_catalog(settings, query, _normalize_filters(filters), _limit(limit, 100))
    except ValueError as exc:
        return _envelope("invalid_input", {"query": query, "items": [], "count": 0}, [str(exc)])
    except CatalogIndexNotReady:
        return _envelope("index_not_ready", {"query": query, "items": [], "count": 0})
    return _envelope("ok", {"query": query, "items": [record.to_dict() for record in records], "count": len(records)})


def retrieve(settings: Settings, query: str, paper_ids: list[str] | None = None, *, filters: dict[str, Any] | None = None, task: str = "auto", limit: int = 8, mode: str = "hybrid", regions: list[str] | None = None, max_chars: int = 24000) -> dict[str, Any]:
    """执行唯一正文 RAG 入口；JEV 只在 task=auto 时细分正文任务。"""
    normalized_mode = str(mode).casefold()
    if normalized_mode not in {"lexical", "semantic", "hybrid"}:
        return _envelope("invalid_input", _empty_payload(query), ["mode must be lexical, semantic or hybrid"])
    try:
        selected_regions = _normalize_regions(regions)
        normalized_filters = _normalize_filters(filters)
        bounded_limit = _limit(limit, 50)
        bounded_chars = max(1, int(max_chars))
    except (TypeError, ValueError) as exc:
        return _envelope("invalid_input", _empty_payload(query), [str(exc)])
    try:
        decision = classify_retrieve(settings, query, paper_ids) if str(task).casefold() == "auto" else _explicit_task(task)
    except ValueError as exc:
        return _envelope("invalid_input", _empty_payload(query), [str(exc)])
    try:
        candidates = _resolve_candidates(settings, paper_ids, normalized_filters)
        if decision.task in {RetrieveTask.SUMMARY, RetrieveTask.COMPARISON} and candidates is None:
            candidates = _discover_candidates(settings, query, bounded_limit)
        if candidates is not None and not candidates:
            return _envelope("not_found", {**_empty_payload(query), "task": decision.task.value}, ["no papers match the supplied constraints"])
        if decision.task is RetrieveTask.COMPARISON and len(candidates or []) < 2:
            return _envelope("invalid_input", {**_empty_payload(query), "task": decision.task.value}, ["comparison requires at least two unique papers"])
    except CatalogIndexNotReady:
        return _envelope("catalog_not_ready", {**_empty_payload(query), "task": decision.task.value})

    papers = _paper_records(settings, candidates)
    warnings = [decision.warning] if decision.warning else []
    status = "ok"
    try:
        if decision.task in {RetrieveTask.SUMMARY, RetrieveTask.COMPARISON}:
            items = _ordered_sources(settings, candidates or (), bounded_limit, query)
            for item in items:
                item["evidence_role"] = "direct"
        else:
            items, retrieve_warnings = _retrieve_items(settings, query, candidates, selected_regions, normalized_mode, bounded_limit)
            warnings.extend(retrieve_warnings)
            if decision.task is RetrieveTask.REASON:
                items = _reason_items(settings, items, bounded_limit)
            status = _warning_status(retrieve_warnings)
    except CatalogIndexNotReady:
        return _envelope("catalog_not_ready", {**_empty_payload(query), "task": decision.task.value})
    except LlamaIndexError as exc:
        warnings.extend([f"semantic_unavailable:{type(exc).__name__}", "lexical_fallback"])
        try:
            items, lexical_warnings = _retrieve_items(settings, query, candidates, selected_regions, "lexical", bounded_limit)
            warnings.extend(lexical_warnings)
        except CatalogIndexNotReady:
            return _envelope("catalog_not_ready", {**_empty_payload(query), "task": decision.task.value})
        status = _index_error_status(exc)
    if not papers:
        papers = _paper_records(settings, _unique_ids(item.get("paper_id") for item in items))
    _assign_source_ids(items)
    context_text, truncated = _render_context(items, bounded_chars, decision.task is RetrieveTask.COMPARISON)
    return _envelope(status, {"query": query, "task": decision.task.value, "routing": decision.to_dict(), "papers": [record.to_dict() for record in papers], "items": items, "count": len(items), "context_text": context_text, "truncated": truncated}, warnings)


def _retrieve_items(settings: Settings, query: str, candidates: list[str] | None, regions: tuple[str, ...], mode: str, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    """执行统一的 lexical/semantic/hybrid 证据召回。"""
    service = _get_index_service(settings)
    index = None if mode == "lexical" else service.load()
    semantic = _build_semantic_retriever(index, candidates, regions, max(settings.llamaindex_semantic_top_k, limit * 5, 50)) if index else None
    retriever = HybridRetriever(settings, semantic, paper_ids=candidates, regions=regions, mode=mode, lexical_top_k=settings.llamaindex_lexical_top_k, semantic_top_k=settings.llamaindex_semantic_top_k, rrf_k=settings.llamaindex_rrf_k)
    nodes = retriever.retrieve(query)[:limit]
    items: list[dict[str, Any]] = []
    for result in nodes:
        item = node_metadata(result.node)
        item.update({"score": result.score, "semantic_score": result.node.metadata.get("semantic_score"), "lexical_rank": result.node.metadata.get("lexical_rank"), "semantic_rank": result.node.metadata.get("semantic_rank"), "rrf_score": result.node.metadata.get("rrf_score", result.score), "page_start_display": _display_page(item.get("page_start")), "page_end_display": _display_page(item.get("page_end")), "evidence_role": "direct"})
        items.append(item)
    return items, list(retriever.warnings)


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


def _discover_candidates(settings: Settings, query: str, limit: int) -> list[str]:
    """summary/comparison 无显式范围时，使用元数据索引确定论文集合。"""
    records = search_catalog(settings, query, {}, max(2, min(limit, 20)))
    return _unique_ids(record.base_id for record in records)


def _paper_records(settings: Settings, paper_ids: Iterable[str] | None) -> list[Any]:
    result: list[Any] = []
    for paper_id in _unique_ids(paper_ids or ()):
        record = get_metadata(settings, paper_id)
        if record:
            result.append(record)
    return result


def _ordered_sources(settings: Settings, paper_ids: Iterable[str], limit: int, query: str) -> list[dict[str, Any]]:
    """为 summary/comparison 按论文保留摘要和正文 Chunk。"""
    ids = _unique_ids(paper_ids)
    per_paper = max(1, math.ceil(limit / max(1, len(ids))))
    selected: list[dict[str, Any]] = []
    for paper_id in ids:
        chunks = [item for item in list_chunks(settings, paper_id, 10000) if item.get("region") in _BODY_REGIONS]
        if not chunks:
            continue
        abstract = [item for item in chunks if item.get("region") == "abstract"]
        body = [item for item in chunks if item.get("region") != "abstract"]
        chosen = (abstract[:1] + body)[:per_paper] or chunks[:1]
        for item in chosen:
            selected.append({**item, "score": None, "semantic_score": None, "lexical_rank": None, "semantic_rank": None, "rrf_score": None, "page_start_display": _display_page(item.get("page_start")), "page_end_display": _display_page(item.get("page_end")), "query": query})
    return selected[: max(limit, len(ids))]


def _reason_items(settings: Settings, direct: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """为原因类问题补充同章节和相邻 Chunk，仍然只返回证据。"""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    direct_limit = max(1, math.ceil(limit / 2))
    for item in direct[:direct_limit]:
        key = str(item.get("chunk_id"))
        if key not in seen:
            result.append(item)
            seen.add(key)
        chunks = list_chunks(settings, str(item.get("paper_id")), 10000)
        ordinal = int(item.get("ordinal") or 0)
        for neighbor in chunks:
            if neighbor.get("region") not in _BODY_REGIONS or abs(int(neighbor.get("ordinal") or 0) - ordinal) > 1:
                continue
            neighbor_id = str(neighbor.get("chunk_id"))
            if neighbor_id in seen:
                continue
            if len(result) >= limit:
                return result[:limit]
            result.append({**neighbor, "score": None, "semantic_score": None, "lexical_rank": None, "semantic_rank": None, "rrf_score": None, "evidence_role": "context", "page_start_display": _display_page(neighbor.get("page_start")), "page_end_display": _display_page(neighbor.get("page_end"))})
            seen.add(neighbor_id)
            if len(result) >= limit:
                return result
    return result[:limit]


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


def _normalize_regions(regions: list[str] | None) -> tuple[str, ...]:
    selected = tuple(str(value) for value in (regions or _BODY_REGIONS) if str(value))
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
    return "milvus_unavailable" if "Milvus" in str(error) else "embedding_unavailable"


def _warning_status(warnings: Iterable[str]) -> str:
    """将语义层降级 warning 映射为稳定的 MCP 状态。"""

    for warning in warnings:
        if str(warning).startswith("semantic_unavailable:"):
            return "milvus_unavailable" if "Milvus" in str(warning) else "embedding_unavailable"
    return "ok"


def _empty_payload(query: str) -> dict[str, Any]:
    return {"query": query, "task": None, "papers": [], "items": [], "count": 0, "context_text": "", "truncated": False}


def index_status(settings: Settings) -> dict[str, Any]:
    return _envelope("ok", _get_index_service(settings).status())


def rebuild_index(settings: Settings) -> dict[str, Any]:
    try:
        result = _get_index_service(settings).rebuild()
        _get_index_service.cache_clear()
        return _envelope("completed", result)
    except CatalogIndexNotReady:
        return _envelope("catalog_not_ready", {})
    except LlamaIndexError as exc:
        return _envelope("failed", {}, [str(exc)])


def _envelope(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "data": data, "warnings": [item for item in (warnings or []) if item], "read_only": status not in {"completed"}}


@lru_cache(maxsize=4)
def _get_index_service(settings: Settings) -> IndexService:
    return IndexService(settings)


__all__ = ["index_status", "rebuild_index", "retrieve", "search"]
