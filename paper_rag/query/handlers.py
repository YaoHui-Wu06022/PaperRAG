from __future__ import annotations

from typing import Any

from paper_rag.catalog.service import CatalogIndexNotReady, search_catalog
from paper_rag.config import Settings
from paper_rag.query.schemas import QueryIntent, QueryIntentDecision, QueryRequest
from paper_rag.reading.service import context_chunks, get_context, search_content


def execute_query(settings: Settings, request: QueryRequest, decision: QueryIntentDecision) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, dict[str, Any]]:
    if decision.needs_clarification or decision.intent == QueryIntent.CLARIFY:
        return [], [], "请补充论文 ID、主题或想了解的具体内容。", {}
    if decision.intent == QueryIntent.PAPER_DISCOVERY:
        try: records = search_catalog(settings, request.query, limit=50)
        except CatalogIndexNotReady:
            return [], [], "论文 Catalog 尚未同步，请先执行 paper_catalog_sync。", {"index_ready": False, "status": "index_not_ready"}
        return [record.to_dict() for record in records], [], None, {"index_ready": True}
    if decision.intent == QueryIntent.METADATA_LOOKUP:
        context = get_context(settings, request.query, mode="metadata", paper_ids=request.paper_ids)
        return context.records, [], context.message, {"content_available": context.content_available}
    if decision.intent in {QueryIntent.PAPER_SUMMARY, QueryIntent.PAPER_COMPARISON, QueryIntent.PAPER_CONTENT}:
        mode = {QueryIntent.PAPER_SUMMARY: "summary", QueryIntent.PAPER_COMPARISON: "comparison", QueryIntent.PAPER_CONTENT: "content"}[decision.intent]
        context = get_context(settings, request.query, mode=mode, paper_ids=request.paper_ids)
        if not context.content_available:
            return context.records, [], context.message, {"content_available": False, "requires_ingestion": bool(context.missing_assets), "missing_assets": context.missing_assets}
        if decision.intent == QueryIntent.PAPER_CONTENT:
            found = search_content(settings, request.query, list(request.paper_ids) or None, 8)
            if found["status"] != "ok":
                return context.records, [], "论文 Chunk 索引尚未同步，请先执行 paper_catalog_sync。", {"status": found["status"], "content_available": False, "missing_assets": []}
            return context.records, found["items"], None if found["items"] else "没有找到匹配的正文证据。", {"content_available": bool(found["items"]), "status": "ok", "missing_assets": []}
        ids = [record["paper_id"] for record in context.records]
        if decision.intent in {QueryIntent.PAPER_SUMMARY, QueryIntent.PAPER_COMPARISON}:
            found = context_chunks(settings, ids, 20)
            items, truncated = _limit_context(found["items"], 24000, separate=decision.intent == QueryIntent.PAPER_COMPARISON)
            return context.records, items, None if items else "没有可用的正文 Chunk。", {"content_available": bool(items), "status": found["status"], "missing_assets": [], "truncated": truncated, "context_chars": sum(len(item.get("text", "")) for item in items)}
        found = search_content(settings, request.query, ids, 50)
        items = found.get("items", []) if found.get("status") == "ok" else []
        return context.records, items, None if items else "没有找到匹配的正文证据。", {"content_available": bool(items), "status": found.get("status", "ok")}
    return [], [], decision.error or "当前 Paper RAG 不支持该查询类型。", {}


def _limit_context(items: list[dict[str, Any]], budget: int, *, separate: bool = False) -> tuple[list[dict[str, Any]], bool]:
    if not items:
        return [], False
    per_paper: dict[str, int] = {}
    if separate:
        papers = list(dict.fromkeys(str(item.get("paper_id")) for item in items))
        each = max(1, budget // max(1, len(papers)))
        per_paper = {paper: each for paper in papers}
    used: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    truncated = False
    for item in items:
        key = str(item.get("paper_id"))
        cap = per_paper.get(key, budget)
        text = str(item.get("text") or "")
        if used.get(key, 0) + len(text) > cap:
            truncated = True
            continue
        result.append(item)
        used[key] = used.get(key, 0) + len(text)
    return result, truncated
