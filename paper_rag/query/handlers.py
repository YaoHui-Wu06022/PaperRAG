"""将查询意图映射到共享 Catalog 和 Reading 服务。"""

from __future__ import annotations

from typing import Any

from paper_rag.catalog.service import search_catalog
from paper_rag.config import Settings
from paper_rag.query.schemas import QueryIntent, QueryIntentDecision, QueryRequest
from paper_rag.reading.service import get_context


def execute_query(
    settings: Settings,
    request: QueryRequest,
    decision: QueryIntentDecision,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, dict[str, Any]]:
    """执行只读查询，并复用 Catalog/Reading 服务。"""

    if decision.needs_clarification or decision.intent == QueryIntent.CLARIFY:
        return [], [], "请补充论文 ID、主题或想了解的具体内容。", {}
    if decision.intent == QueryIntent.PAPER_DISCOVERY:
        records = search_catalog(settings, request.query, limit=50)
        return [record.to_dict() for record in records], [], None, {}
    if decision.intent == QueryIntent.METADATA_LOOKUP:
        context = get_context(settings, request.query, mode="summary", paper_ids=request.paper_ids)
        return context.records, [], context.message, {
            "content_available": context.content_available,
            "missing_assets": context.missing_assets,
        }
    if decision.intent in {
        QueryIntent.PAPER_SUMMARY,
        QueryIntent.PAPER_COMPARISON,
        QueryIntent.PAPER_CONTENT,
    }:
        mode = {
            QueryIntent.PAPER_SUMMARY: "summary",
            QueryIntent.PAPER_COMPARISON: "comparison",
            QueryIntent.PAPER_CONTENT: "content",
        }[decision.intent]
        context = get_context(settings, request.query, mode=mode, paper_ids=request.paper_ids)
        return context.records, context.contexts, context.message, {
            "content_available": context.content_available,
            "missing_assets": context.missing_assets,
        }
    return [], [], decision.error or "当前 Paper RAG 尚不支持该查询类型。", {}
