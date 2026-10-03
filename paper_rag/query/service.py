"""统一只读查询服务。"""

from __future__ import annotations

from paper_rag.config import Settings
from paper_rag.query.classifier import classify_query
from paper_rag.query.handlers import execute_query
from paper_rag.query.jev import JevClient
from paper_rag.query.schemas import QueryRequest, QueryResult


def query_papers(
    settings: Settings,
    query: str,
    paper_ids: list[str] | None = None,
    *,
    client: JevClient | None = None,
) -> QueryResult:
    """识别 query 并从本地论文资产返回只读上下文。"""

    request = QueryRequest(query=str(query or ""), paper_ids=tuple(paper_ids or ()))
    decision = classify_query(settings, request, client=client)
    items, context, message, capabilities = execute_query(settings, request, decision)
    return QueryResult(
        decision=decision,
        query=request.query,
        items=items,
        context=context,
        message=message,
        capabilities=capabilities,
    )


__all__ = ["query_papers"]
