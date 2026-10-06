"""LlamaIndex 论文元数据和正文 RAG MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.llamaindex.service import retrieve, search
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_settings


@mcp.tool(name="library_search", description="只搜索论文元数据，如标题、作者、分类和年份；不读取正文、不调用 Embedding 或 Milvus。")
def library_search(query: str, filters: dict[str, Any] | None = None, limit: int = 20) -> dict[str, Any]:
    return search(get_settings(), query, filters, limit)


@mcp.tool(name="library_retrieve", description="唯一正文 RAG 工具，默认 hybrid 检索。Agent 先选择本工具，服务内部再由 JEV 将任务分为 fact/reason/summary/comparison；MCP 只返回 evidence，Agent 用 source_id 写 [S#] 引用。")
def library_retrieve(
    query: str,
    paper_ids: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    task: str = "auto",
    mode: str = "hybrid",
    regions: list[str] | None = None,
    limit: int = 8,
    max_chars: int = 24000,
) -> dict[str, Any]:
    return retrieve(get_settings(), query, paper_ids, filters=filters, task=task, limit=limit, mode=mode, regions=regions, max_chars=max_chars)


__all__ = ["library_retrieve", "library_search"]
