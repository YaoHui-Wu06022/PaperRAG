"""LlamaIndex 论文元数据和正文 RAG MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.llamaindex.service import retrieve, search
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_settings


@mcp.tool(name="library_search", description="按关键词、作者、分类或年份搜索本地论文元数据；纯元数据问题使用此工具，不读取正文。data.presentation.answer_text 为确定性答案，Agent 应原样输出。")
def library_search(query: str, filters: dict[str, Any] | None = None, limit: int = 20) -> dict[str, Any]:
    return search(get_settings(), query, filters, limit)


@mcp.tool(name="library_retrieve", description="Agent 选择正文检索后调用；服务内部按 fact/reason/summary/comparison 分类，只返回正文 Chunk 证据，不生成答案。hybrid 模式允许 Agent 根据证据组织回答；filters 会先约束候选论文。")
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
