"""Paper RAG 统一只读查询 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_settings
from paper_rag.query.service import query_papers


@mcp.tool(
    name="paper_query",
    description="识别论文知识库内部查询意图，并从本地 ArXiv/MinerU 资料返回只读上下文；下载、解析和任务状态请调用专用工具。",
)
def paper_query(query: str, paper_ids: list[str] | None = None) -> dict[str, Any]:
    """对自然语言 query 做意图识别并读取本地论文资料。"""

    result = query_papers(get_settings(), query, paper_ids)
    return result.to_dict()


__all__ = ["paper_query"]
