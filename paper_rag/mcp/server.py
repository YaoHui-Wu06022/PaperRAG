"""Paper RAG MCP 启动入口与公共工具导出。"""

from __future__ import annotations

from paper_rag.mcp._app import mcp
from paper_rag.mcp import tools  # noqa: F401 - 导入副作用负责注册工具
from paper_rag.mcp.runtime import get_settings
from paper_rag.mcp.toolsets import TOOLSETS, apply_toolsets, validate_toolsets
from paper_rag.mcp.tools.acquisition import paper_arxiv_download, paper_job_status
from paper_rag.mcp.tools.ingestion import paper_arxiv_ingest
from paper_rag.mcp.tools.catalog import (
    paper_asset_status,
    paper_catalog_sync,
    paper_get_assets,
    paper_get_metadata,
    paper_list,
    paper_search,
    paper_search_chunks,
    paper_get_chunk,
    paper_get_fulltext,
    paper_get_references,
    paper_get_citations,
    paper_citation_graph,
    paper_embedding_status,
    paper_embedding_rebuild,
)
from paper_rag.mcp.tools.query import paper_query


def _registered_tool_names() -> set[str]:
    """读取当前 FastMCP 注册的工具名称。"""

    provider = getattr(mcp, "_local_provider")
    components = getattr(provider, "_components")
    return {
        key.removeprefix("tool:").split("@", 1)[0]
        for key in components
        if key.startswith("tool:")
    }


validate_toolsets(_registered_tool_names())
ENABLED_TOOLSETS = apply_toolsets(mcp, raw=get_settings().paper_rag_toolsets)

__all__ = [
    "TOOLSETS",
    "ENABLED_TOOLSETS",
    "mcp",
    "paper_arxiv_download",
    "paper_arxiv_ingest",
    "paper_job_status",
    "paper_asset_status",
    "paper_catalog_sync",
    "paper_get_assets",
    "paper_get_metadata",
    "paper_list",
    "paper_search",
    "paper_search_chunks",
    "paper_get_chunk",
    "paper_get_fulltext",
    "paper_get_references",
    "paper_get_citations",
    "paper_citation_graph",
    "paper_embedding_status",
    "paper_embedding_rebuild",
    "paper_query",
]


def main() -> None:
    """以 stdio 传输启动 MCP 服务。"""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

