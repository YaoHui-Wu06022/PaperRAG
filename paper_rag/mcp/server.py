"""Paper RAG LlamaIndex MCP 启动入口。"""

from __future__ import annotations

from paper_rag.mcp._app import mcp
from paper_rag.mcp import tools  # noqa: F401 - 导入副作用负责注册工具
from paper_rag.mcp.runtime import get_settings
from paper_rag.mcp.toolsets import TOOLSETS, apply_toolsets, validate_toolsets
from paper_rag.mcp.tools.acquisition import library_acquire_arxiv, library_job_status
from paper_rag.mcp.tools.answer import library_validate_answer
from paper_rag.mcp.tools.ingestion import library_ingest_mineru
from paper_rag.mcp.tools.catalog import (
    library_catalog_sync,
    library_citation,
    library_get_chunk,
    library_get_metadata,
    library_index_rebuild,
    library_index_status,
    library_read,
)
from paper_rag.mcp.tools.query import library_retrieve, library_search


def _registered_tool_names() -> set[str]:
    """读取当前 FastMCP 注册的工具名称。"""

    provider = getattr(mcp, "_local_provider")
    components = getattr(provider, "_components")
    return {key.removeprefix("tool:").split("@", 1)[0] for key in components if key.startswith("tool:")}


validate_toolsets(_registered_tool_names())
ENABLED_TOOLSETS = apply_toolsets(mcp, raw=get_settings().paper_rag_toolsets)

__all__ = [
    "TOOLSETS", "ENABLED_TOOLSETS", "mcp", "library_acquire_arxiv", "library_catalog_sync",
    "library_citation", "library_get_chunk", "library_get_metadata",
    "library_index_rebuild", "library_index_status", "library_ingest_mineru", "library_job_status",
    "library_read", "library_retrieve", "library_search", "library_validate_answer",
]


def main() -> None:
    """以 stdio 传输启动 MCP 服务。"""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
