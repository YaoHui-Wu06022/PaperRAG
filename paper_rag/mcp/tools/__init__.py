"""导入工具模块以完成 FastMCP 工具注册。"""

from paper_rag.mcp.tools import acquisition  # noqa: F401
from paper_rag.mcp.tools import answer  # noqa: F401
from paper_rag.mcp.tools import catalog  # noqa: F401
from paper_rag.mcp.tools import ingestion  # noqa: F401
from paper_rag.mcp.tools import query  # noqa: F401

__all__ = ["acquisition", "catalog", "ingestion", "query"]

