"""Paper RAG 的 FastMCP 应用实例与生命周期。"""

from __future__ import annotations

from contextlib import asynccontextmanager

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - 本地运行时尚未安装 FastMCP 时使用 MCP SDK
    from mcp.server.fastmcp import FastMCP


@asynccontextmanager
async def server_lifespan(_server: FastMCP):
    """为后续资源初始化保留统一生命周期入口。"""

    yield {}


mcp = FastMCP(
    "paper-rag",
    instructions="Paper RAG MCP：提供原始论文获取工具，解析和检索由后续阶段接入。",
    lifespan=server_lifespan,
)

