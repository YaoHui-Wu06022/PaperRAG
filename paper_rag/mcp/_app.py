"""Paper RAG 的 FastMCP 应用实例与生命周期。"""

from __future__ import annotations

from contextlib import asynccontextmanager

try:
    # 生产依赖使用独立的 fastmcp 包。
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - 仅兼容旧版本地 MCP 运行时
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

