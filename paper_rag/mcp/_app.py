"""Paper RAG 的 FastMCP 应用实例与生命周期。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastmcp import FastMCP


@asynccontextmanager
async def server_lifespan(_server: FastMCP):
    """为后续资源初始化保留统一生命周期入口。"""

    yield {}


mcp = FastMCP(
    "paper-rag",
    instructions="Paper RAG MCP：提供 ArXiv 获取、MinerU 入库和本地论文知识库查询工具。",
    lifespan=server_lifespan,
)

