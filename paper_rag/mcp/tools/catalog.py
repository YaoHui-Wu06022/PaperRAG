"""论文 Catalog 的结构化 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.catalog.service import (
    catalog_status,
    get_asset_status,
    get_assets,
    get_metadata,
    list_papers,
    rebuild_catalog,
    search_catalog,
)
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_settings


@mcp.tool(name="paper_list", description="按结构化条件列出本地 ArXiv 论文。")
def paper_list(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """列出本地 Catalog 中的论文。"""

    records = list_papers(get_settings(), filters)
    return {"items": [record.to_dict() for record in records], "count": len(records)}


@mcp.tool(name="paper_search", description="按关键词、作者、分类或年份检索本地 ArXiv 论文。")
def paper_search(
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """执行确定性的 Catalog 搜索，不调用 Jev。"""

    records = search_catalog(get_settings(), query, filters, limit=max(1, min(int(limit), 100)))
    return {"query": query, "items": [record.to_dict() for record in records], "count": len(records)}


@mcp.tool(name="paper_get_metadata", description="读取指定论文的结构化元数据。")
def paper_get_metadata(paper_id: str) -> dict[str, Any]:
    """按 base ID 或 canonical ID 读取元数据。"""

    record = get_metadata(get_settings(), paper_id)
    return {"paper_id": paper_id, "found": record is not None, "metadata": record.to_dict() if record else None}


@mcp.tool(name="paper_get_assets", description="读取指定论文的本地文件资产清单。")
def paper_get_assets(paper_id: str) -> dict[str, Any]:
    """返回 PDF、metadata 和 MinerU 目录清单，不查询异步任务。"""

    return get_assets(get_settings(), paper_id)


@mcp.tool(name="paper_asset_status", description="查询指定论文的持久化资产状态，不查询异步任务。")
def paper_asset_status(paper_id: str) -> dict[str, Any]:
    """返回某篇论文是否已有 PDF、metadata 和 MinerU 结果。"""

    return get_asset_status(get_settings(), paper_id)


@mcp.tool(name="paper_catalog_sync", description="预览或重建本地论文 Catalog 派生索引。")
def paper_catalog_sync(confirm: bool = False) -> dict[str, Any]:
    """确认后重建 SQLite Catalog，不修改 ArXiv 原始文件。"""

    settings = get_settings()
    status = catalog_status(settings)
    if not confirm:
        return {
            "status": "confirmation_required",
            "current": status,
            "planned_action": "扫描 data/sources/arxiv 并重建 SQLite Catalog",
        }
    return {"status": "completed", "result": rebuild_catalog(settings)}


__all__ = [
    "paper_asset_status",
    "paper_catalog_sync",
    "paper_get_assets",
    "paper_get_metadata",
    "paper_list",
    "paper_search",
]
