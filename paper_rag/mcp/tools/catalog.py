"""论文 Catalog、全文、引用和 LlamaIndex 管理 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.catalog.service import (
    CatalogIndexNotReady,
    catalog_status,
    citation_graph,
    get_asset_status,
    get_assets,
    get_chunk,
    get_citations,
    get_metadata,
    get_references,
    rebuild_catalog,
)
from paper_rag.llamaindex.service import index_status, rebuild_index
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_jobs, get_settings
from paper_rag.reading.service import read_fulltext


def _read_only(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "data": data, "warnings": warnings or [], "read_only": True}


@mcp.tool(name="library_get_metadata", description="读取指定论文的结构化元数据。")
def library_get_metadata(paper_id: str) -> dict[str, Any]:
    try:
        record = get_metadata(get_settings(), paper_id)
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "metadata": None})
    return _read_only("ok" if record else "not_found", {"paper_id": paper_id, "metadata": record.to_dict() if record else None})


@mcp.tool(name="library_get_assets", description="读取指定论文的本地文件资产清单。")
def library_get_assets(paper_id: str) -> dict[str, Any]:
    try:
        return _read_only("ok", get_assets(get_settings(), paper_id))
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "assets": {}})


@mcp.tool(name="library_get_asset_status", description="查询指定论文的 PDF、metadata 和 MinerU 资产状态。")
def library_get_asset_status(paper_id: str) -> dict[str, Any]:
    try:
        return _read_only("ok", get_asset_status(get_settings(), paper_id))
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id})


@mcp.tool(name="library_get_chunk", description="读取指定 Chunk 的原文及来源定位。")
def library_get_chunk(chunk_id: str) -> dict[str, Any]:
    try:
        item = get_chunk(get_settings(), chunk_id)
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"chunk": None})
    return _read_only("ok" if item else "not_found", {"chunk": item})


@mcp.tool(name="library_read", description="按 Unicode 字符分页读取论文 MinerU 正文。")
def library_read(paper_id: str, offset: int = 0, limit: int = 12000) -> dict[str, Any]:
    result = read_fulltext(get_settings(), paper_id, offset, limit)
    status = str(result.pop("status", "ok"))
    return _read_only(status, result)


@mcp.tool(name="library_get_references", description="读取论文的参考文献条目。")
def library_get_references(paper_id: str) -> dict[str, Any]:
    try:
        return _read_only("ok", get_references(get_settings(), paper_id))
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "items": []})


@mcp.tool(name="library_get_citations", description="读取本地 Catalog 中引用指定论文的论文。")
def library_get_citations(paper_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return _read_only("ok", get_citations(get_settings(), paper_id, filters))
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "items": []})


@mcp.tool(name="library_get_citation_graph", description="读取指定论文的本地引用关系图。")
def library_get_citation_graph(paper_id: str, direction: str = "both", depth: int = 1, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if direction not in {"in", "out", "both"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "nodes": [], "edges": []})
    try:
        return _read_only("ok", citation_graph(get_settings(), paper_id, direction, depth, filters))
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "nodes": [], "edges": []})


@mcp.tool(name="library_index_status", description="查询 LlamaIndex 和 Milvus 正文索引状态。")
def library_index_status() -> dict[str, Any]:
    return index_status(get_settings())


@mcp.tool(name="library_index_rebuild", description="确认后异步重建 LlamaIndex Milvus 正文索引。")
def library_index_rebuild(confirm: bool = False) -> dict[str, Any]:
    settings = get_settings()
    current = index_status(settings)
    if not confirm:
        return {"status": "confirmation_required", "data": {"current": current.get("data", current)}, "warnings": [], "read_only": True}

    def worker(report):
        report("开始重建 LlamaIndex Milvus 索引")
        result = rebuild_index(settings)
        report("LlamaIndex Milvus 索引重建完成")
        return result

    return {"status": "queued", "job": get_jobs().submit("llamaindex_index_rebuild", worker), "warnings": [], "read_only": False}


@mcp.tool(name="library_catalog_sync", description="确认后重建本地论文 Catalog 派生索引。")
def library_catalog_sync(confirm: bool = False) -> dict[str, Any]:
    settings = get_settings()
    status = catalog_status(settings)
    if not confirm:
        return {"status": "confirmation_required", "data": {"current": status}, "warnings": [], "read_only": True}
    return {"status": "completed", "data": rebuild_catalog(settings), "warnings": [], "read_only": False}


__all__ = [
    "library_catalog_sync", "library_get_asset_status", "library_get_assets", "library_get_chunk",
    "library_get_citation_graph", "library_get_citations", "library_get_metadata", "library_get_references",
    "library_index_rebuild", "library_index_status", "library_read",
]
