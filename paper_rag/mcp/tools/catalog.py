"""论文 Catalog、全文、引用和 LlamaIndex 管理 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.catalog.service import (
    CatalogIndexNotReady,
    catalog_status,
    citation_graph,
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
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "metadata": None, "assets": {}, "asset_status": {}})
    if record is None:
        return _read_only("not_found", {"paper_id": paper_id, "metadata": None, "assets": {}, "asset_status": {}})
    assets = record.assets
    asset_status = {
        "state": record.state,
        "pdf": "present" if assets.get("pdf", {}).get("present") else "missing",
        "metadata": "present" if assets.get("metadata", {}).get("present") else "missing",
        "mineru": "present" if assets.get("mineru", {}).get("present") else "missing",
    }
    return _read_only("ok", {"paper_id": paper_id, "metadata": record.to_dict(), "assets": assets, "asset_status": asset_status})


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


@mcp.tool(name="library_citation", description="读取论文参考文献、被引论文或本地引用关系图；引用查询只访问 SQLite 图，不检索正文。")
def library_citation(paper_id: str, mode: str = "graph", direction: str = "both", depth: int = 1, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode not in {"references", "citations", "graph"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "mode": mode, "items": [], "nodes": [], "edges": []})
    if direction not in {"in", "out", "both"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "mode": mode, "nodes": [], "edges": []})
    try:
        settings = get_settings()
        if mode == "references":
            data = get_references(settings, paper_id)
        elif mode == "citations":
            data = get_citations(settings, paper_id, filters)
        else:
            data = citation_graph(settings, paper_id, direction, depth, filters)
        return _read_only("ok", data)
    except CatalogIndexNotReady:
        return _read_only("catalog_not_ready", {"paper_id": paper_id, "mode": mode, "items": [], "nodes": [], "edges": []})


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
    "library_catalog_sync", "library_citation", "library_get_chunk", "library_get_metadata",
    "library_index_rebuild", "library_index_status", "library_read",
]
