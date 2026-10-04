"""论文 Catalog 的结构化 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.catalog.service import (
    CatalogIndexNotReady,
    catalog_status,
    get_asset_status,
    get_assets,
    get_metadata,
    list_papers,
    rebuild_catalog,
    search_catalog,
    search_chunks,
    get_chunk,
    get_references,
    get_citations,
    citation_graph,
)
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_settings
from paper_rag.reading.service import read_fulltext


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

    try:
        records = search_catalog(get_settings(), query, filters, limit=max(1, min(int(limit), 100)))
    except CatalogIndexNotReady:
        return {
            "query": query,
            "status": "index_not_ready",
            "items": [],
            "count": 0,
        }
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


@mcp.tool(name="paper_search_chunks", description="使用词法、语义或混合方式检索论文正文证据。")
def paper_search_chunks(query: str, paper_ids: list[str] | None = None, limit: int = 8, mode: str = "hybrid") -> dict[str, Any]:
    if mode in {"semantic", "hybrid"}:
        from paper_rag.semantic.service import hybrid_search
        try:
            return {"query": query, **hybrid_search(get_settings(), query, paper_ids, max(1, min(int(limit), 50)), mode)}
        except Exception as exc:
            return {"status": "failed", "query": query, "items": [], "error": str(exc)}
    if mode != "lexical":
        return {"status": "invalid_mode", "query": query, "items": []}
    try:
        items = search_chunks(get_settings(), query, paper_ids, max(1, min(int(limit), 50)))
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "query": query, "items": []}
    return {"status": "ok", "query": query, "items": items, "count": len(items)}


@mcp.tool(name="paper_get_references", description="读取论文的参考文献条目和解析状态。")
def paper_get_references(paper_id: str) -> dict[str, Any]:
    try:
        return get_references(get_settings(), paper_id)
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "paper_id": paper_id, "items": []}


@mcp.tool(name="paper_get_citations", description="读取本地 Catalog 中引用指定论文的论文。")
def paper_get_citations(paper_id: str) -> dict[str, Any]:
    try:
        return get_citations(get_settings(), paper_id)
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "paper_id": paper_id, "items": []}


@mcp.tool(name="paper_citation_graph", description="读取指定论文的本地引用关系图。")
def paper_citation_graph(paper_id: str, direction: str = "both", depth: int = 1) -> dict[str, Any]:
    if direction not in {"in", "out", "both"}:
        return {"status": "invalid_direction", "paper_id": paper_id, "nodes": [], "edges": []}
    try:
        return citation_graph(get_settings(), paper_id, direction, depth)
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "paper_id": paper_id, "nodes": [], "edges": []}


@mcp.tool(name="paper_embedding_status", description="查询正文 Chunk 向量索引配置和同步状态。")
def paper_embedding_status() -> dict[str, Any]:
    from paper_rag.semantic.service import embedding_status
    return embedding_status(get_settings())


@mcp.tool(name="paper_embedding_rebuild", description="预览或显式重建 Milvus Chunk 向量索引。")
def paper_embedding_rebuild(confirm: bool = False) -> dict[str, Any]:
    settings = get_settings()
    from paper_rag.semantic.service import embedding_status, rebuild_embeddings
    if not confirm:
        status = embedding_status(settings)
        return {"status": "confirmation_required", "planned_chunks": status.get("chunks", 0), "model": settings.embedding_model, "dimensions": settings.embedding_dimensions}
    try:
        return rebuild_embeddings(settings)
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


@mcp.tool(name="paper_get_chunk", description="读取指定 Chunk 的完整原文及来源定位。")
def paper_get_chunk(chunk_id: str) -> dict[str, Any]:
    try:
        item = get_chunk(get_settings(), chunk_id)
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "chunk": None}
    return {"status": "ok" if item else "not_found", "chunk": item}


@mcp.tool(name="paper_get_fulltext", description="按 Unicode 字符分页读取论文 MinerU 正文。")
def paper_get_fulltext(paper_id: str, offset: int = 0, limit: int = 12000) -> dict[str, Any]:
    return read_fulltext(get_settings(), paper_id, offset, limit)


@mcp.tool(name="paper_catalog_sync", description="预览或重建本地论文 Catalog 派生索引。")
def paper_catalog_sync(confirm: bool = False) -> dict[str, Any]:
    """确认后重建 SQLite Catalog，不修改 ArXiv 原始文件。"""

    settings = get_settings()
    status = catalog_status(settings)
    if not confirm:
        return {
            "status": "confirmation_required",
            "current": status,
            "source_papers": status["source_papers"],
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
    "paper_search_chunks",
    "paper_get_chunk",
    "paper_get_fulltext",
    "paper_get_references",
    "paper_get_citations",
    "paper_citation_graph",
    "paper_embedding_status",
    "paper_embedding_rebuild",
]
