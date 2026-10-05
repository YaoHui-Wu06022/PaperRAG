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
from paper_rag.presentation import attach_presentation, citation_presentation


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


@mcp.tool(name="library_citation", description="读取论文参考文献、被引论文或本地引用关系图；引用查询只访问 SQLite 图，不检索正文。data.presentation.answer_text 为确定性答案，Agent 应原样输出。")
def library_citation(paper_id: str, mode: str = "graph", direction: str = "both", depth: int = 1, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode not in {"references", "citations", "graph"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "mode": mode, "items": [], "nodes": [], "edges": []})
    if direction not in {"in", "out", "both"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "mode": mode, "nodes": [], "edges": []})
    try:
        settings = get_settings()
        title = None
        if mode == "references":
            data = get_references(settings, paper_id)
            record = get_metadata(settings, paper_id)
            title = record.title if record else None
        elif mode == "citations":
            data = get_citations(settings, paper_id, filters)
        else:
            data = citation_graph(settings, paper_id, direction, depth, filters)
        title_ids = [paper_id]
        if mode == "references":
            local_reference_items = [
                item
                for item in data.get("items", [])
                if item.get("resolution") == "local" and item.get("matched_paper_id")
            ]
            title_ids.extend(str(item.get("matched_paper_id")) for item in local_reference_items[:10])
        elif mode == "citations":
            title_ids.extend(str(item.get("source_paper_id")) for item in data.get("items", [])[:10] if item.get("source_paper_id"))
        else:
            for edge in data.get("edges", [])[:10]:
                if edge.get("source_paper_id"):
                    title_ids.append(str(edge["source_paper_id"]))
                if edge.get("target_arxiv_id"):
                    title_ids.append(str(edge["target_arxiv_id"]))
        titles = _load_titles(settings, title_ids)
        return attach_presentation(
            _read_only("ok", data),
            citation_presentation(data, mode, title_or_paper_id=title, title_lookup=titles),
        )
    except CatalogIndexNotReady:
        data = {"paper_id": paper_id, "mode": mode, "items": [], "nodes": [], "edges": []}
        return attach_presentation(_read_only("catalog_not_ready", data), citation_presentation(data, mode))


def _load_titles(settings: Any, paper_ids: list[str]) -> dict[str, str]:
    """读取展示所需的少量本地论文题目，不改变结构化引用结果。"""

    titles: dict[str, str] = {}
    for paper_id in dict.fromkeys(paper_ids):
        record = get_metadata(settings, paper_id)
        if not record:
            continue
        for identifier in (record.paper_id, record.base_id, record.canonical_id):
            if identifier:
                titles[str(identifier).casefold()] = record.title
    return titles


@mcp.tool(name="library_index_status", description="查询 LlamaIndex 和 Milvus 正文索引状态。")
def library_index_status() -> dict[str, Any]:
    return index_status(get_settings())


@mcp.tool(name="library_index_rebuild", description="确认后异步同步 LlamaIndex Milvus 正文索引；auto 会在兼容时复用 Embedding。")
def library_index_rebuild(confirm: bool = False, mode: str = "auto") -> dict[str, Any]:
    if mode not in {"auto", "incremental", "full"}:
        return {"status": "invalid_input", "data": {"mode": mode}, "warnings": ["mode must be auto, incremental or full"], "read_only": True}
    settings = get_settings()
    current = index_status(settings)
    if not confirm:
        return {"status": "confirmation_required", "data": {"current": current.get("data", current)}, "warnings": [], "read_only": True}

    def worker(report):
        report(f"开始同步 LlamaIndex Milvus 索引（{mode}）")
        result = rebuild_index(settings, mode=mode)
        report("LlamaIndex Milvus 索引同步完成")
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
