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
    scan_catalog,
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


@mcp.tool(name="library_citation", description="读取论文参考文献、被引论文或本地引用关系图；可传 paper_id 或 paper_title，引用查询只访问 SQLite 图，不检索正文。data.presentation.answer_text 已确定性组织，Agent 应原样输出；graph 双向查询默认两跳，单独查询引用或被引用默认一跳。")
def library_citation(paper_id: str | None = None, paper_title: str | None = None, mode: str = "graph", direction: str = "both", depth: int | None = None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode not in {"references", "citations", "graph"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "mode": mode, "items": [], "nodes": [], "edges": []})
    if direction not in {"in", "out", "both"}:
        return _read_only("invalid_input", {"paper_id": paper_id, "mode": mode, "nodes": [], "edges": []})
    if bool(paper_id) == bool(paper_title):
        return _read_only("invalid_input", {"paper_id": paper_id, "paper_title": paper_title, "mode": mode, "items": [], "nodes": [], "edges": []}, ["exactly one of paper_id or paper_title is required"])
    try:
        settings = get_settings()
        record = _resolve_citation_paper(settings, paper_id=paper_id, paper_title=paper_title)
        if record is None:
            return _read_only("not_found", {"paper_id": paper_id, "paper_title": paper_title, "mode": mode, "items": [], "nodes": [], "edges": []}, ["paper was not found in the local catalog"])
        resolved_paper_id = record.base_id
        title = None
        if mode == "references":
            data = get_references(settings, resolved_paper_id)
            title = record.title
        elif mode == "citations":
            data = get_citations(settings, resolved_paper_id, filters)
        else:
            data = citation_graph(settings, resolved_paper_id, direction, depth, filters)
        title_ids = [resolved_paper_id]
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
            for edge in data.get("edges", []):
                source_id = str(edge.get("source_paper_id") or "")
                target_id = str(edge.get("target_arxiv_id") or "")
                edge_depth = int(edge.get("depth") or 1)
                if edge_depth > 2:
                    continue
                if source_id:
                    title_ids.append(source_id)
                if target_id:
                    title_ids.append(target_id)
        titles = _load_titles(settings, title_ids)
        return attach_presentation(
            _read_only("ok", data),
            citation_presentation(data, mode, title_or_paper_id=title, title_lookup=titles),
        )
    except CatalogIndexNotReady:
        data = {"paper_id": paper_id, "mode": mode, "items": [], "nodes": [], "edges": []}
        return attach_presentation(_read_only("catalog_not_ready", data), citation_presentation(data, mode))
    except ValueError as exc:
        data = {"paper_id": paper_id, "paper_title": paper_title, "mode": mode, "items": [], "nodes": [], "edges": []}
        return _read_only("invalid_input", data, [str(exc)])


def _resolve_citation_paper(settings: Any, *, paper_id: str | None, paper_title: str | None) -> Any:
    """将引用工具的论文 ID 或题目解析为唯一的本地论文。"""

    if paper_id:
        return get_metadata(settings, paper_id)
    wanted = _normalize_title(paper_title or "")
    if not wanted:
        return None
    matches = [record for record in scan_catalog(settings) if _normalize_title(record.title) == wanted]
    if len(matches) > 1:
        raise ValueError("paper_title matches multiple local papers")
    return matches[0] if matches else None


def _normalize_title(value: str) -> str:
    """规范化题目大小写、Unicode 和标点后用于本地精确匹配。"""

    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


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


@mcp.tool(name="library_index_rebuild", description="确认后异步同步 LlamaIndex Milvus 正文索引")
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
