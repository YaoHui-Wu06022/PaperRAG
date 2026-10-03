"""ArXiv 本地 Catalog 服务。

Catalog 只读取 ``data/sources/arxiv`` 中的原始资产，并可将扫描结果写入
SQLite 派生索引。MinerU 尚未成功生成的论文仍然可以被元数据检索。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from paper_rag.config import Settings


@dataclass(frozen=True)
class CatalogRecord:
    """一篇论文在本地 Catalog 中的只读记录。"""

    paper_id: str
    base_id: str
    canonical_id: str
    title: str
    authors: tuple[str, ...] = ()
    abstract: str = ""
    categories: tuple[str, ...] = ()
    published_at: str | None = None
    updated_at: str | None = None
    abs_url: str | None = None
    pdf_url: str | None = None
    metadata_path: Path | None = None
    pdf_path: Path | None = None
    mineru_dir: Path | None = None
    state: str = "missing"
    assets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 MCP 和 CLI 可安全序列化的结构。"""

        return {
            "paper_id": self.paper_id,
            "base_id": self.base_id,
            "canonical_id": self.canonical_id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "categories": list(self.categories),
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "mineru_dir": str(self.mineru_dir) if self.mineru_dir else None,
            "state": self.state,
            "assets": self.assets,
        }


def scan_catalog(settings: Settings) -> list[CatalogRecord]:
    """扫描 ArXiv 目录，不读取旧的 ``data/mineru_output``。"""

    root = settings.arxiv_data_dir
    if not root.is_dir():
        return []
    records: list[CatalogRecord] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        records.append(_record_from_metadata(metadata_path, metadata))
    return records


def list_papers(settings: Settings, filters: dict[str, Any] | None = None) -> list[CatalogRecord]:
    """按结构化条件列出论文。"""

    filters = filters or {}
    records = scan_catalog(settings)
    result: list[CatalogRecord] = []
    for record in records:
        if filters.get("author") and not _contains(record.authors, str(filters["author"])):
            continue
        if filters.get("category") and not _contains(record.categories, str(filters["category"])):
            continue
        if filters.get("state") and record.state != str(filters["state"]):
            continue
        if filters.get("year") and not str(record.published_at or "").startswith(str(filters["year"])):
            continue
        result.append(record)
    return result


def search_catalog(
    settings: Settings,
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[CatalogRecord]:
    """在标题、摘要、作者、分类和 ArXiv ID 中执行轻量检索。"""

    records = list_papers(settings, filters)
    text = str(query or "").strip().casefold()
    if not text:
        return records[: max(0, limit)]
    terms = _terms(text)
    scored: list[tuple[int, CatalogRecord]] = []
    for record in records:
        haystack = " ".join(
            [
                record.title,
                record.abstract,
                " ".join(record.authors),
                " ".join(record.categories),
                record.base_id,
                record.canonical_id,
            ]
        ).casefold()
        score = sum(1 for term in terms if term in haystack)
        if text in haystack:
            score += 2
        if score:
            scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], item[1].title.casefold()))
    return [record for _, record in scored[: max(0, limit)]]


def get_metadata(settings: Settings, paper_id: str) -> CatalogRecord | None:
    """按 base ID 或 canonical ID 读取论文元数据。"""

    wanted = str(paper_id).casefold()
    return next(
        (
            record
            for record in scan_catalog(settings)
            if record.base_id.casefold() == wanted or record.canonical_id.casefold() == wanted
        ),
        None,
    )


def get_assets(settings: Settings, paper_id: str) -> dict[str, Any]:
    """返回论文文件资产清单，不包含异步任务状态。"""

    record = get_metadata(settings, paper_id)
    if record is None:
        return {"paper_id": str(paper_id), "found": False, "assets": {}}
    return {"paper_id": record.paper_id, "found": True, "assets": record.assets}


def get_asset_status(settings: Settings, paper_id: str) -> dict[str, Any]:
    """返回论文持久化资产状态，与 JobManager 状态分离。"""

    record = get_metadata(settings, paper_id)
    if record is None:
        return {"paper_id": str(paper_id), "found": False, "state": "missing"}
    return {
        "paper_id": record.paper_id,
        "found": True,
        "state": record.state,
        "pdf": "present" if record.assets.get("pdf", {}).get("present") else "missing",
        "metadata": "present" if record.assets.get("metadata", {}).get("present") else "missing",
        "mineru": "present" if record.assets.get("mineru", {}).get("present") else "missing",
    }


def rebuild_catalog(settings: Settings) -> dict[str, Any]:
    """将当前 ArXiv 目录扫描结果写入 SQLite 派生索引。"""

    records = scan_catalog(settings)
    db_path = settings.paper_catalog_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                base_id TEXT NOT NULL,
                canonical_id TEXT NOT NULL,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                abstract TEXT NOT NULL,
                categories TEXT NOT NULL,
                published_at TEXT,
                updated_at TEXT,
                state TEXT NOT NULL,
                metadata_path TEXT,
                pdf_path TEXT,
                mineru_dir TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                paper_id UNINDEXED,
                title,
                abstract,
                authors,
                categories
            );
            DELETE FROM papers;
            DELETE FROM papers_fts;
            """
        )
        for record in records:
            connection.execute(
                """
                INSERT INTO papers (
                    paper_id, base_id, canonical_id, title, authors, abstract,
                    categories, published_at, updated_at, state,
                    metadata_path, pdf_path, mineru_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.paper_id,
                    record.base_id,
                    record.canonical_id,
                    record.title,
                    json.dumps(record.authors, ensure_ascii=False),
                    record.abstract,
                    json.dumps(record.categories, ensure_ascii=False),
                    record.published_at,
                    record.updated_at,
                    record.state,
                    str(record.metadata_path) if record.metadata_path else None,
                    str(record.pdf_path) if record.pdf_path else None,
                    str(record.mineru_dir) if record.mineru_dir else None,
                ),
            )
            connection.execute(
                "INSERT INTO papers_fts (paper_id, title, abstract, authors, categories) VALUES (?, ?, ?, ?, ?)",
                (record.paper_id, record.title, record.abstract, " ".join(record.authors), " ".join(record.categories)),
            )
    return {"database": str(db_path), "papers": len(records), "status": "rebuilt"}


def catalog_status(settings: Settings) -> dict[str, Any]:
    """返回 Catalog 的派生索引状态，不查询异步任务。"""

    records = scan_catalog(settings)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.state] = counts.get(record.state, 0) + 1
    return {
        "database": str(settings.paper_catalog_db_path),
        "database_exists": settings.paper_catalog_db_path.is_file(),
        "papers": len(records),
        "states": counts,
    }


def _record_from_metadata(metadata_path: Path, metadata: dict[str, Any]) -> CatalogRecord:
    source_dir = metadata_path.parent
    pdf_path = source_dir / "paper.pdf"
    mineru_dir = source_dir / "mineru"
    mineru_present = mineru_dir.is_dir() and (mineru_dir / "full.md").is_file()
    state = "missing"
    if pdf_path.is_file():
        state = "ingested" if mineru_present else "ready_for_ingest"
    assets = {
        "pdf": {"present": pdf_path.is_file(), "path": str(pdf_path)},
        "metadata": {"present": metadata_path.is_file(), "path": str(metadata_path)},
        "mineru": {"present": mineru_present, "path": str(mineru_dir)},
    }
    base_id = str(metadata.get("base_id") or source_dir.name)
    canonical_id = str(metadata.get("canonical_id") or base_id)
    return CatalogRecord(
        paper_id=base_id,
        base_id=base_id,
        canonical_id=canonical_id,
        title=str(metadata.get("title") or base_id),
        authors=tuple(str(item) for item in metadata.get("authors", []) if item),
        abstract=str(metadata.get("abstract") or ""),
        categories=tuple(str(item) for item in metadata.get("categories", []) if item),
        published_at=_optional_string(metadata.get("published_at")),
        updated_at=_optional_string(metadata.get("updated_at")),
        abs_url=_optional_string(metadata.get("abs_url")),
        pdf_url=_optional_string(metadata.get("pdf_url")),
        metadata_path=metadata_path,
        pdf_path=pdf_path,
        mineru_dir=mineru_dir,
        state=state,
        assets=assets,
    )


def _terms(value: str) -> list[str]:
    compact = re.sub(r"\s+", "", value)
    terms = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]+", value)
    if compact and compact not in terms:
        terms.append(compact)
    return [term for term in terms if len(term) > 1]


def _contains(values: tuple[str, ...], query: str) -> bool:
    wanted = query.casefold()
    return any(wanted in value.casefold() for value in values)


def _optional_string(value: Any) -> str | None:
    return str(value) if value else None


__all__ = [
    "CatalogRecord",
    "catalog_status",
    "get_asset_status",
    "get_assets",
    "get_metadata",
    "list_papers",
    "rebuild_catalog",
    "scan_catalog",
    "search_catalog",
]
