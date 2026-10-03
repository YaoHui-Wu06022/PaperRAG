"""从 Catalog 和已存在的论文资产构造只读阅读上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from paper_rag.catalog.service import CatalogRecord, get_metadata, search_catalog
from paper_rag.config import Settings


@dataclass(frozen=True)
class ReadingContext:
    """供 Agent 组织答案的论文上下文。"""

    records: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    content_available: bool = True
    missing_assets: list[str] = field(default_factory=list)
    message: str | None = None


def get_context(
    settings: Settings,
    query: str,
    *,
    mode: str,
    paper_ids: tuple[str, ...] = (),
) -> ReadingContext:
    """按 summary、comparison 或 content 模式读取论文上下文。"""

    records = _select_records(settings, query, mode=mode, paper_ids=paper_ids)
    public_records = [_public_record(record) for record in records]
    if not records:
        return ReadingContext(records=[], message="没有找到匹配的论文。")

    contexts: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        fulltext_path = record.mineru_dir / "full.md" if record.mineru_dir else None
        if not isinstance(fulltext_path, Path) or not fulltext_path.is_file():
            missing.append(f"{record.paper_id}:mineru/full.md")
            continue
        try:
            text = fulltext_path.read_text(encoding="utf-8")
        except OSError:
            missing.append(f"{record.paper_id}:mineru/full.md")
            continue
        if mode == "content":
            text = _matching_text(text, query)
        contexts.append(
            {
                "paper_id": record.paper_id,
                "canonical_id": record.canonical_id,
                "title": record.title,
                "source": str(fulltext_path),
                "text": text[:12000],
            }
        )

    # 摘要和比较可以先使用 metadata；正文问题必须有全文。
    content_available = not missing or mode in {"summary", "comparison"}
    message = None
    if missing and mode == "content":
        message = "论文正文尚未完成 MinerU 解析。"
    return ReadingContext(public_records, contexts, content_available, missing, message)


def _select_records(
    settings: Settings,
    query: str,
    *,
    mode: str,
    paper_ids: tuple[str, ...],
) -> list[CatalogRecord]:
    if paper_ids:
        records = [get_metadata(settings, paper_id) for paper_id in paper_ids]
        selected = [record for record in records if record is not None]
        return selected[:2] if mode == "comparison" else selected[:1]

    ids = _ARXIV_IDS.findall(query)
    if ids:
        records = [get_metadata(settings, paper_id) for paper_id in ids]
        selected = [record for record in records if record is not None]
        if selected:
            return selected[:2] if mode == "comparison" else selected[:1]

    limit = 2 if mode == "comparison" else 1
    return search_catalog(settings, query, limit=limit)


def _public_record(record: CatalogRecord) -> dict[str, Any]:
    return {
        "paper_id": record.paper_id,
        "base_id": record.base_id,
        "canonical_id": record.canonical_id,
        "title": record.title,
        "authors": list(record.authors),
        "abstract": record.abstract,
        "categories": list(record.categories),
        "published_at": record.published_at,
        "updated_at": record.updated_at,
        "abs_url": record.abs_url,
        "pdf_url": record.pdf_url,
    }


def _matching_text(text: str, query: str) -> str:
    terms = _terms(query)
    if not terms:
        return text
    paragraphs = re.split(r"\n\s*\n", text)
    matches = [paragraph for paragraph in paragraphs if any(term in paragraph.casefold() for term in terms)]
    return "\n\n".join(matches) if matches else text


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9][a-z0-9.-]*", value.casefold()) if len(term) > 1]


_ARXIV_IDS = re.compile(r"\b(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z0-9.-]*/\d{7}(?:v\d+)?)\b", re.I)


__all__ = ["ReadingContext", "get_context"]
