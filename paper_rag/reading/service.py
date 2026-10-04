from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from paper_rag.catalog.service import CatalogIndexNotReady, CatalogRecord, catalog_status, get_chunk, get_metadata, list_chunks, search_catalog, search_chunks
from paper_rag.config import Settings

@dataclass(frozen=True)
class ReadingContext:
    records: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    content_available: bool = True
    missing_assets: list[str] = field(default_factory=list)
    message: str | None = None
    truncated: bool = False
    next_offset: int | None = None


def get_context(settings: Settings, query: str, *, mode: str, paper_ids: tuple[str, ...] = ()) -> ReadingContext:
    try:
        records = _select_records(settings, query, mode=mode, paper_ids=paper_ids)
    except CatalogIndexNotReady:
        return ReadingContext(message="论文 Catalog 尚未同步，请先执行 paper_catalog_sync。")
    public_records = [_public_record(record) for record in records]
    if not records:
        return ReadingContext(records=[], message="没有找到匹配的论文。")
    if mode == "metadata":
        return ReadingContext(records=public_records)
    missing = [f"{record.paper_id}:mineru/full.md" for record in records if not record.mineru_dir or not (record.mineru_dir / "full.md").is_file()]
    stale = [record.paper_id for record in records if not missing and not _current_mineru(record, settings)]
    if missing:
        return ReadingContext(records=public_records, content_available=False, missing_assets=missing, message="论文正文尚未完成 MinerU 解析。")
    if stale:
        return ReadingContext(records=public_records, content_available=False, message="MinerU 正文结果与当前论文版本不一致，请重新同步。")
    return ReadingContext(records=public_records, contexts=[], content_available=True)


def search_content(settings: Settings, query: str, paper_ids: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
    try:
        return {"status": "ok", "items": search_chunks(settings, query, paper_ids, limit)}
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "items": []}


def context_chunks(settings: Settings, paper_ids: list[str], limit_per_paper: int = 20) -> dict[str, Any]:
    try:
        items = []
        for paper_id in paper_ids:
            items.extend(list_chunks(settings, paper_id, limit_per_paper))
        return {"status": "ok", "items": items}
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "items": []}


def read_chunk(settings: Settings, chunk_id: str) -> dict[str, Any]:
    try:
        item = get_chunk(settings, chunk_id)
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "chunk": None}
    return {"status": "ok" if item else "not_found", "chunk": item}


def read_fulltext(settings: Settings, paper_id: str, offset: int = 0, limit: int = 12000) -> dict[str, Any]:
    record = get_metadata(settings, paper_id)
    if record is None:
        return {"status": "not_found", "paper_id": paper_id, "content": ""}
    if not catalog_status(settings).get("index_ready"):
        return {"status": "index_not_ready", "paper_id": record.paper_id, "content": ""}
    mineru_dir = record.mineru_dir
    fulltext = mineru_dir / "full.md" if mineru_dir else None
    manifest = mineru_dir / "manifest.json" if mineru_dir else None
    if not fulltext or not manifest or not fulltext.is_file():
        return {"status": "requires_ingestion", "paper_id": record.paper_id, "content": "", "missing_assets": [f"{record.paper_id}:mineru/full.md"]}
    try:
        text = fulltext.read_text(encoding="utf-8")
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "stale_content", "paper_id": record.paper_id, "content": ""}
    if manifest_data.get("canonical_id") != record.canonical_id or manifest_data.get("model_version") != settings.mineru_model_version or manifest_data.get("language") != settings.mineru_language or (record.pdf_path and record.pdf_path.is_file() and manifest_data.get("source_sha256") != _sha256(record.pdf_path)):
        return {"status": "stale_content", "paper_id": record.paper_id, "content": ""}
    text = _body_from_abstract(text)
    start = max(0, int(offset)); size = max(1, min(int(limit), 12000)); end = min(len(text), start + size)
    return {"status": "ok", "paper_id": record.paper_id, "offset": start, "limit": size, "content": text[start:end], "total_chars": len(text), "truncated": end < len(text), "next_offset": end if end < len(text) else None}


def _select_records(settings: Settings, query: str, *, mode: str, paper_ids: tuple[str, ...]) -> list[CatalogRecord]:
    if paper_ids:
        selected = [record for record in (get_metadata(settings, paper_id) for paper_id in paper_ids) if record is not None]
        return selected[:5] if mode in {"summary", "comparison"} else selected[:1]
    ids = _ARXIV_IDS.findall(query)
    if ids:
        selected = [record for record in (get_metadata(settings, paper_id) for paper_id in ids) if record is not None]
        if selected:
            return selected[:5] if mode in {"summary", "comparison"} else selected[:1]
    limit = 5 if mode in {"summary", "comparison"} else 1
    return search_catalog(settings, query, limit=limit)


def _current_mineru(record: CatalogRecord, settings: Settings) -> bool:
    directory = record.mineru_dir
    if not directory or not (directory / "full.md").is_file() or not (directory / "manifest.json").is_file(): return False
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return False
    return manifest.get("canonical_id") == record.canonical_id and manifest.get("model_version") == settings.mineru_model_version and manifest.get("language") == settings.mineru_language and (not record.pdf_path or not record.pdf_path.is_file() or manifest.get("source_sha256") == _sha256(record.pdf_path))


def _sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _body_from_abstract(text: str) -> str:
    match = re.search(r"(?im)^\s*(?:#+\s*)?abstract\s*:?[ \t]*$", text)
    return text[match.start():] if match else text


def _public_record(record: CatalogRecord) -> dict[str, Any]:
    return {"paper_id": record.paper_id, "base_id": record.base_id, "canonical_id": record.canonical_id, "title": record.title, "authors": list(record.authors), "abstract": record.abstract, "categories": list(record.categories), "published_at": record.published_at, "updated_at": record.updated_at, "abs_url": record.abs_url, "pdf_url": record.pdf_url}

_ARXIV_IDS = re.compile(r"\b(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z0-9.-]*/\d{7}(?:v\d+)?)\b", re.I)

__all__ = ["ReadingContext", "get_context", "search_content", "context_chunks", "read_chunk", "read_fulltext"]
