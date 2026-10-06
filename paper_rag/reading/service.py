from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from paper_rag.catalog.service import CatalogIndexNotReady, catalog_status, get_chunk, get_metadata
from paper_rag.config import Settings

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

__all__ = ["read_chunk", "read_fulltext"]
