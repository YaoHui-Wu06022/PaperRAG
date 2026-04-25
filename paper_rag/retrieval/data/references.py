from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Settings


@dataclass(frozen=True)
class ReferenceDocument:
    doc_id: str
    paper_id: str
    title: str
    ref_index: int
    raw_text: str
    page: int | None
    source_block_id: str | None


def load_reference_documents(settings: Settings, paper_ids: set[str] | None = None) -> list[ReferenceDocument]:
    documents: list[ReferenceDocument] = []
    if not settings.paper_data_dir.exists():
        return documents
    for directory in sorted(path for path in settings.paper_data_dir.iterdir() if path.is_dir()):
        if paper_ids is not None and directory.name not in paper_ids:
            continue
        metadata = read_json(directory / "metadata.json")
        title = str(metadata.get("title") or directory.name)
        references_path = directory / "references.jsonl"
        if not references_path.exists():
            continue
        for row in read_jsonl(references_path):
            ref_index = int(row.get("ref_index") or 0)
            documents.append(ReferenceDocument(
                doc_id=f"{directory.name}::ref_{ref_index:04d}",
                paper_id=directory.name,
                title=title,
                ref_index=ref_index,
                raw_text=str(row.get("raw_text") or ""),
                page=row.get("page"),
                source_block_id=row.get("source_block_id"),
            ))
    return documents


def to_evidence_reference_document(document: ReferenceDocument) -> dict[str, Any]:
    return {
        "paper_id": document.paper_id,
        "title": document.title,
        "ref_index": document.ref_index,
        "raw_text": document.raw_text,
        "page": document.page,
        "source_block_id": document.source_block_id,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
