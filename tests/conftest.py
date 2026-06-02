import json
from pathlib import Path

import pytest

from paper_rag.config import Settings
from paper_rag.ingest.manifest import Manifest, ManifestRecord


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    for name in ["pdf", "mineru_output", "paper_data", "index", "archive"]:
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    return Settings.load(tmp_path)


def save_manifest(settings: Settings, records: list[ManifestRecord]) -> Manifest:
    manifest = Manifest(settings.manifest_path)
    manifest.records = {record.file_hash: record for record in records}
    manifest.save()
    return manifest


def add_paper(
    settings: Settings,
    *,
    file_hash: str,
    paper_id: str,
    title: str,
    authors: list[str] | None = None,
    year: dict[str, int | None] | None = None,
    venue: str | None = None,
    chunks: list[dict] | None = None,
    references: list[dict] | None = None,
) -> ManifestRecord:
    paper_dir = settings.paper_data_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "title": title,
        "author": authors or [],
        "year": year or {"preprint_year": None, "publish_year": None},
        "venue": venue,
        "pdf_path": str(settings.pdf_dir / f"{paper_id}.pdf"),
    }
    write_json(paper_dir / "metadata.json", metadata)
    write_jsonl(paper_dir / "chunks.jsonl", chunks or [])
    write_jsonl(paper_dir / "blocks.jsonl", blocks_from_chunks(chunks or []))
    write_jsonl(paper_dir / "references.jsonl", references or [])
    return ManifestRecord(
        file_hash=file_hash,
        status="active",
        pdf_path=metadata["pdf_path"],
        title=title,
        author=authors or [],
        year=metadata["year"],
        venue=venue,
        paper_data_path=str(paper_dir),
    )


def chunk_row(
    paper_id: str,
    index: int,
    *,
    region: str,
    text: str,
    section: str = "Introduction",
) -> dict:
    return {
        "chunk_id": f"{paper_id}::chunk_{index:04d}",
        "paper_id": paper_id,
        "chunk_index": index,
        "region": region,
        "section_id": f"sec_{section.lower()}",
        "section_path": [section],
        "pages": [index + 1],
        "block_ids": [f"b{index:06d}"],
        "text": text,
        "embedding_text": f"Paper: {paper_id}\nSection: {section}\n\n{text}",
        "char_count": len(text),
    }


def blocks_from_chunks(chunks: list[dict]) -> list[dict]:
    rows = []
    for order, chunk in enumerate(chunks):
        rows.append({
            "block_id": chunk["block_ids"][0],
            "order": order,
            "region": chunk["region"],
            "type": "paragraph",
            "text": chunk["text"],
            "page": chunk["pages"][0],
            "bbox": None,
            "section_id": chunk["section_id"],
            "section_path": chunk["section_path"],
        })
    return rows


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
