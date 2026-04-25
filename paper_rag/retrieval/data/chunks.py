from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkDocument:
    chunk_id: str
    paper_id: str
    chunk_index: int
    region: str
    section_id: str
    title: str
    section_path: list[str]
    pages: list[int]
    block_ids: list[str]
    text: str
    embedding_text: str

    @property
    def section_path_text(self) -> str:
        return " > ".join(self.section_path)

    @property
    def pages_text(self) -> str:
        return ",".join(str(page) for page in self.pages)


def load_chunk_documents(paper_data_dir: Path) -> list[ChunkDocument]:
    documents: list[ChunkDocument] = []
    for directory in sorted(p for p in paper_data_dir.iterdir() if p.is_dir()):
        chunks_path = directory / "chunks.jsonl"
        metadata_path = directory / "metadata.json"
        if not chunks_path.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        title = str(metadata.get("title") or directory.name)
        for row in read_jsonl(chunks_path):
            documents.append(parse_chunk_row(row, title))
    return documents


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_chunk_row(row: dict[str, Any], title: str) -> ChunkDocument:
    section_path = row.get("section_path") or []
    pages = row.get("pages") or []
    return ChunkDocument(
        chunk_id=str(row["chunk_id"]),
        paper_id=str(row["paper_id"]),
        chunk_index=int(row["chunk_index"]),
        region=str(row.get("region") or ""),
        section_id=str(row.get("section_id") or ""),
        title=title,
        section_path=[str(part) for part in section_path],
        pages=[int(page) for page in pages],
        block_ids=[str(block_id) for block_id in row.get("block_ids") or []],
        text=str(row.get("text") or ""),
        embedding_text=str(row.get("embedding_text") or ""),
    )
