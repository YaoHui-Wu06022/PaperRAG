from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestRecord:
    file_hash: str
    status: str
    pdf_path: str | None = None
    title: str | None = None
    author: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    mineru_output_path: str | None = None
    archived_mineru_output_path: str | None = None
    paper_data_path: str | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRecord":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_hash": self.file_hash,
            "status": self.status,
            "pdf_path": self.pdf_path,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "venue": self.venue,
            "mineru_output_path": self.mineru_output_path,
            "archived_mineru_output_path": self.archived_mineru_output_path,
            "paper_data_path": self.paper_data_path,
            "message": self.message,
        }


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, ManifestRecord] = {}

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        manifest = cls(path)
        if not path.exists():
            return manifest
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = ManifestRecord.from_dict(json.loads(line))
            manifest.records[record.file_hash] = record
        return manifest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
            for record in sorted(self.records.values(), key=lambda r: r.file_hash)
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def get(self, file_hash: str) -> ManifestRecord | None:
        return self.records.get(file_hash)

    def upsert(self, record: ManifestRecord) -> None:
        self.records[record.file_hash] = record
