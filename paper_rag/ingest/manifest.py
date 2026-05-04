"""manifest.jsonl 的读写和论文状态记录。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def normalize_year(value: Any) -> dict[str, int | None]:
    """把历史 int 年份和当前双年份结构统一成固定 dict。"""
    if isinstance(value, dict):
        return {
            "preprint_year": parse_year_value(value.get("preprint_year")),
            "publish_year": parse_year_value(value.get("publish_year")),
        }
    return {
        "preprint_year": None,
        "publish_year": parse_year_value(value),
    }


def parse_year_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def effective_year(value: Any) -> int | None:
    """返回用于展示和命名的优先年份。"""
    year = normalize_year(value)
    return year.get("preprint_year") or year.get("publish_year")


@dataclass
class ManifestRecord:
    """manifest 中的一条 PDF/论文记录。"""

    file_hash: str
    status: str
    pdf_path: str | None = None
    title: str | None = None
    author: list[str] = field(default_factory=list)
    year: dict[str, int | None] = field(default_factory=lambda: {"preprint_year": None, "publish_year": None})
    venue: str | None = None
    mineru_output_path: str | None = None
    archived_mineru_output_path: str | None = None
    paper_data_path: str | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRecord":
        """加载旧 manifest 时忽略未知字段，避免历史文件阻塞入库。"""
        known = {field.name for field in cls.__dataclass_fields__.values()}
        values = {k: v for k, v in data.items() if k in known}
        values["year"] = normalize_year(values.get("year"))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_hash": self.file_hash,
            "status": self.status,
            "pdf_path": self.pdf_path,
            "title": self.title,
            "author": self.author,
            "year": normalize_year(self.year),
            "venue": self.venue,
            "mineru_output_path": self.mineru_output_path,
            "archived_mineru_output_path": self.archived_mineru_output_path,
            "paper_data_path": self.paper_data_path,
            "message": self.message,
        }


class Manifest:
    """内存中的 manifest 索引，以 file_hash 作为稳定主键。"""

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
        # 固定排序让 manifest diff 稳定，也便于人工检查状态变化。
        lines = [
            json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
            for record in sorted(self.records.values(), key=lambda r: r.file_hash)
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def get(self, file_hash: str) -> ManifestRecord | None:
        return self.records.get(file_hash)

    def upsert(self, record: ManifestRecord) -> None:
        self.records[record.file_hash] = record
