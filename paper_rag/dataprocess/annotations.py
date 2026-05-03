from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Settings


def paper_annotations_path(settings: Settings) -> Path:
    return settings.data_dir / "paper_annotations.json"


def load_paper_annotations(settings: Settings) -> dict[str, dict[str, Any]]:
    path = paper_annotations_path(settings)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): normalize_paper_annotation(value) for key, value in payload.items() if isinstance(value, dict)}


def normalize_paper_annotation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(value.get("title") or "").strip(),
        "aliases": annotation_string_list(value.get("aliases")),
        "tags": normalize_paper_tags(value.get("tags")),
    }


def normalize_paper_tags(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {"zh": [], "en": []}
    return {
        "zh": annotation_string_list(value.get("zh")),
        "en": annotation_string_list(value.get("en")),
    }


def annotation_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def upsert_paper_annotation(annotations: dict[str, dict[str, Any]], file_hash: str, title: str) -> None:
    annotation = normalize_paper_annotation(annotations.get(file_hash, {}))
    annotation["title"] = title
    annotations[file_hash] = annotation


def save_paper_annotations(settings: Settings, annotations: dict[str, dict[str, Any]]) -> None:
    path = paper_annotations_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        key: annotations[key]
        for key in sorted(annotations)
    }
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
