from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...config import Settings
from ...dataprocess.annotations import load_paper_annotations
from .utils import normalized_text


PaperTags = dict[str, list[str]]


@dataclass(frozen=True)
class PaperAnnotationEntry:
    """单篇论文的人工别名和标签索引项。"""

    title: str
    paper_title_key: str
    aliases: list[str]
    tags: PaperTags


def load_paper_annotation_entries(settings: Settings) -> list[PaperAnnotationEntry]:
    """统一读取 paper_annotations.json，生成 aliases/tags 共用索引。"""
    entries: list[PaperAnnotationEntry] = []
    for annotation in load_paper_annotations(settings).values():
        title = str(annotation.get("title") or "").strip()
        if not title:
            continue
        entries.append(PaperAnnotationEntry(
            title=title,
            paper_title_key=paper_title_key(title),
            aliases=annotation_string_list(annotation.get("aliases")),
            tags=normalize_tags(annotation.get("tags")),
        ))
    return entries


def normalize_tags(value: Any) -> PaperTags:
    """把 annotation tags 规范成 zh/en 两个字符串列表。"""
    if not isinstance(value, dict):
        return {"zh": [], "en": []}
    return {
        "zh": annotation_string_list(value.get("zh")),
        "en": annotation_string_list(value.get("en")),
    }


def annotation_string_list(value: Any) -> list[str]:
    """清洗 annotation 中的字符串列表字段。"""
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def paper_title_key(title: Any) -> str:
    """生成 annotation 和 manifest 标题对齐用的稳定 key。"""
    text = str(title or "").strip()
    return normalized_text(text) or " ".join(text.lower().split())
