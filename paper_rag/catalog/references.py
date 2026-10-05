"""MinerU Reference 条目提取与本地 ArXiv 引用边。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any

_ARXIV = re.compile(r"(?<![A-Za-z0-9])((?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v\d+)?)(?![A-Za-z0-9_.])", re.I)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_NUMBER = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s*(.*)$")


@dataclass(frozen=True)
class Reference:
    reference_id: str
    source_paper_id: str
    source_canonical_id: str
    ordinal: int
    raw_text: str
    page_start: int | None
    page_end: int | None
    target_arxiv_id: str | None
    target_doi: str | None
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_references(content_list: list[dict[str, Any]], *, source_paper_id: str, source_canonical_id: str, local_ids: set[str]) -> tuple[list[Reference], list[str]]:
    """只读取 Reference 区域的 ref_text，连续同编号条目合并。"""
    items: list[tuple[str, int | None]] = []
    in_reference = False
    for block in content_list:
        text = _text(block)
        kind = str(block.get("type") or block.get("category") or "").casefold()
        if kind in {"heading", "section_header"} or (kind == "text" and (block.get("text_level") or block.get("level"))):
            key = re.sub(r"^[\d.\s]+", "", text.casefold()).strip()
            if key in {"references", "reference", "bibliography"}:
                in_reference = True
            elif in_reference:
                # Reference 之后出现新的章节时，避免把后续正文误当成引用条目。
                break
            continue
        if not in_reference or not text:
            continue
        if kind not in {"ref_text", "reference", "text", "paragraph"} and "ref" not in kind:
            continue
        items.append((text, _page(block)))
    grouped: list[tuple[str, int | None]] = []
    current_number: str | None = None
    for text, page in items:
        match = _NUMBER.match(text)
        number = match.group(1) or match.group(2) if match else None
        body = match.group(3) if match else text
        if number is not None:
            grouped.append((body, page)); current_number = number
        elif grouped and current_number is not None:
            previous, previous_page = grouped[-1]
            grouped[-1] = (f"{previous} {text}".strip(), previous_page if previous_page is not None else page)
        else:
            grouped.append((text, page)); current_number = None
    warnings: list[str] = []
    result: list[Reference] = []
    unresolved_count = 0
    for ordinal, (raw_text, page) in enumerate(grouped, 1):
        target = _ARXIV.search(raw_text)
        target_id = target.group(1) if target else None
        doi = _DOI.search(raw_text)
        doi_value = doi.group(0).rstrip(".,;") if doi else None
        if target_id:
            resolution = "local" if normalize_arxiv_id(target_id).casefold() in local_ids else "external"
        else:
            resolution = "unresolved"
            unresolved_count += 1
        result.append(Reference(f"{source_paper_id}:{ordinal}", source_paper_id, source_canonical_id, ordinal, raw_text, page, page, target_id, doi_value, resolution))
    if unresolved_count:
        warnings.append(f"{unresolved_count} 条引用未解析到 ArXiv ID")
    return result, warnings


def _text(block: dict[str, Any]) -> str:
    for key in ("ref_text", "text", "content"):
        value = block.get(key)
        if isinstance(value, str) and value.strip(): return value.strip()
    return ""


def _page(block: dict[str, Any]) -> int | None:
    value = block.get("page_idx")
    return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else None


def normalize_arxiv_id(value: str) -> str:
    """只移除 ArXiv 版本后缀，保留 ID 中其他字母。"""

    return re.sub(r"v\d+$", "", str(value).strip(), flags=re.IGNORECASE)


__all__ = ["Reference", "extract_references", "normalize_arxiv_id"]
