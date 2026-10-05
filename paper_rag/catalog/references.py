"""MinerU Reference 条目提取与本地 ArXiv 引用边。"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

_ARXIV = re.compile(r"(?<![A-Za-z0-9])(?:arxiv\s*:\s*)?((?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v\d+)?)(?![A-Za-z0-9_])", re.I)
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
    matched_paper_id: str | None = None
    match_method: str | None = None
    match_score: float | None = None
    duplicate_of_reference_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperIdentity:
    """用于引用目标匹配的最小论文身份信息。"""

    paper_id: str
    base_id: str
    canonical_id: str
    title: str
    authors: tuple[str, ...] = ()
    published_at: str | None = None
    doi: str | None = None


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
    for ordinal, (raw_text, page) in enumerate(grouped, 1):
        target = _ARXIV.search(raw_text)
        target_id = target.group(1) if target else None
        doi = _DOI.search(raw_text)
        doi_value = doi.group(0).rstrip(".,;") if doi else None
        if target_id:
            resolution = "local" if normalize_arxiv_id(target_id).casefold() in local_ids else "external"
        else:
            resolution = "unresolved"
        result.append(Reference(f"{source_paper_id}:{ordinal}", source_paper_id, source_canonical_id, ordinal, raw_text, page, page, target_id, doi_value, resolution))
    return result, warnings


def resolve_references(references: list[Reference], identities: list[PaperIdentity]) -> tuple[list[Reference], dict[str, int], list[str]]:
    """按 ArXiv、DOI、标题作者年份顺序解析本地引用目标。"""

    by_arxiv = {normalize_arxiv_id(item.base_id).casefold(): item for item in identities}
    by_canonical = {normalize_arxiv_id(item.canonical_id).casefold(): item for item in identities}
    by_doi = {
        normalized: item
        for item in identities
        if (normalized := normalize_doi(item.doi))
    }
    stats = {key: 0 for key in ("arxiv_exact", "doi_exact", "title_author_year", "external", "ambiguous", "unresolved")}
    warnings: list[str] = []
    resolved: list[Reference] = []
    for reference in references:
        target = None
        method = None
        score: float | None = None
        external_identifier = False
        target_id = normalize_arxiv_id(reference.target_arxiv_id) if reference.target_arxiv_id else None
        if target_id:
            target = by_arxiv.get(target_id.casefold()) or by_canonical.get(target_id.casefold())
            if target:
                method, score = "arxiv_exact", 1.0
            else:
                external_identifier = True
        if target is None and reference.target_doi:
            target = by_doi.get(normalize_doi(reference.target_doi))
            if target:
                method, score = "doi_exact", 1.0
            else:
                external_identifier = True
        if target is None:
            target, score, ambiguous = _match_by_metadata(reference.raw_text, identities)
            if target and not ambiguous:
                method = "title_author_year"
            elif ambiguous:
                stats["ambiguous"] += 1
                resolved.append(replace(reference, resolution="ambiguous", match_method="ambiguous", match_score=score))
                continue
        if target:
            stats[method] += 1
            resolved.append(replace(reference, target_arxiv_id=target.base_id, resolution="local", matched_paper_id=target.paper_id, match_method=method, match_score=score))
        elif external_identifier:
            stats["external"] += 1
            resolved.append(replace(reference, resolution="external", match_method="external"))
        else:
            stats["unresolved"] += 1
            resolved.append(replace(reference, resolution="unresolved", match_method="unresolved"))
    resolved, duplicate_count = _mark_duplicate_references(resolved)
    for key, value in stats.items():
        if value:
            warnings.append(f"{key}: {value}")
    if duplicate_count:
        warnings.append(f"duplicates: {duplicate_count}")
    return resolved, stats, warnings


def _mark_duplicate_references(references: list[Reference]) -> tuple[list[Reference], int]:
    """保留原始重复条目，但为每个目标选择唯一的主引用。"""

    groups: dict[tuple[str, str], list[Reference]] = {}
    for reference in references:
        if reference.matched_paper_id:
            groups.setdefault((reference.source_paper_id, reference.matched_paper_id), []).append(reference)
    canonical_ids: dict[str, str] = {}
    duplicate_count = 0
    priority = {"arxiv_exact": 3, "doi_exact": 2, "title_author_year": 1}
    for group in groups.values():
        if len(group) < 2:
            continue
        canonical = max(group, key=lambda item: (priority.get(item.match_method or "", 0), item.match_score or 0.0, -item.ordinal))
        for reference in group:
            if reference.reference_id != canonical.reference_id:
                canonical_ids[reference.reference_id] = canonical.reference_id
                duplicate_count += 1
    return [replace(reference, duplicate_of_reference_id=canonical_ids.get(reference.reference_id)) for reference in references], duplicate_count


def normalize_doi(value: str | None) -> str:
    """规范化 DOI，移除常见 URL 前缀和尾部引用标点。"""

    if not value:
        return ""
    result = str(value).strip().casefold()
    result = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", result)
    result = re.sub(r"^doi:\s*", "", result)
    result = result.split("?", 1)[0].split("#", 1)[0]
    return result.rstrip(" .,;:)]}>\"'")


def normalize_reference_text(value: str | None) -> str:
    """将标题、作者和引用正文转换为稳定的比较文本。"""

    if not value:
        return ""
    result = unicodedata.normalize("NFKC", str(value)).casefold()
    result = re.sub(r"\\[a-z]+(?:\{[^}]*\})?", " ", result)
    result = re.sub(r"[^\w]+", " ", result, flags=re.UNICODE)
    return re.sub(r"\s+", " ", result).strip()


def _match_by_metadata(raw_text: str, identities: list[PaperIdentity]) -> tuple[PaperIdentity | None, float | None, bool]:
    raw_normalized = normalize_reference_text(raw_text)
    raw_tokens = set(raw_normalized.split())
    years = set(re.findall(r"(?:19|20)\d{2}", raw_normalized))
    candidates: list[tuple[float, PaperIdentity]] = []
    for identity in identities:
        title = normalize_reference_text(identity.title)
        if not title:
            continue
        title_tokens = set(title.split())
        overlap = len(title_tokens & raw_tokens) / max(1, len(title_tokens))
        sequence = SequenceMatcher(None, title, raw_normalized).ratio()
        title_score = 1.0 if title in raw_normalized else 0.7 * overlap + 0.3 * sequence
        author_tokens = {_author_surname(author) for author in identity.authors if _author_surname(author)}
        author_score = len(author_tokens & raw_tokens) / max(1, min(2, len(author_tokens)))
        year = str(identity.published_at or "")[:4]
        year_score = 1.0 if year and (year in years or any(abs(int(year) - int(value)) <= 1 for value in years)) else 0.0
        score = 0.72 * title_score + 0.18 * min(author_score, 1.0) + 0.10 * year_score
        if title_score >= 0.78 and (author_score > 0 or year_score > 0) and score >= 0.72:
            candidates.append((score, identity))
    candidates.sort(key=lambda item: (-item[0], item[1].base_id))
    if not candidates:
        return None, None, False
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
        return None, candidates[0][0], True
    return candidates[0][1], candidates[0][0], False


def _author_surname(value: str) -> str:
    normalized = normalize_reference_text(value)
    return normalized.split()[-1] if normalized else ""


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


__all__ = [
    "PaperIdentity",
    "Reference",
    "extract_references",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_reference_text",
    "resolve_references",
]
