"""将 MinerU content_list 转成区域化、章节增强的正文 Chunk。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

CHUNK_RULE_VERSION = "content-list-regions-v3-1200-tree"
MAX_CHARS = 1200
MAX_OVERLAP = 150
_SKIP_TYPES = {"title", "author", "authors", "affiliation", "header", "footer", "page_header", "page_footer", "page_number", "page_footnote", "aside_text"}
_STRUCTURED_TYPES = {"formula", "equation", "table", "code", "image", "figure", "chart", "list", "list_item"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    canonical_id: str
    ordinal: int
    region: str
    chapter_number: str | None
    chapter_title: str | None
    section_path: tuple[str, ...]
    section_label: str | None
    type: str
    text: str
    retrieval_text: str
    page_start: int | None
    page_end: int | None
    source_blocks: tuple[dict[str, Any], ...]
    asset_refs: tuple[str, ...]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["section_path"] = list(self.section_path)
        result["source_blocks"] = list(self.source_blocks)
        result["asset_refs"] = list(self.asset_refs)
        return result


def load_content_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("content_list.json 无法解析") from exc
    if isinstance(value, dict):
        value = value.get("content_list") or value.get("items")
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError("content_list.json 必须包含非空对象列表")
    return value


def build_chunks(content_list: list[dict[str, Any]], *, paper_id: str, canonical_id: str, content_hash: str) -> tuple[list[Chunk], list[str]]:
    """按 Abstract、Content、Appendix 分区构建 Chunk；Reference 仅由引用服务处理。"""
    chunks: list[Chunk] = []
    warnings: list[str] = []
    region = "metadata"
    section: list[str] = []
    chapter_number: str | None = None
    chapter_title: str | None = None
    heading_stack: list[str] = []
    pending: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush() -> None:
        nonlocal pending, pending_text
        text = "\n\n".join(item.strip() for item in pending_text if item.strip()).strip()
        if text and region in {"abstract", "content", "appendix"}:
            chunks.extend(_split_text(text, tuple(pending), paper_id, canonical_id, region, chapter_number, chapter_title, section, content_hash, len(chunks)))
        pending, pending_text = [], []

    for index, block in enumerate(content_list):
        kind = str(block.get("type") or block.get("category") or "").casefold()
        text = _block_text(block)
        if kind in _SKIP_TYPES:
            continue
        is_heading = kind in {"heading", "section_header"} or (kind == "text" and (block.get("text_level") or block.get("level")))
        heading_key = _heading_key(text)
        if is_heading:
            if heading_key == "abstract":
                flush(); region = "abstract"; section = ["abstract"]; heading_stack = []; chapter_number = chapter_title = None
                continue
            if heading_key in {"acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments"}:
                flush()
                heading_stack = [text]
                section = [region, text]
                chapter_number = None
                chapter_title = text
                continue
            if heading_key in {"references", "reference", "bibliography"}:
                flush(); region = "reference"; section = ["reference"]; heading_stack = []; chapter_number = chapter_title = None
                continue
            appendix_label, appendix_title = _appendix_heading(text)
            if heading_key == "appendix" or heading_key == "appendices" or heading_key.startswith("appendix ") or re.match(r"^[a-z](?:\.\d+)*\s+appendix", heading_key) or (region in {"reference", "appendix"} and appendix_label):
                flush()
                if region != "appendix":
                    heading_stack = []
                region = "appendix"
                if appendix_label:
                    depth = appendix_label.count(".") + 1
                    heading_stack = heading_stack[: depth - 1]
                    heading_stack.append(text)
                    section = [region, *heading_stack]
                    chapter_number, chapter_title = appendix_label, appendix_title
                else:
                    section = [region]
                    heading_stack = []
                    chapter_number, chapter_title = _chapter(text)
                    _set_section(section, region, text)
                continue
            if region == "reference":
                continue
            if region == "metadata":
                continue
            flush()
            if region == "abstract":
                region = "content"; section = ["content"]; heading_stack = []
            number, title = _chapter(text)
            if number:
                chapter_number, chapter_title = number, title
                depth = number.count(".") + 1
                heading_stack = heading_stack[: depth - 1]
                heading_stack.append(text)
                section = [region, *heading_stack]
            elif text:
                chapter_title = text
                heading_stack.append(text)
                section = [region, *heading_stack]
            continue
        if region in {"metadata", "reference"}:
            continue
        source = {
            "index": index,
            "page_idx": _page(block),
            "bbox": block.get("bbox"),
            "type": kind or "unknown",
            "text_format": block.get("text_format"),
        }
        if kind in {"text", "paragraph"}:
            if text:
                pending.append(source); pending_text.append(text)
            continue
        if kind in _STRUCTURED_TYPES:
            flush()
            before, after = _neighbor_context(content_list, index)
            source["context_before"] = before
            source["context_after"] = after
            if not text and kind in {"image", "figure", "chart"}:
                text = str(block.get("img_path") or block.get("image_path") or "").strip()
                if text:
                    warnings.append(f"media_text_missing:{index}:{kind}")
            if not text:
                warnings.append(f"第 {index} 个 {kind} 块没有可索引文本")
                continue
            chunks.append(_make_chunk(text, (source,), paper_id, canonical_id, region, chapter_number, chapter_title, section, kind, content_hash, len(chunks), _asset_refs(block), context_before=before, context_after=after))
            continue
        if text:
            flush()
            warnings.append(f"未知内容类型 {kind or '空类型'}，按独立文本块保留")
            chunks.append(_make_chunk(text, (source,), paper_id, canonical_id, region, chapter_number, chapter_title, section, "unknown", content_hash, len(chunks), _asset_refs(block)))
    flush()
    return chunks, list(dict.fromkeys(warnings))


def _split_text(text: str, blocks: tuple[dict[str, Any], ...], paper_id: str, canonical_id: str, region: str, chapter_number: str | None, chapter_title: str | None, section: list[str], content_hash: str, start: int) -> list[Chunk]:
    if len(text) <= MAX_CHARS:
        return [_make_chunk(text, blocks, paper_id, canonical_id, region, chapter_number, chapter_title, section, "text", content_hash, start, ())]
    result: list[Chunk] = []
    offset = 0
    while offset < len(text):
        end = min(offset + MAX_CHARS, len(text))
        if end < len(text):
            boundary = max(text.rfind(mark, offset + 200, end) for mark in ("。", "！", "？", ". ", "! ", "? ", "\n"))
            if boundary > offset:
                end = boundary + 1
        piece = text[offset:end].strip()
        if piece:
            result.append(_make_chunk(piece, blocks, paper_id, canonical_id, region, chapter_number, chapter_title, section, "text", content_hash, start + len(result), ()))
        if end >= len(text):
            break
        offset = max(offset + 1, end - MAX_OVERLAP)
    return result


def _make_chunk(text: str, blocks: tuple[dict[str, Any], ...], paper_id: str, canonical_id: str, region: str, chapter_number: str | None, chapter_title: str | None, section: list[str], kind: str, content_hash: str, ordinal: int, refs: tuple[str, ...], *, context_before: str = "", context_after: str = "") -> Chunk:
    normalized_section = _normalize_section(section, region)
    prefix = [*normalized_section]
    body = [text]
    if context_before:
        body.insert(0, f"[context_before]\n{context_before}")
    if context_after:
        body.append(f"[context_after]\n{context_after}")
    retrieval_text = "\n".join(prefix + body)
    page_values = [item["page_idx"] for item in blocks if isinstance(item.get("page_idx"), int)]
    identity = json.dumps([canonical_id, content_hash, CHUNK_RULE_VERSION, ordinal, text], ensure_ascii=False, separators=(",", ":"))
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    section_path = tuple(normalized_section)
    return Chunk(chunk_id, paper_id, canonical_id, ordinal, region, chapter_number, chapter_title, section_path, (section_path[-1] if section_path else None), kind, text, retrieval_text, min(page_values) if page_values else None, max(page_values) if page_values else None, blocks, refs, text_hash)


def _set_section(section: list[str], region: str, text: str) -> None:
    section[:] = [region, text] if text else [region]


def _chapter(text: str) -> tuple[str | None, str | None]:
    value = re.sub(r"<[^>]+>", "", text or "").strip()
    match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", value)
    return (match.group(1), match.group(2).strip()) if match else (None, value or None)


def _appendix_heading(text: str) -> tuple[str | None, str | None]:
    """识别 MinerU 常见的 A、A.1 形式附录标题。"""

    value = re.sub(r"<[^>]+>", "", text or "").strip()
    match = re.match(r"^([A-Z](?:\.\d+)*\.?)\s+(.+)$", value)
    if not match:
        return None, None
    return match.group(1).rstrip("."), match.group(2).strip()


def _heading_key(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", text or "").strip().casefold()
    value = re.sub(r"^[\d.\s]+", "", value)
    return re.sub(r"[：:]$", "", value).strip()


def _block_text(block: dict[str, Any]) -> str:
    kind = str(block.get("type") or block.get("category") or "").casefold()
    if kind == "table":
        return _join_block_fields(block, ("table_caption", "table_body", "table_footnote"))
    if kind in {"image", "figure"}:
        return _join_block_fields(block, ("image_caption", "image_footnote", "content"))
    if kind == "chart":
        return _join_block_fields(block, ("chart_caption", "chart_footnote", "content"))
    for key in ("text", "content", "latex", "equation", "table_body", "table", "caption", "image_caption", "ref_text"):
        value = block.get(key)
        if isinstance(value, str) and value.strip(): return value.strip()
        if isinstance(value, list):
            joined = "\n".join(str(x) for x in value if x)
            if joined.strip(): return joined.strip()
    return ""


def _join_block_fields(block: dict[str, Any], keys: tuple[str, ...]) -> str:
    """按 MinerU 字段顺序保留标题、正文和脚注。"""

    parts: list[str] = []
    for key in keys:
        value = block.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
    return "\n".join(parts)


def _neighbor_context(content_list: list[dict[str, Any]], index: int, window: int = 4) -> tuple[str, str]:
    """为结构化块提取相邻正文，增强语义检索但不改写原始块。"""

    def find(step: int) -> str:
        position = index + step
        checked = 0
        while 0 <= position < len(content_list) and checked < window:
            block = content_list[position]
            kind = str(block.get("type") or block.get("category") or "").casefold()
            text = _block_text(block)
            if kind in {"heading", "section_header"} or (kind == "text" and (block.get("text_level") or block.get("level"))):
                break
            if kind in _SKIP_TYPES or kind in _STRUCTURED_TYPES:
                position += step
                checked += 1
                continue
            if kind in {"text", "paragraph"} and text:
                return text[-320:] if step < 0 else text[:320]
            position += step
            checked += 1
        return ""

    return find(-1), find(1)


def _normalize_section(section: list[str], region: str) -> list[str]:
    """去除重复目录项，同时确保区域位于路径首部。"""

    values = [str(item).strip() for item in section if str(item).strip()]
    if not values or values[0] != region:
        values.insert(0, region)
    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def _asset_refs(block: dict[str, Any]) -> tuple[str, ...]:
    values = [str(block[key]).strip() for key in ("img_path", "image_path", "path", "asset_ref") if isinstance(block.get(key), str) and str(block[key]).strip()]
    return tuple(dict.fromkeys(values))


def _page(block: dict[str, Any]) -> int | None:
    value = block.get("page_idx")
    return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else None


def _level(block: dict[str, Any]) -> int:
    try: return max(1, int(block.get("text_level") or block.get("level") or 1))
    except (TypeError, ValueError): return 1


__all__ = ["CHUNK_RULE_VERSION", "Chunk", "MAX_CHARS", "MAX_OVERLAP", "build_chunks", "load_content_list"]
