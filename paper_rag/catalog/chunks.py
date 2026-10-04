from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

CHUNK_RULE_VERSION = "content-list-v1"
MAX_CHARS = 2000
MAX_OVERLAP = 200

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    canonical_id: str
    ordinal: int
    section_path: tuple[str, ...]
    type: str
    text: str
    page_start: int | None
    page_end: int | None
    source_blocks: tuple[dict[str, Any], ...]
    asset_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["section_path"] = list(self.section_path)
        result["source_blocks"] = list(self.source_blocks)
        result["asset_refs"] = list(self.asset_refs)
        return result


def load_content_list(path: Path) -> list[dict[str, Any]]:
    """读取 MinerU content_list.json，并拒绝不明确或损坏的结构。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("content_list.json 无法解析") from exc
    if isinstance(value, dict):
        value = value.get("content_list") or value.get("items")
    if not isinstance(value, list) or not value:
        raise ValueError("content_list.json 必须包含非空列表")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("content_list.json 包含无效条目")
    return value


def build_chunks(
    content_list: list[dict[str, Any]],
    *,
    paper_id: str,
    canonical_id: str,
    content_hash: str,
) -> tuple[list[Chunk], list[str]]:
    """按阅读顺序将 MinerU 结构块转换为稳定 Chunk。"""
    chunks: list[Chunk] = []
    warnings: list[str] = []
    section: list[str] = ["metadata"]
    started = False
    pending: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush() -> None:
        if not pending_text:
            return
        text = "\n\n".join(item.strip() for item in pending_text if item.strip()).strip()
        if not text:
            pending.clear(); pending_text.clear(); return
        blocks = tuple(dict(item) for item in pending)
        chunks.extend(_split_text(text, blocks, paper_id, canonical_id, section, content_hash, len(chunks)))
        pending.clear(); pending_text.clear()

    for index, block in enumerate(content_list):
        kind = str(block.get("type") or block.get("category") or "").casefold()
        text = _block_text(block)
        page = _page(block)
        source = {"index": index, "page_idx": page, "bbox": block.get("bbox")}
        if kind in {"header", "footer", "page_header", "page_footer", "page_number", "page_footnote", "aside_text"}:
            continue
        is_heading = kind in {"title", "heading", "header"} or (kind == "text" and (block.get("text_level") or block.get("level")))
        if is_heading:
            heading = _heading_key(text)
            if not started:
                if heading == "abstract":
                    started = True
                    section[:] = ["abstract"]
                continue
            if heading in {"references", "reference", "bibliography"}:
                flush(); section[:] = ["reference"]; continue
            if heading in {"appendix", "appendices"} or heading.startswith("appendix "):
                flush(); section[:] = ["appendix"]; continue
            flush()
            if section and section[0] == "abstract":
                section[0] = "content"
            level = _level(block)
            if len(section) > 1:
                section[:] = section[: 1 + max(0, level - 2)]
            if text:
                section.append(text)
            continue
        if kind in {"text", "paragraph", "list", "list_item"}:
            if not started:
                continue
            if text:
                pending.append({**source, "type": kind})
                pending_text.append(text)
            continue
        if kind in {"table", "equation", "image", "figure", "code", "formula", "chart"}:
            if not started:
                continue
            flush()
            if not text and kind in {"image", "figure", "chart"}:
                text = str(block.get("img_path") or block.get("image_path") or "").strip()
            if not text:
                warnings.append(f"第 {index} 个 {kind} 块没有可索引文本")
                continue
            refs = _asset_refs(block)
            chunks.append(_make_chunk(text, (source | {"type": kind},), paper_id, canonical_id, section, kind, content_hash, len(chunks), refs))
            continue
        if text and started:
            flush()
            warnings.append(f"未知内容类型 {kind or '空类型'}，按独立文本块保留")
            chunks.append(_make_chunk(text, (source | {"type": kind or "unknown"},), paper_id, canonical_id, section, kind or "unknown", content_hash, len(chunks), _asset_refs(block)))
    flush()
    return chunks, list(dict.fromkeys(warnings))


def _split_text(text: str, blocks: tuple[dict[str, Any], ...], paper_id: str, canonical_id: str, section: list[str], content_hash: str, start: int) -> list[Chunk]:
    if len(text) <= MAX_CHARS:
        return [_make_chunk(text, blocks, paper_id, canonical_id, section, "text", content_hash, start, ())]
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
            result.append(_make_chunk(piece, blocks, paper_id, canonical_id, section, "text", content_hash, start + len(result), ()))
        if end >= len(text):
            break
        offset = max(offset + 1, end - MAX_OVERLAP)
    return result


def _make_chunk(text: str, blocks: tuple[dict[str, Any], ...], paper_id: str, canonical_id: str, section: list[str], kind: str, content_hash: str, ordinal: int, refs: tuple[str, ...]) -> Chunk:
    page_values = [item["page_idx"] for item in blocks if isinstance(item.get("page_idx"), int)]
    identity = json.dumps([canonical_id, content_hash, CHUNK_RULE_VERSION, ordinal, text], ensure_ascii=False, separators=(",", ":"))
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return Chunk(chunk_id, paper_id, canonical_id, ordinal, tuple(section), kind, text, min(page_values) if page_values else None, max(page_values) if page_values else None, blocks, refs)


def _heading_key(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", text or "").strip().casefold()
    value = re.sub(r"^[\d.\s]+", "", value)
    return re.sub(r"[：:]$", "", value).strip()


def _block_text(block: dict[str, Any]) -> str:
    for key in ("text", "content", "latex", "equation", "table_body", "table", "caption", "image_caption"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = "\n".join(str(x) for x in value if x)
            if joined.strip(): return joined.strip()
    return ""


def _asset_refs(block: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("img_path", "image_path", "path", "asset_ref"):
        value = block.get(key)
        if isinstance(value, str) and value.strip(): values.append(value.strip())
    return tuple(dict.fromkeys(values))


def _page(block: dict[str, Any]) -> int | None:
    value = block.get("page_idx")
    return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else None


def _level(block: dict[str, Any]) -> int:
    value = block.get("text_level") or block.get("level") or 1
    try: return max(1, int(value))
    except (TypeError, ValueError): return 1

__all__ = ["CHUNK_RULE_VERSION", "Chunk", "MAX_CHARS", "MAX_OVERLAP", "build_chunks", "load_content_list"]
