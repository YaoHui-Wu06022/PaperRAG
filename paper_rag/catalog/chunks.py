"""将 MinerU content_list 转成区域化、章节增强的正文 Chunk。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any

CHUNK_RULE_VERSION = "content-list-regions-v6-boundary-aware"
MAX_CHARS = 1200
MAX_OVERLAP = 150
_SKIP_TYPES = {"title", "author", "authors", "affiliation", "header", "footer", "page_header", "page_footer", "page_number", "page_footnote", "aside_text"}
_STRUCTURED_TYPES = {"formula", "equation", "table", "code", "image", "figure", "chart", "list", "list_item"}
_ACKNOWLEDGEMENT_KEYS = {"acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments"}
_CONTENTS_KEYS = {"contents", "table of contents"}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?])\s+|(?<=[.!?])\s+(?=[A-Z\u3400-\u9fff])")
_APPENDIX_PROMPT_STARTS = {
    "can", "could", "did", "do", "explain", "from", "here", "how", "please", "provide", "send", "the", "what", "when", "where", "which", "who", "why", "would", "write",
}


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
    retrieval_text_hash: str

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


def build_chunks(content_list: list[dict[str, Any]], *, paper_id: str, canonical_id: str, content_hash: str, document_title: str | None = None) -> tuple[list[Chunk], list[str]]:
    """按 Abstract、Content、Appendix 分区构建 Chunk；Reference 仅由引用服务处理。"""
    chunks: list[Chunk] = []
    warnings: list[str] = []
    region = "metadata"
    section: list[str] = []
    chapter_number: str | None = None
    chapter_title: str | None = None
    heading_stack: list[str] = []
    acknowledgement_seen = False
    contents_ignored = False
    pending: list[dict[str, Any]] = []
    pending_text: list[str] = []
    implicit_abstract_index = _implicit_abstract_index(content_list)

    def flush() -> None:
        nonlocal pending, pending_text
        text = "\n\n".join(item.strip() for item in pending_text if item.strip()).strip()
        if text and region in {"abstract", "content", "appendix"}:
            chunks.extend(_split_text(text, tuple(pending), paper_id, canonical_id, region, chapter_number, chapter_title, section, content_hash, len(chunks)))
        pending, pending_text = [], []

    def source_for(index: int, block: dict[str, Any], kind: str) -> dict[str, Any]:
        return {
            "index": index,
            "page_idx": _page(block),
            "bbox": block.get("bbox"),
            "type": kind or "unknown",
            "text_format": block.get("text_format"),
        }

    for index, block in enumerate(content_list):
        kind = str(block.get("type") or block.get("category") or "").casefold()
        text = _block_text(block, document_title=document_title)
        if kind in _SKIP_TYPES:
            continue
        abstract_body = _abstract_body(text)
        heading_key = _heading_key(text)
        appendix_label, appendix_title = _appendix_heading(text)
        is_heading = kind in {"heading", "section_header"} or (
            kind == "text"
            and (block.get("text_level") or block.get("level"))
            and (region != "appendix" or _is_appendix_heading_candidate(text, appendix_label))
        )
        if contents_ignored and not is_heading:
            continue
        if contents_ignored:
            contents_ignored = False
        if abstract_body is not None:
            flush(); region = "abstract"; section = ["abstract"]; heading_stack = []; chapter_number = chapter_title = None; acknowledgement_seen = False
            if abstract_body:
                pending.append(source_for(index, block, "text")); pending_text.append(abstract_body)
            continue
        if index == implicit_abstract_index and region == "metadata":
            flush(); region = "abstract"; section = ["abstract"]; heading_stack = []; chapter_number = chapter_title = None; acknowledgement_seen = False
            pending.append(source_for(index, block, "text")); pending_text.append(text)
            continue
        if is_heading:
            if heading_key == "abstract":
                flush(); region = "abstract"; section = ["abstract"]; heading_stack = []; chapter_number = chapter_title = None; acknowledgement_seen = False
                continue
            if heading_key in _ACKNOWLEDGEMENT_KEYS:
                flush()
                heading_stack = [text]
                section = [region, text]
                chapter_number = None
                chapter_title = text
                acknowledgement_seen = True
                continue
            if heading_key in {"references", "reference", "bibliography"}:
                flush(); region = "reference"; section = ["reference"]; heading_stack = []; chapter_number = chapter_title = None; acknowledgement_seen = False
                continue
            if heading_key in _CONTENTS_KEYS:
                flush()
                if region == "abstract":
                    region = "content"
                section = [region]
                heading_stack = []
                chapter_number = chapter_title = None
                contents_ignored = True
                continue
            implicit_appendix = region == "content" and acknowledgement_seen and appendix_label
            if heading_key == "appendix" or heading_key == "appendices" or heading_key.startswith("appendix ") or re.match(r"^[a-z](?:\.\d+)*\s+appendix", heading_key) or (region in {"reference", "appendix"} and appendix_label) or implicit_appendix:
                flush()
                if region != "appendix":
                    heading_stack = []
                region = "appendix"
                acknowledgement_seen = False
                if appendix_label:
                    depth = _appendix_depth(appendix_label)
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
            elif acknowledgement_seen and heading_stack and _heading_key(heading_stack[0]) in _ACKNOWLEDGEMENT_KEYS:
                # 致谢后的 Contributions 等标题应从正文顶层开始，不能继承致谢路径。
                heading_stack = []
                section = [region]
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
        source_kind = "text" if kind == "ref_text" and region == "appendix" else kind
        source = source_for(index, block, source_kind)
        if source_kind in {"text", "paragraph"}:
            if text:
                pending.append(source); pending_text.append(text)
            continue
        if kind in _STRUCTURED_TYPES:
            flush()
            before, after = _neighbor_context(content_list, index)
            retrieval_body = None
            if kind == "table":
                retrieval_body = _table_retrieval_text(block, document_title=document_title)
                before = _plain_text(before)
                after = _plain_text(after)
            source["context_before"] = before
            source["context_after"] = after
            if not text and kind in {"image", "figure", "chart"}:
                text = str(block.get("img_path") or block.get("image_path") or "").strip()
                if text:
                    warnings.append(f"media_text_missing:{index}:{kind}")
            if not text:
                warnings.append(f"第 {index} 个 {kind} 块没有可索引文本")
                continue
            chunks.append(_make_chunk(text, (source,), paper_id, canonical_id, region, chapter_number, chapter_title, section, kind, content_hash, len(chunks), _asset_refs(block), context_before=before, context_after=after, retrieval_body=retrieval_body))
            continue
        if text:
            flush()
            warnings.append(f"未知内容类型 {kind or '空类型'}，按独立文本块保留")
            chunks.append(_make_chunk(text, (source,), paper_id, canonical_id, region, chapter_number, chapter_title, section, "unknown", content_hash, len(chunks), _asset_refs(block)))
    flush()
    return chunks, list(dict.fromkeys(warnings))


def _split_text(text: str, blocks: tuple[dict[str, Any], ...], paper_id: str, canonical_id: str, region: str, chapter_number: str | None, chapter_title: str | None, section: list[str], content_hash: str, start: int) -> list[Chunk]:
    """按段落、句子和词边界切分文本，避免重叠窗口从半词开始。"""

    if len(text) <= MAX_CHARS:
        return [_make_chunk(text, blocks, paper_id, canonical_id, region, chapter_number, chapter_title, section, "text", content_hash, start, ())]

    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", text.strip()):
        value = paragraph.strip()
        if not value:
            continue
        units.extend(_split_paragraph(value))

    result: list[Chunk] = []
    offset = 0
    while offset < len(units):
        end = offset
        size = 0
        while end < len(units):
            addition = len(units[end]) + (2 if end > offset else 0)
            if end > offset and size + addition > MAX_CHARS:
                break
            size += addition
            end += 1

        if end == offset:
            end += 1
        piece = "\n\n".join(units[offset:end]).strip()
        if piece:
            result.append(_make_chunk(piece, blocks, paper_id, canonical_id, region, chapter_number, chapter_title, section, "text", content_hash, start + len(result), ()))
        if end >= len(units):
            break

        overlap = 0
        overlap_size = 0
        while end - overlap - 1 >= offset:
            candidate = len(units[end - overlap - 1]) + (2 if overlap else 0)
            if overlap and overlap_size + candidate > MAX_OVERLAP:
                break
            if not overlap and candidate > MAX_OVERLAP:
                break
            overlap_size += candidate
            overlap += 1
        offset = max(offset + 1, end - overlap)
    return result


def _split_paragraph(paragraph: str) -> list[str]:
    """优先按句子切分；单句过长时只在空白边界切分。"""

    if len(paragraph) <= MAX_CHARS:
        return [paragraph]
    sentences = [part.strip() for part in re.split(_SENTENCE_BOUNDARY_RE, paragraph) if part.strip()]
    if len(sentences) == 1 and sentences[0] == paragraph:
        return _split_words(paragraph)
    result: list[str] = []
    for sentence in sentences:
        result.extend(_split_words(sentence) if len(sentence) > MAX_CHARS else [sentence])
    return result


def _split_words(value: str) -> list[str]:
    """在词边界切分超长句；无空白的长 token 保持完整。"""

    result: list[str] = []
    remaining = value.strip()
    while len(remaining) > MAX_CHARS:
        boundary = remaining.rfind(" ", 1, MAX_CHARS + 1)
        if boundary <= 0:
            break
        result.append(remaining[:boundary].strip())
        remaining = remaining[boundary + 1 :].lstrip()
    if remaining:
        result.append(remaining)
    return result


def _make_chunk(text: str, blocks: tuple[dict[str, Any], ...], paper_id: str, canonical_id: str, region: str, chapter_number: str | None, chapter_title: str | None, section: list[str], kind: str, content_hash: str, ordinal: int, refs: tuple[str, ...], *, context_before: str = "", context_after: str = "", retrieval_body: str | None = None) -> Chunk:
    normalized_section = _normalize_section(section, region)
    prefix = [*normalized_section]
    body = [text if retrieval_body is None else retrieval_body]
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
    retrieval_text_hash = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
    return Chunk(chunk_id, paper_id, canonical_id, ordinal, region, chapter_number, chapter_title, section_path, (section_path[-1] if section_path else None), kind, text, retrieval_text, min(page_values) if page_values else None, max(page_values) if page_values else None, blocks, refs, text_hash, retrieval_text_hash)


def _set_section(section: list[str], region: str, text: str) -> None:
    section[:] = [region, text] if text else [region]


def _abstract_body(text: str) -> str | None:
    value = re.sub(r"<[^>]+>", "", text or "").strip()
    if value.casefold() == "abstract":
        return ""
    match = re.match(r"^abstract\s*[-–—:]\s*(.*)$", value, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _implicit_abstract_index(content_list: list[dict[str, Any]]) -> int | None:
    """没有 Abstract 标题时，定位首个主章节前的最后一段长正文。"""

    if any(_abstract_body(_block_text(block)) is not None for block in content_list):
        return None
    for index, block in enumerate(content_list[1:], 1):
        kind = str(block.get("type") or block.get("category") or "").casefold()
        text = _block_text(block)
        if kind != "text" or not (block.get("text_level") or block.get("level")) or not _chapter(text)[0]:
            continue
        for candidate in range(index - 1, 0, -1):
            previous = content_list[candidate]
            previous_kind = str(previous.get("type") or previous.get("category") or "").casefold()
            previous_text = _block_text(previous)
            if previous_kind == "text" and len(previous_text) >= 200:
                return candidate
        return None
    return None


def _chapter(text: str) -> tuple[str | None, str | None]:
    value = re.sub(r"<[^>]+>", "", text or "").strip()
    match = re.match(r"^((?:\d+(?:\.\d+)*|[IVXLCDM]+(?:\.\d+)*))\.?\s+(.+)$", value, flags=re.IGNORECASE)
    return (match.group(1), match.group(2).strip()) if match else (None, value or None)


def _appendix_heading(text: str) -> tuple[str | None, str | None]:
    """识别 MinerU 常见的 A、A1、A.1 和 A1.1 形式附录标题。"""

    value = re.sub(r"<[^>]+>", "", text or "").strip()
    match = re.match(r"^([A-Z](?:\d+)?(?:\.\d+)*\.?)\s+(.+)$", value)
    if not match:
        return None, None
    return match.group(1).rstrip("."), match.group(2).strip()


def _is_appendix_heading_candidate(text: str, appendix_label: str | None) -> bool:
    """兼容无编号小标题，同时排除 MinerU 误标的示例 prompt。"""

    if appendix_label:
        return True
    value = re.sub(r"<[^>]+>", "", text or "").strip()
    if not value or len(value) > 100 or value.startswith("("):
        return False
    if re.search(r"[?!]", value) or value.endswith((",", ".", ":", ";")):
        return False
    words = value.split()
    if len(words) > 10 or words[0].casefold() in _APPENDIX_PROMPT_STARTS:
        return False
    return value[0].isupper()


def _appendix_depth(label: str) -> int:
    """按字母编号后的数字层级计算附录深度。"""

    # A1、A2 是根级附录；A2.1、A2.2 则都是 A2 的直接子级。
    return 1 + label.count(".")


def _heading_key(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", text or "").strip().casefold()
    value = re.sub(r"^[\d.\s]+", "", value)
    return re.sub(r"[：:]$", "", value).strip()


def _block_text(block: dict[str, Any], *, document_title: str | None = None) -> str:
    kind = str(block.get("type") or block.get("category") or "").casefold()
    if kind == "table":
        return _table_source_text(block, document_title=document_title)
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


class _PlainTextHTMLParser(HTMLParser):
    """提取 HTML 中的可检索文本，并忽略脚本和样式内容。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


class _TableHTMLParser(HTMLParser):
    """读取表格行和显式表头，不展开 rowspan/colspan。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.header_rows: set[int] = set()
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_is_header = False
        self._thead_depth = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if self._ignored_depth:
            if tag in {"script", "style"}:
                self._ignored_depth += 1
            return
        if tag in {"script", "style"}:
            self._ignored_depth = 1
            return
        if tag == "thead":
            self._thead_depth += 1
        elif tag == "tr":
            self._finish_cell()
            self._finish_row()
            self._row = []
        elif tag in {"th", "td"}:
            if self._row is None:
                self._row = []
            self._finish_cell()
            self._cell = []
            self._cell_is_header = tag == "th" or self._thead_depth > 0

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self._ignored_depth:
            if tag in {"script", "style"}:
                self._ignored_depth -= 1
            return
        if tag in {"th", "td"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()
        elif tag == "thead" and self._thead_depth:
            self._thead_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._ignored_depth:
            self._cell.append(data)

    def close(self) -> None:
        super().close()
        self._finish_cell()
        self._finish_row()

    def _finish_cell(self) -> None:
        if self._cell is None:
            return
        if self._row is None:
            self._row = []
        self._row.append(_collapse_whitespace("".join(self._cell)))
        if self._cell_is_header:
            self._row_is_header = True
        self._cell = None
        self._cell_is_header = False

    def _finish_row(self) -> None:
        if self._row is None:
            return
        row = [cell for cell in self._row]
        if row and any(cell for cell in row):
            row_index = len(self.rows)
            self.rows.append(row)
            if getattr(self, "_row_is_header", False):
                self.header_rows.add(row_index)
        self._row = None
        self._row_is_header = False


def _table_source_text(block: dict[str, Any], *, document_title: str | None = None) -> str:
    """保留表格展示用源文本，优先使用 MinerU 提供的独立 text。"""

    explicit = _field_text(block.get("text"))
    if explicit:
        return explicit
    fields = []
    caption = _join_block_fields(block, ("table_caption",))
    if caption and not _caption_matches_document_title(caption, document_title):
        fields.append(caption)
    fields.extend(_join_block_fields(block, key) for key in (("table_body",), ("table_footnote",)))
    return "\n".join(value for value in fields if value).strip()


def _table_retrieval_text(block: dict[str, Any], *, document_title: str | None = None) -> str:
    """将表格 HTML 转为可用于 FTS 和 Embedding 的结构化纯文本。"""

    parts: list[str] = []
    caption = _plain_text(_join_block_fields(block, ("table_caption",)))
    footnote = _plain_text(_join_block_fields(block, ("table_footnote",)))
    if caption and not _caption_matches_document_title(caption, document_title):
        parts.append(caption)

    html_source = _join_block_fields(block, ("table_body",)) or _field_text(block.get("text"))
    rows: list[list[str]] = []
    header_rows: set[int] = set()
    if html_source:
        parser = _TableHTMLParser()
        try:
            parser.feed(html_source)
            parser.close()
            rows = parser.rows
            header_rows = parser.header_rows
        except (AssertionError, ValueError):
            rows = []
            header_rows = set()

    if rows:
        header_index = min(header_rows) if header_rows else None
        if header_index is not None:
            columns = rows[header_index]
            parts.append(f"Columns: {' | '.join(columns)}")
            for row_index, row in enumerate(rows):
                if row_index in header_rows:
                    continue
                parts.append(_render_table_row(row, columns))
        else:
            parts.extend(" | ".join(row) for row in rows if any(row))
    elif html_source:
        fallback = _plain_text(html_source)
        if fallback:
            parts.append(fallback)

    if footnote:
        parts.append(f"Footnote: {footnote}")
    return "\n".join(part for part in parts if part).strip()


def _caption_matches_document_title(caption: str, document_title: str | None) -> bool:
    if not caption or not document_title:
        return False
    return _collapse_whitespace(_plain_text(caption)).casefold() == _collapse_whitespace(_plain_text(document_title)).casefold()


def _render_table_row(row: list[str], columns: list[str]) -> str:
    """按列名配对数据，缺失列为空，多余列使用稳定的 Column N。"""

    width = max(len(row), len(columns))
    labels = list(columns) + [f"Column {index}" for index in range(len(columns) + 1, width + 1)]
    values = row + [""] * (width - len(row))
    return " | ".join(f"{labels[index]}: {values[index]}" for index in range(width))


def _field_text(value: Any) -> str:
    """将 MinerU 字段统一转换为源文本。"""

    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None).strip()
    return str(value).strip() if value is not None else ""


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _plain_text(value: str) -> str:
    """清理 HTML 标签、实体、脚本和样式，保留可检索文本。"""

    if not value:
        return ""
    parser = _PlainTextHTMLParser()
    try:
        parser.feed(value)
        parser.close()
        return _collapse_whitespace(" ".join(parser.parts))
    except (AssertionError, ValueError):
        return _collapse_whitespace(re.sub(r"<[^>]*>", " ", value))


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
