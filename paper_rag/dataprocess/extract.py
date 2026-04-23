from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..utils import normalize_text, slugify_title


SPECIAL_TITLE_NORMALIZED = {
    "abstract",
    "references",
    "reference",
    "bibliography",
    "appendix",
    "acknowledgement",
    "acknowledgements",
    "acknowledgments",
    "acknowledgment",
    "keywords",
    "ccsconcepts",
}

KEYWORD_TITLE_NORMALIZED = {
    "keyword",
    "keywords",
    "indexterm",
    "indexterms",
    "ccsconcept",
    "ccsconcepts",
}

ACKNOWLEDGEMENT_PREFIXES = (
    "acknowledg",
    "funding",
    "disclosure",
)

DEFAULT_CHUNK_TARGET_CHARS = 1400
DEFAULT_CHUNK_OVERLAP_CHARS = 200
MAX_CHUNK_EQUATION_CHARS = 500

IGNORED_TYPES = {
    "page_footnote",
    "page_aside_text",
    "page_number",
    "page_header",
    "page_footer",
}

@dataclass
class FlatBlock:
    index: int
    page: int
    block_index: int
    type: str
    text: str
    bbox: list[Any] | None
    raw: dict[str, Any]


@dataclass
class ExtractionResult:
    title: str
    paper_data_dir: Path
    metadata_path: Path
    toc_path: Path
    blocks_path: Path
    chunks_path: Path
    references_path: Path
    sections: list[dict[str, Any]]
    block_count: int
    chunk_count: int
    reference_count: int
    warnings: list[str]


def load_content_list_v2(path: Path) -> list[list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages: list[list[dict[str, Any]]] = []
    for page in data:
        if isinstance(page, list):
            pages.append([b for b in page if isinstance(b, dict)])
        elif isinstance(page, dict):
            blocks = page.get("value") or page.get("blocks") or page.get("content") or []
            pages.append([b for b in blocks if isinstance(b, dict)])
        else:
            pages.append([])
    return pages


def flatten_pages(pages: list[list[dict[str, Any]]]) -> list[FlatBlock]:
    blocks: list[FlatBlock] = []
    order = 0
    for page_index, page in enumerate(pages, start=1):
        for block_index, block in enumerate(page):
            blocks.append(
                FlatBlock(
                    index=order,
                    page=page_index,
                    block_index=block_index,
                    type=str(block.get("type") or ""),
                    text=block_to_text(block),
                    bbox=block.get("bbox"),
                    raw=block,
                )
            )
            order += 1
    return blocks


def block_to_text(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    content = block.get("content") or {}
    if block_type == "title":
        return pieces_to_text(content.get("title_content")).strip()
    if block_type == "paragraph":
        return pieces_to_text(content.get("paragraph_content")).strip()
    if block_type == "list":
        lines = []
        for item in content.get("list_items") or []:
            if not isinstance(item, dict):
                continue
            text = pieces_to_text(item.get("item_content")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
    if block_type == "image":
        return pieces_to_text(content.get("image_caption")).strip()
    if block_type == "table":
        caption = pieces_to_text(content.get("table_caption")).strip()
        return table_to_semistructured_text(caption, str(content.get("html") or ""))
    if block_type == "equation_interline":
        return str(content.get("math_content") or "").strip()
    if block_type == "code":
        return pieces_to_text(content).strip()
    if block_type in {"page_header", "page_footer", "page_number", "page_aside_text", "page_footnote"}:
        return pieces_to_text(content.get(f"{block_type}_content")).strip()
    if block_type in IGNORED_TYPES:
        return ""
    return pieces_to_text(content).strip()


def pieces_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(pieces_to_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        return "".join(
            pieces_to_text(item)
            for item in value.values()
            if isinstance(item, (str, list, dict))
        )
    return ""


def html_to_text(html: str) -> str:
    html = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", html, flags=re.I)
    html = re.sub(r"</tr>\s*<tr[^>]*>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html).strip()


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._cell_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._cell_tag = tag

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            text = normalize_whitespace("".join(self._current_cell))
            self._current_row.append(text)
            self._current_cell = None
            self._cell_tag = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def html_table_rows(html: str) -> list[list[str]]:
    parser = TableHTMLParser()
    parser.feed(html)
    return parser.rows


def is_probable_header_row(row: list[str]) -> bool:
    non_empty = [cell for cell in row if cell]
    if len(non_empty) < 2:
        return False
    numeric_cells = sum(1 for cell in non_empty if re.fullmatch(r"[\d\s.,%≈<>≤≥+\-–—/]+", cell))
    return numeric_cells / len(non_empty) < 0.5


def table_to_semistructured_text(caption: str, html: str) -> str:
    rows = html_table_rows(html)
    if not rows:
        return " ".join(part for part in [caption, html_to_text(html)] if part).strip()
    lines: list[str] = []
    if caption:
        lines.append(f"Table: {caption}")
    max_columns = max((len(row) for row in rows), default=0)
    if is_probable_header_row(rows[0]):
        columns = [cell or f"column_{index + 1}" for index, cell in enumerate(rows[0])]
        data_rows = rows[1:]
    else:
        columns = [f"column_{index + 1}" for index in range(max_columns)]
        data_rows = rows
    if len(columns) < max_columns:
        columns.extend(f"column_{index + 1}" for index in range(len(columns), max_columns))
    if columns:
        lines.append("Columns: " + ", ".join(columns) + ".")
    row_number = 1
    for row in data_rows:
        values = list(row) + [""] * max(0, len(columns) - len(row))
        pairs = [
            f"{column} = {value}"
            for column, value in zip(columns, values)
            if value
        ]
        if not pairs:
            continue
        lines.append(f"Row {row_number}: " + "; ".join(pairs) + ".")
        row_number += 1
    return "\n".join(lines).strip()


def structural_title_text(text: str) -> str:
    return strip_heading_number(text)


def is_special_title(text: str) -> bool:
    return normalize_text(structural_title_text(text)) in SPECIAL_TITLE_NORMALIZED


def is_keyword_marker(text: str) -> bool:
    title_text = structural_title_text(text)
    normalized = normalize_text(title_text)
    if normalized in KEYWORD_TITLE_NORMALIZED:
        return True
    return bool(re.match(r"^\s*(keywords?|index terms?|ccs concepts?)\b", title_text, flags=re.I))


def is_acknowledgement_title(text: str) -> bool:
    normalized = normalize_text(structural_title_text(text))
    return normalized.startswith(ACKNOWLEDGEMENT_PREFIXES)


def is_abstract_title(text: str) -> bool:
    return normalize_text(structural_title_text(text)) == "abstract"


def is_references_title(text: str) -> bool:
    return normalize_text(structural_title_text(text)) in {"references", "reference", "bibliography"}


def is_appendix_title(text: str) -> bool:
    return normalize_text(structural_title_text(text)).startswith("appendix")


def starts_with_abstract(text: str) -> bool:
    return bool(re.match(r"^\s*abstract\s*[\.:]\s+", text, flags=re.I))


def strip_abstract_prefix(text: str) -> str:
    return re.sub(r"^\s*abstract\s*[\.:]\s+", "", text, count=1, flags=re.I).strip()


def heading_number(text: str) -> str | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)(?:\.|\))?\s+", text)
    if not match:
        return None
    return match.group(1).rstrip(".")


def strip_heading_number(text: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*(?:\.|\))?\s+", "", text).strip()


def block_level(block: FlatBlock, fallback: int = 1) -> int:
    level = (block.raw.get("content") or {}).get("level")
    return level if isinstance(level, int) and level > 0 else fallback


def abstract_marker(blocks: list[FlatBlock]) -> FlatBlock | None:
    title = next((b for b in blocks if b.type == "title" and is_abstract_title(b.text)), None)
    if title is not None:
        return title
    return next((b for b in blocks if b.type == "paragraph" and starts_with_abstract(b.text)), None)


def extract_title(blocks: list[FlatBlock]) -> str | None:
    title_blocks = [b for b in blocks if b.type == "title" and b.text]
    abstract = abstract_marker(blocks)
    candidates = [
        b for b in title_blocks
        if (abstract is None or b.index < abstract.index) and not is_special_title(b.text)
    ]
    if candidates:
        return candidates[0].text.strip()
    header = next((b for b in blocks if b.page == 1 and b.type == "page_header" and b.text), None)
    if header:
        return header.text.strip()
    return None


def find_region_boundaries(blocks: list[FlatBlock]) -> dict[str, int | None]:
    abstract_title = next((b for b in blocks if b.type == "title" and is_abstract_title(b.text)), None)
    abstract_para = None
    if abstract_title is None:
        abstract_para = next((b for b in blocks if b.type == "paragraph" and starts_with_abstract(b.text)), None)
    abstract_start = (abstract_title.index + 1) if abstract_title else (abstract_para.index if abstract_para else None)
    references_title = next((b for b in blocks if b.type == "title" and is_references_title(b.text)), None)
    appendix_before_ref = None
    if references_title is not None:
        appendix_before_ref = next(
            (
                b for b in blocks
                if b.type == "title" and is_appendix_title(b.text) and b.index < references_title.index
            ),
            None,
        )
    appendix_after_ref = None
    if references_title is not None:
        appendix_after_ref = next(
            (
                b for b in blocks
                if b.type == "title" and b.index > references_title.index and not is_references_title(b.text)
            ),
            None,
        )
    acknowledgement_before_ref = None
    if references_title is not None:
        acknowledgement_before_ref = next(
            (
                b for b in blocks
                if b.type == "title"
                and is_acknowledgement_title(b.text)
                and b.index < references_title.index
                and (abstract_start is None or b.index > abstract_start)
            ),
            None,
        )
    search_end = (
        min(
            value
            for value in [
                appendix_before_ref.index if appendix_before_ref is not None else None,
                acknowledgement_before_ref.index if acknowledgement_before_ref is not None else None,
                references_title.index if references_title is not None else None,
                len(blocks),
            ]
            if value is not None
        )
    )
    body_start = None
    search_start = abstract_start if abstract_start is not None else 0
    title_candidates = []
    for block in blocks:
        if block.index <= search_start:
            continue
        if block.index >= search_end:
            break
        if block.type == "title" and is_valid_body_title(block):
            title_candidates.append(block)
    numbered_candidate = next((b for b in title_candidates if heading_number(b.text)), None)
    if numbered_candidate is not None:
        body_start = numbered_candidate.index
    elif title_candidates:
        body_start = title_candidates[0].index
    keyword_start = None
    if abstract_start is not None and body_start is not None:
        keyword = next(
            (
                b for b in blocks
                if abstract_start <= b.index < body_start and is_keyword_like_block(b)
            ),
            None,
        )
        keyword_start = keyword.index if keyword else None
    return {
        "abstract_title": abstract_title.index if abstract_title else None,
        "abstract_paragraph": abstract_para.index if abstract_para else None,
        "abstract_start": abstract_start,
        "body_start": body_start,
        "keyword_start": keyword_start,
        "appendix_before_references_start": appendix_before_ref.index if appendix_before_ref else None,
        "appendix_after_references_start": appendix_after_ref.index if appendix_after_ref else None,
        "appendix_start": (
            appendix_before_ref.index
            if appendix_before_ref
            else appendix_after_ref.index if appendix_after_ref else None
        ),
        "acknowledgement_start": acknowledgement_before_ref.index if acknowledgement_before_ref else None,
        "references_start": references_title.index if references_title else None,
    }


def is_valid_body_title(block: FlatBlock) -> bool:
    if block.type != "title" or not block.text:
        return False
    if is_special_title(block.text) or is_keyword_marker(block.text) or is_acknowledgement_title(block.text):
        return False
    return True


def is_keyword_like_block(block: FlatBlock) -> bool:
    return block.type in {"title", "paragraph", "list"} and is_keyword_marker(block.text)


def first_existing_boundary(*values: int | None, default: int = 10**12) -> int:
    candidates = [value for value in values if value is not None]
    return min(candidates) if candidates else default


def region_for_block(block: FlatBlock, boundaries: dict[str, int | None]) -> str | None:
    idx = block.index
    abstract_start = boundaries["abstract_start"]
    body_start = boundaries["body_start"]
    keyword_start = boundaries["keyword_start"]
    appendix_before_ref = boundaries["appendix_before_references_start"]
    appendix_after_ref = boundaries["appendix_after_references_start"]
    acknowledgement_start = boundaries["acknowledgement_start"]
    references_start = boundaries["references_start"]
    if abstract_start is not None and idx >= abstract_start:
        abstract_end = first_existing_boundary(
            keyword_start,
            body_start,
            appendix_before_ref,
            acknowledgement_start,
            references_start,
        )
        if idx < abstract_end:
            return "abstract"
    if appendix_before_ref is not None and idx >= appendix_before_ref:
        app_end = references_start or 10**12
        if idx < app_end:
            return "appendix"
    if body_start is not None and idx >= body_start:
        body_end = first_existing_boundary(appendix_before_ref, acknowledgement_start, references_start)
        if idx < body_end:
            return "body"
    if appendix_after_ref is not None and idx >= appendix_after_ref:
        return "appendix"
    if references_start is not None and idx > references_start:
        if appendix_after_ref is not None and idx >= appendix_after_ref:
            return "appendix"
        return "reference"
    return None


def build_toc(blocks: list[FlatBlock], boundaries: dict[str, int | None]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections: list[dict[str, Any]] = []
    used_section_ids: set[str] = set()
    abstract_start = boundaries["abstract_start"]
    body_start = boundaries["body_start"]
    keyword_start = boundaries["keyword_start"]
    appendix_before_ref = boundaries["appendix_before_references_start"]
    appendix_after_ref = boundaries["appendix_after_references_start"]
    appendix_start = boundaries["appendix_start"]
    acknowledgement_start = boundaries["acknowledgement_start"]
    references_start = boundaries["references_start"]
    if abstract_start is not None:
        section = {
            "section_id": "sec_abstract",
            "parent_id": None,
            "number": None,
            "title": "Abstract",
            "region": "abstract",
            "order": len(sections),
            "level": 1,
            "path": ["Abstract"],
            "start_block_index": abstract_start,
            "end_block_index": first_existing_boundary(
                keyword_start,
                body_start,
                appendix_before_ref,
                acknowledgement_start,
                references_start,
                default=len(blocks),
            )
            - 1,
        }
        sections.append(section)
        used_section_ids.add(section["section_id"])
    if body_start is not None:
        body_end = first_existing_boundary(appendix_before_ref, acknowledgement_start, references_start, default=len(blocks))
        body_titles = [
            b for b in blocks
            if body_start <= b.index < body_end and is_valid_body_title(b)
        ]
        has_numbered_system = any(heading_number(b.text) for b in body_titles)
        number_to_section: dict[str, dict[str, Any]] = {}
        for block in body_titles:
            number = heading_number(block.text)
            if has_numbered_system and not number:
                continue
            base_section_id = section_id_for_title(block.text, len(sections))
            section_id = unique_section_id(base_section_id, used_section_ids)
            parent_id = None
            parent_path: list[str] = []
            if number and "." in number:
                parent_number = number.rsplit(".", 1)[0]
                parent = number_to_section.get(parent_number)
                if parent:
                    parent_id = parent["section_id"]
                    parent_path = list(parent["path"])
            title = strip_heading_number(block.text) if number else block.text.strip()
            display_title = f"{number} {title}" if number else title
            level = number.count(".") + 1 if number else block_level(block, 1)
            section = {
                "section_id": section_id,
                "parent_id": parent_id,
                "number": number,
                "title": title,
                "region": "body",
                "order": len(sections),
                "level": level,
                "path": parent_path + [display_title],
                "start_block_index": block.index,
                "end_block_index": body_end - 1,
            }
            sections.append(section)
            used_section_ids.add(section_id)
            if number:
                number_to_section[number] = section
    if appendix_start is not None:
        section = {
            "section_id": unique_section_id("sec_appendix", used_section_ids),
            "parent_id": None,
            "number": None,
            "title": "Appendix",
            "region": "appendix",
            "order": len(sections),
            "level": 1,
            "path": ["Appendix"],
            "start_block_index": appendix_start,
            "end_block_index": (
                (references_start - 1)
                if appendix_before_ref is not None and references_start is not None
                else len(blocks) - 1
            ),
        }
        sections.append(section)
        used_section_ids.add(section["section_id"])
    finalize_section_ends(sections)
    return sections, sections_to_tree(sections)


def finalize_section_ends(sections: list[dict[str, Any]]) -> None:
    ordered = sorted(sections, key=lambda s: s["start_block_index"])
    for index, section in enumerate(ordered):
        current_end = section["end_block_index"]
        section_level = int(section.get("level") or 1)
        for candidate in ordered[index + 1:]:
            if candidate["region"] != section["region"]:
                if candidate["start_block_index"] > section["start_block_index"]:
                    current_end = min(current_end, candidate["start_block_index"] - 1)
                    break
                continue
            candidate_level = int(candidate.get("level") or 1)
            if candidate_level <= section_level:
                current_end = min(current_end, candidate["start_block_index"] - 1)
                break
        section["end_block_index"] = current_end


def unique_section_id(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    counter = 2
    while f"{base}_{counter}" in used:
        counter += 1
    return f"{base}_{counter}"


def section_id_for_title(text: str, order: int) -> str:
    number = heading_number(text)
    if number:
        return "sec_" + number.replace(".", "_")
    slug = slugify_title(text, fallback=f"s{order:03d}").lower()
    return f"sec_{slug}"


def sections_to_tree(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = {
        s["section_id"]: {k: v for k, v in s.items() if k != "parent_id"} | {"children": []}
        for s in sections
    }
    roots: list[dict[str, Any]] = []
    for section in sections:
        node = nodes[section["section_id"]]
        parent_id = section.get("parent_id")
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)
    return roots


def current_section_id(block: FlatBlock, sections: list[dict[str, Any]], region: str) -> str | None:
    if region == "abstract":
        return "sec_abstract"
    region_sections = [
        s for s in sections
        if s["region"] == region and s["start_block_index"] <= block.index
    ]
    if not region_sections:
        return None
    return max(region_sections, key=lambda s: s["start_block_index"])["section_id"]


def section_paths(sections: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {section["section_id"]: list(section.get("path") or []) for section in sections}


def build_blocks(blocks: list[FlatBlock], boundaries: dict[str, int | None], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    section_start_blocks = {section["start_block_index"] for section in sections}
    paths_by_id = section_paths(sections)
    for block in blocks:
        region = region_for_block(block, boundaries)
        if region not in {"abstract", "body", "appendix"}:
            continue
        if block.type in IGNORED_TYPES:
            continue
        if is_keyword_like_block(block) or (block.type == "title" and is_acknowledgement_title(block.text)):
            continue
        if block.type == "list" and block.raw.get("content", {}).get("list_type") == "reference_list":
            continue
        if block.type == "title" and region == "body" and block.index not in section_start_blocks:
            if heading_number(block.text):
                continue
        text = block_text_for_output(block, boundaries).strip()
        if not text:
            continue
        section_id = current_section_id(block, sections, region)
        if section_id is None:
            continue
        row = {
            "block_id": f"b{block.index:06d}",
            "order": len(output),
            "region": region,
            "type": block.type,
            "text": text,
            "page": block.page,
            "bbox": block.bbox,
            "section_id": section_id,
            "section_path": paths_by_id.get(section_id, []),
        }
        row.update(structured_block_fields(block))
        output.append(row)
    return output


def block_text_for_output(block: FlatBlock, boundaries: dict[str, int | None]) -> str:
    if block.index == boundaries.get("abstract_paragraph") and starts_with_abstract(block.text):
        return strip_abstract_prefix(block.text)
    return block.text


def source_path_from_content(content: dict[str, Any]) -> str | None:
    source = content.get("image_source")
    if isinstance(source, dict) and isinstance(source.get("path"), str):
        return source["path"]
    return None


def structured_block_fields(block: FlatBlock) -> dict[str, Any]:
    content = block.raw.get("content") or {}
    if block.type == "image":
        caption = pieces_to_text(content.get("image_caption")).strip()
        fields: dict[str, Any] = {"caption": caption}
        source_path = source_path_from_content(content)
        if source_path:
            fields["source_path"] = source_path
        return fields
    if block.type == "table":
        caption = pieces_to_text(content.get("table_caption")).strip()
        html = str(content.get("html") or "")
        fields = {"caption": caption, "html": html}
        source_path = source_path_from_content(content)
        if source_path:
            fields["source_path"] = source_path
        return fields
    return {}


def build_references(blocks: list[FlatBlock], boundaries: dict[str, int | None]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for block in blocks:
        if region_for_block(block, boundaries) != "reference":
            continue
        if block.type != "list" or block.raw.get("content", {}).get("list_type") != "reference_list":
            continue
        for item in block.raw.get("content", {}).get("list_items") or []:
            raw_text = pieces_to_text(item.get("item_content") if isinstance(item, dict) else item).strip()
            if not raw_text:
                continue
            ref_index = reference_index(raw_text, len(output) + 1)
            output.append(
                {
                    "reference_id": f"ref_{ref_index:03d}",
                    "ref_index": ref_index,
                    "raw_text": raw_text,
                    "page": block.page,
                    "source_block_id": f"b{block.index:06d}",
                }
            )
    return output


def reference_index(raw_text: str, fallback_index: int) -> int:
    match = re.match(r"^\s*(\[(\d+)\]|\(?(\d+)\)?[.)])\s*", raw_text)
    if not match:
        return fallback_index
    number = match.group(2) or match.group(3)
    return int(number)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_chunks(
    blocks: list[dict[str, Any]],
    metadata: dict[str, Any],
    paper_id: str,
    *,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    target_chars = max(1, target_chars)
    overlap_chars = max(0, overlap_chars)
    chunks: list[dict[str, Any]] = []
    for section_blocks in group_chunk_blocks(blocks):
        current: list[dict[str, Any]] = []
        current_len = 0
        previous_text = ""
        for block in section_blocks:
            block_text = chunk_block_text(block)
            if not block_text:
                continue
            block_len = len(block_text)
            if current and current_len + block_len + 2 > target_chars:
                chunk = make_chunk(chunks, paper_id, metadata, current, previous_text, overlap_chars)
                chunks.append(chunk)
                previous_text = chunk["text"]
                current = []
                current_len = 0
            current.append(block | {"_chunk_text": block_text})
            current_len += block_len + (2 if current_len else 0)
            if block_len > target_chars:
                chunk = make_chunk(chunks, paper_id, metadata, current, previous_text, overlap_chars)
                chunks.append(chunk)
                previous_text = chunk["text"]
                current = []
                current_len = 0
        if current:
            chunk = make_chunk(chunks, paper_id, metadata, current, previous_text, overlap_chars)
            chunks.append(chunk)
    return chunks


def group_chunk_blocks(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_key: tuple[str | None, str | None] | None = None
    current: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda b: int(b.get("order") or 0)):
        region = block.get("region")
        if region not in {"abstract", "body"}:
            continue
        key = (str(region), str(block.get("section_id") or ""))
        if current and key != current_key:
            groups.append(current)
            current = []
        current_key = key
        current.append(block)
    if current:
        groups.append(current)
    return groups


def chunk_block_text(block: dict[str, Any]) -> str:
    text = str(block.get("text") or "").strip()
    if not text:
        return ""
    if block.get("type") == "equation_interline":
        if len(text) > MAX_CHUNK_EQUATION_CHARS:
            return ""
        return f"Equation: {text}"
    return text


def make_chunk(
    chunks: list[dict[str, Any]],
    paper_id: str,
    metadata: dict[str, Any],
    blocks: list[dict[str, Any]],
    previous_text: str,
    overlap_chars: int,
) -> dict[str, Any]:
    chunk_index = len(chunks)
    text = "\n\n".join(str(block.get("_chunk_text") or "").strip() for block in blocks).strip()
    section_path = list(blocks[0].get("section_path") or [])
    pages = sorted({page for block in blocks if isinstance((page := block.get("page")), int)})
    block_ids = [str(block["block_id"]) for block in blocks if block.get("block_id")]
    overlap_text = previous_text[-overlap_chars:].strip() if overlap_chars and previous_text else ""
    embedding_body = "\n\n".join(part for part in [overlap_text, text] if part)
    embedding_text = build_embedding_text(
        title=str(metadata.get("title") or ""),
        section_path=section_path,
        text=embedding_body,
    )
    return {
        "chunk_id": f"{paper_id}::chunk_{chunk_index:04d}",
        "paper_id": paper_id,
        "chunk_index": chunk_index,
        "region": blocks[0].get("region"),
        "section_id": blocks[0].get("section_id"),
        "section_path": section_path,
        "pages": pages,
        "block_ids": block_ids,
        "text": text,
        "embedding_text": embedding_text,
        "char_count": len(text),
    }


def build_embedding_text(title: str, section_path: list[Any], text: str) -> str:
    section = " > ".join(str(part) for part in section_path if part)
    prefix = [f"Paper: {title.strip()}"]
    if section:
        prefix.append(f"Section: {section}")
    return "\n".join(prefix).strip() + "\n\n" + text.strip()


def extract_paper_data(
    mineru_output_dir: Path,
    paper_data_dir: Path,
    metadata: dict[str, Any],
    *,
    chunk_target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> ExtractionResult:
    content_path = mineru_output_dir / "content_list_v2.json"
    pages = load_content_list_v2(content_path)
    flat_blocks = flatten_pages(pages)
    title = metadata.get("title") or extract_title(flat_blocks)
    if not title:
        raise ValueError(f"No title found in {content_path}")
    boundaries = find_region_boundaries(flat_blocks)
    warnings = extraction_warnings(boundaries)
    sections, tree = build_toc(flat_blocks, boundaries)
    blocks = build_blocks(flat_blocks, boundaries, sections)
    paper_id = paper_data_dir.name
    metadata_for_chunks = {"title": title}
    chunks = build_chunks(
        blocks,
        metadata_for_chunks,
        paper_id,
        target_chars=chunk_target_chars,
        overlap_chars=chunk_overlap_chars,
    )
    references = build_references(flat_blocks, boundaries)

    if paper_data_dir.exists():
        shutil.rmtree(paper_data_dir)
    paper_data_dir.mkdir(parents=True, exist_ok=True)

    metadata_out = {
        "title": title,
        "author": metadata.get("author") or [],
        "year": metadata.get("year"),
        "venue": metadata.get("venue"),
        "pdf_path": metadata.get("pdf_path"),
    }
    toc = {"sections": sections, "tree": tree}
    write_json(paper_data_dir / "metadata.json", metadata_out)
    write_json(paper_data_dir / "toc.json", toc)
    write_jsonl(paper_data_dir / "blocks.jsonl", blocks)
    write_jsonl(paper_data_dir / "chunks.jsonl", chunks)
    write_jsonl(paper_data_dir / "references.jsonl", references)
    return ExtractionResult(
        title=title,
        paper_data_dir=paper_data_dir,
        metadata_path=paper_data_dir / "metadata.json",
        toc_path=paper_data_dir / "toc.json",
        blocks_path=paper_data_dir / "blocks.jsonl",
        chunks_path=paper_data_dir / "chunks.jsonl",
        references_path=paper_data_dir / "references.jsonl",
        sections=sections,
        block_count=len(blocks),
        chunk_count=len(chunks),
        reference_count=len(references),
        warnings=warnings,
    )


def extraction_warnings(boundaries: dict[str, int | None]) -> list[str]:
    warnings: list[str] = []
    if boundaries.get("abstract_title") is None and boundaries.get("abstract_paragraph") is None:
        warnings.append("Abstract marker not found")
    return warnings
