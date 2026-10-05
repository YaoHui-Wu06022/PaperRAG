"""从 SQLite Chunk 构建稳定的 LlamaIndex TextNode。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from llama_index.core.schema import TextNode

from paper_rag.catalog.service import CatalogIndexNotReady
from paper_rag.config import Settings


NODE_METADATA_KEYS = (
    "chunk_id",
    "paper_id",
    "canonical_id",
    "ordinal",
    "region",
    "chapter_number",
    "chapter_title",
    "section_path",
    "section_label",
    "type",
    "page_start",
    "page_end",
    "content_hash",
    "source_blocks",
    "asset_refs",
)


def load_nodes(
    settings: Settings,
    *,
    paper_ids: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
) -> list[TextNode]:
    """读取 Catalog 中的 Chunk，并按稳定 ID 转成 TextNode。"""

    path = settings.paper_catalog_db_path
    if not path.is_file():
        raise CatalogIndexNotReady("catalog index is not ready; run paper_catalog_sync first")
    clauses = ["1=1"]
    params: list[Any] = []
    ids = [str(value) for value in (paper_ids or ()) if str(value)]
    selected_regions = [str(value) for value in (regions or ("abstract", "content", "appendix")) if str(value)]
    selected_regions = [value for value in selected_regions if value != "reference"]
    if ids:
        clauses.append("paper_id IN (" + ",".join("?" for _ in ids) + ")")
        params.extend(ids)
    if selected_regions:
        clauses.append("region IN (" + ",".join("?" for _ in selected_regions) + ")")
        params.extend(selected_regions)
    sql = """
        SELECT chunk_id, paper_id, canonical_id, ordinal, region,
               chapter_number, chapter_title, section_path, section_label,
               type, text, retrieval_text, page_start, page_end,
               source_blocks, asset_refs, content_hash
          FROM chunks
         WHERE """ + " AND ".join(clauses) + " ORDER BY ordinal"
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise CatalogIndexNotReady("chunk index is not ready; run paper_catalog_sync first") from exc
    return [chunk_row_to_node(row) for row in rows]


def chunk_row_to_node(row: tuple[Any, ...]) -> TextNode:
    """把 Catalog 行转换为不会触发二次切块的 TextNode。"""

    (
        chunk_id,
        paper_id,
        canonical_id,
        ordinal,
        region,
        chapter_number,
        chapter_title,
        section_path,
        section_label,
        kind,
        text,
        retrieval_text,
        page_start,
        page_end,
        source_blocks,
        asset_refs,
        content_hash,
    ) = row
    metadata = {
        "chunk_id": str(chunk_id),
        "paper_id": str(paper_id),
        "canonical_id": str(canonical_id),
        "ordinal": int(ordinal),
        "region": str(region),
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "section_path": _json_value(section_path, []),
        "section_label": section_label,
        "type": str(kind),
        "page_start": page_start,
        "page_end": page_end,
        "content_hash": str(content_hash),
        "source_blocks": _json_value(source_blocks, []),
        "asset_refs": _json_value(asset_refs, []),
        "content_text": str(text or ""),
    }
    return TextNode(
        id_=str(chunk_id),
        text=str(retrieval_text or text or ""),
        metadata=metadata,
    )


def node_metadata(node: TextNode) -> dict[str, Any]:
    """提取 MCP 所需的稳定 metadata，并还原 JSON 字段。"""

    result = {key: node.metadata.get(key) for key in NODE_METADATA_KEYS}
    result["text"] = node.metadata.get("content_text") or node.get_content()
    return result


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        parsed = json.loads(value or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return parsed


__all__ = ["NODE_METADATA_KEYS", "chunk_row_to_node", "load_nodes", "node_metadata"]
