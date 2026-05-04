"""content 命中 chunk 后的 block 窗口扩展。"""

from __future__ import annotations

from typing import Any

from ....config import Settings
from ...data.chunks_load import ChunkDocument, read_jsonl


def context_unit(settings: Settings, candidate: Any, block_window: int) -> dict[str, Any]:
    """把命中的 chunk 扩展成回答层可用的上下文单元。"""
    document = candidate.document
    return {
        "chunk_id": document.chunk_id,
        "paper_id": document.paper_id,
        "title": document.title,
        "section_path": document.section_path,
        "pages": document.pages,
        "score": candidate.score,
        "sources": candidate.sources,
        "chunk_text": document.text,
        "expanded_blocks": expand_blocks(settings, document, block_window),
    }


def expand_blocks(settings: Settings, document: ChunkDocument, block_window: int) -> list[dict[str, Any]]:
    """按命中 chunk 的 block_ids，在同一 section 内扩展前后窗口。"""
    blocks_path = settings.paper_data_dir / document.paper_id / "blocks.jsonl"
    if not blocks_path.exists() or not document.block_ids:
        return []
    blocks = read_jsonl(blocks_path)
    section_blocks = [
        block for block in blocks
        if str(block.get("section_id") or "") == document.section_id
    ]
    hit_ids = set(document.block_ids)
    hit_positions = [
        index for index, block in enumerate(section_blocks)
        if str(block.get("block_id") or "") in hit_ids
    ]
    if not hit_positions:
        return []
    # 只在同一 section 内扩展，避免把相邻页面但语义无关的段落混入 context。
    start = max(0, min(hit_positions) - block_window)
    end = min(len(section_blocks), max(hit_positions) + block_window + 1)
    return [
        block_to_evidence(block, str(block.get("block_id") or "") in hit_ids)
        for block in section_blocks[start:end]
    ]


def block_to_evidence(block: dict[str, Any], is_hit_block: bool) -> dict[str, Any]:
    """裁剪 block 字段，并标记是否为原始命中 block。"""
    return {
        "block_id": block.get("block_id"),
        "order": block.get("order"),
        "region": block.get("region"),
        "type": block.get("type"),
        "text": block.get("text"),
        "page": block.get("page"),
        "section_id": block.get("section_id"),
        "section_path": block.get("section_path") or [],
        "is_hit_block": is_hit_block,
    }
