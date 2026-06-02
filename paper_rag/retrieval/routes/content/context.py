"""content 命中 chunk 后的 block 窗口扩展。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.chunks import ChunkDocument, read_jsonl


def context_unit(
    settings: Settings,
    candidate: Any,
    block_window: int,
    *,
    include_expanded_blocks: bool = True,
) -> dict[str, Any]:
    """把命中的 chunk 扩展成回答层可用的上下文单元。"""
    chunk_document = candidate.chunk_document
    unit = {
        "chunk_id": chunk_document.chunk_id,
        "paper_id": chunk_document.paper_id,
        "title": chunk_document.title,
        "section_path": chunk_document.section_path,
        "pages": chunk_document.pages,
        "score": candidate.score,
        "sources": candidate.sources,
        "chunk_text": chunk_document.text,
    }
    if include_expanded_blocks:
        unit["expanded_blocks"] = expand_blocks(settings, chunk_document, block_window)
    return unit


def expand_blocks(settings: Settings, chunk_document: ChunkDocument, block_window: int) -> list[dict[str, Any]]:
    """按命中 chunk 的 block_ids，在同一 section 内扩展前后窗口。"""
    blocks_path = settings.paper_data_dir / chunk_document.paper_id / "blocks.jsonl"
    if not blocks_path.exists() or not chunk_document.block_ids:
        return []
    blocks = read_jsonl(blocks_path)
    section_blocks = [
        block for block in blocks
        if str(block.get("section_id") or "") == chunk_document.section_id
    ]
    hit_ids = set(chunk_document.block_ids)
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
