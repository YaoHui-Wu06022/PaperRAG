from __future__ import annotations

from typing import Any

from ..config import Settings
from .data.chunks import ChunkDocument, read_jsonl


def context_unit(settings: Settings, candidate: Any, block_window: int) -> dict[str, Any]:
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
    start = max(0, min(hit_positions) - block_window)
    end = min(len(section_blocks), max(hit_positions) + block_window + 1)
    return [
        block_to_evidence(block, str(block.get("block_id") or "") in hit_ids)
        for block in section_blocks[start:end]
    ]


def block_to_evidence(block: dict[str, Any], is_hit_block: bool) -> dict[str, Any]:
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
