from __future__ import annotations

import json
from pathlib import Path

from paper_rag.catalog.service import rebuild_catalog, search_chunks
from paper_rag.config import Settings
from paper_rag.reading.service import read_fulltext
from paper_rag.catalog.chunks import build_chunks


def make_source(tmp_path: Path) -> Settings:
    settings = Settings.load(tmp_path)
    source = settings.arxiv_data_dir / "1706.03762"
    (source / "mineru").mkdir(parents=True)
    (source / "paper.pdf").write_bytes(b"%PDF-test")
    (source / "metadata.json").write_text(json.dumps({"base_id": "1706.03762", "canonical_id": "1706.03762v7", "title": "Attention", "abstract": "summary"}), encoding="utf-8")
    (source / "mineru" / "full.md").write_text("Abstract text\nContent text", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256((source / "paper.pdf").read_bytes()).hexdigest()
    (source / "mineru" / "manifest.json").write_text(json.dumps({"canonical_id": "1706.03762v7", "source_sha256": digest, "model_version": settings.mineru_model_version, "language": settings.mineru_language}), encoding="utf-8")
    (source / "mineru" / "content_list.json").write_text(json.dumps([
        {"type": "text", "text": "Title and authors", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "Abstract attention evidence", "page_idx": 0},
        {"type": "text", "text": "1 Introduction", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "Transformer content evidence", "page_idx": 1},
        {"type": "text", "text": "References", "text_level": 2, "page_idx": 2},
        {"type": "text", "text": "Metadata reference should remain searchable", "page_idx": 2},
    ]), encoding="utf-8")
    return settings


def test_chunk_regions_skip_frontmatter_and_index_body(tmp_path: Path):
    settings = make_source(tmp_path)
    result = rebuild_catalog(settings)
    assert result["chunks"] == 2
    items = search_chunks(settings, "Transformer")
    assert items and items[0]["section_path"][0] == "content"
    assert all("Title and authors" not in item["text"] for item in search_chunks(settings, "Title"))
    export = json.loads((settings.arxiv_data_dir / "1706.03762" / "mineru" / "chunks.json").read_text(encoding="utf-8"))
    assert export["chunk_count"] == 2
    assert len(export["chunks"]) == 2


def test_fulltext_pagination_is_unicode_bounded(tmp_path: Path):
    settings = make_source(tmp_path)
    from paper_rag.catalog.service import rebuild_catalog
    rebuild_catalog(settings)
    result = read_fulltext(settings, "1706.03762", 0, 5)
    assert result["status"] == "ok"
    assert result["total_chars"] > 5
    assert result["truncated"] is True
    assert result["next_offset"] == 5


def test_chunks_never_cross_section_boundaries():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "5 Training", "text_level": 1},
        {"type": "text", "text": "short training section"},
        {"type": "text", "text": "5.1 Details", "text_level": 2},
        {"type": "text", "text": "short detail section"},
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    assert [chunk.section_label for chunk in chunks] == ["abstract", "5 Training", "5.1 Details"]
    assert all("short training section" not in chunk.text or chunk.section_label == "5 Training" for chunk in chunks)


def test_structured_chunk_keeps_reading_order_and_neighbor_context():
    chunks, warnings = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "4 Method", "text_level": 1},
        {"type": "text", "text": "The method updates the matrix."},
        {"type": "equation", "text": "$$x = y$$", "text_format": "latex"},
        {"type": "text", "text": "The equation is used during training."},
        {"type": "text", "text": "References", "text_level": 1},
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    equation = next(item for item in chunks if item.type == "equation")
    assert equation.text == "$$x = y$$"
    assert equation.retrieval_text.startswith("content\n4 Method")
    assert not equation.retrieval_text.startswith("1706.03762")
    assert "The method updates the matrix." in equation.retrieval_text
    assert "The equation is used during training." in equation.retrieval_text
    assert equation.source_blocks[0]["text_format"] == "latex"
    assert not warnings
