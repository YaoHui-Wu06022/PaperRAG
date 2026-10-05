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


def test_arabic_and_roman_sibling_headings_reset_section_path():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "I. Introduction", "text_level": 2},
        {"type": "text", "text": "intro text"},
        {"type": "text", "text": "II. Related Work", "text_level": 2},
        {"type": "text", "text": "related text"},
        {"type": "text", "text": "3. Deep Residual Learning", "text_level": 2},
        {"type": "text", "text": "residual text"},
        {"type": "text", "text": "3.1. Residual Learning", "text_level": 3},
        {"type": "text", "text": "detail text"},
    ], paper_id="1512.03385", canonical_id="1512.03385v1", content_hash="test")

    assert [chunk.section_path for chunk in chunks] == [
        ("abstract",),
        ("content", "I. Introduction"),
        ("content", "II. Related Work"),
        ("content", "3. Deep Residual Learning"),
        ("content", "3. Deep Residual Learning", "3.1. Residual Learning"),
    ]


def test_inline_and_implicit_abstracts_are_indexed():
    inline, _ = build_chunks([
        {"type": "text", "text": "Paper title", "text_level": 1},
        {"type": "text", "text": "Abstract—This is the inline abstract body."},
        {"type": "text", "text": "I. Introduction", "text_level": 2},
        {"type": "text", "text": "Main content."},
    ], paper_id="1905.03175", canonical_id="1905.03175v1", content_hash="test")
    assert inline[0].region == "abstract"
    assert inline[0].text == "This is the inline abstract body."

    implicit, _ = build_chunks([
        {"type": "text", "text": "Paper title", "text_level": 1},
        {"type": "text", "text": "Authors"},
        {"type": "text", "text": "This is an abstract paragraph without an Abstract heading. " * 8},
        {"type": "text", "text": "1 Introduction", "text_level": 2},
        {"type": "text", "text": "Main content."},
    ], paper_id="2203.15556", canonical_id="2203.15556v1", content_hash="test")
    assert implicit[0].region == "abstract"
    assert implicit[1].section_path == ("content", "1 Introduction")


def test_contents_body_is_excluded_from_content_chunks():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "Contents", "text_level": 2},
        {"type": "text", "text": "1 Introduction 3  2 Approach 6"},
        {"type": "text", "text": "1 Introduction", "text_level": 2},
        {"type": "text", "text": "real introduction"},
    ], paper_id="2005.14165", canonical_id="2005.14165v4", content_hash="test")
    assert [chunk.text for chunk in chunks] == ["abstract text", "real introduction"]


def test_paper_title_caption_is_removed_without_affecting_normal_captions():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "A. Details", "text_level": 2},
        {"type": "table", "table_caption": ["Paper Title"], "table_body": "<table><tr><td>Value</td></tr></table>"},
        {"type": "table", "table_caption": ["Table 1. Results"], "table_body": "<table><tr><td>Value</td></tr></table>"},
    ], paper_id="1902.00751", canonical_id="1902.00751v2", content_hash="test", document_title="Paper Title")
    tables = [chunk for chunk in chunks if chunk.type == "table"]
    assert "Paper Title" not in tables[0].text
    assert "Paper Title" not in tables[0].retrieval_text
    assert "Table 1. Results" in tables[1].text
    assert "Table 1. Results" in tables[1].retrieval_text


def test_appendix_ref_text_is_normalized_to_text():
    chunks, warnings = build_chunks([
        {"type": "text", "text": "References", "text_level": 1},
        {"type": "ref_text", "text": "[1] Reference"},
        {"type": "text", "text": "A Examples", "text_level": 2},
        {"type": "ref_text", "text": "Dialogue content from the appendix."},
    ], paper_id="2302.13971", canonical_id="2302.13971v1", content_hash="test")
    appendix = [chunk for chunk in chunks if chunk.region == "appendix"]
    assert appendix[0].type == "text"
    assert appendix[0].text == "Dialogue content from the appendix."
    assert not any("unknown" in warning for warning in warnings)


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


def test_table_text_keeps_html_and_retrieval_text_is_structured_plain_text():
    table_html = "<table><tr><th>Model</th><th>Accuracy</th><th>Params</th></tr><tr><td><b>BERT</b></td><td>89.2</td><td>110M</td></tr></table>"
    chunks, warnings = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "1 Method", "text_level": 2},
        {"type": "text", "text": "The table compares model quality."},
        {
            "type": "table",
            "text": table_html,
            "table_caption": ["Table 2: Performance comparison."],
            "table_body": table_html,
            "table_footnote": ["Accuracy is measured on the validation set."],
        },
        {"type": "text", "text": "The result favors BERT."},
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    table = next(item for item in chunks if item.type == "table")
    assert table.text == table_html
    assert "<table>" not in table.retrieval_text
    assert "<td>" not in table.retrieval_text
    assert "Table 2: Performance comparison." in table.retrieval_text
    assert "Columns: Model | Accuracy | Params" in table.retrieval_text
    assert "Model: BERT | Accuracy: 89.2 | Params: 110M" in table.retrieval_text
    assert "Footnote: Accuracy is measured on the validation set." in table.retrieval_text
    assert "The table compares model quality." in table.retrieval_text
    assert "The result favors BERT." in table.retrieval_text
    assert not warnings


def test_table_fallback_fields_and_malformed_html_are_safe():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "1 Results", "text_level": 2},
        {
            "type": "table",
            "table_caption": ["Table &amp; comparison"],
            "table_body": "<table><script>ignore()</script><style>.x{}</style><tr><td>Model</td><td>Score</td></tr><tr><td><b>BERT</b>",
            "table_footnote": [],
        },
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    table = next(item for item in chunks if item.type == "table")
    assert "<table>" in table.text
    assert "Table &amp; comparison" in table.text
    assert "<table>" not in table.retrieval_text
    assert "Table & comparison" in table.retrieval_text
    assert "Model | Score" in table.retrieval_text
    assert "BERT" in table.retrieval_text
    assert "ignore" not in table.retrieval_text
    assert ".x" not in table.retrieval_text


def test_table_without_explicit_header_keeps_row_order_and_missing_cells():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "1 Results", "text_level": 2},
        {
            "type": "table",
            "table_body": "<table><tr><td colspan='2'>Metrics</td></tr><tr><td>BERT</td><td>89.2</td></tr><tr><td>RoBERTa</td></tr></table>",
        },
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    table = next(item for item in chunks if item.type == "table")
    assert "Columns:" not in table.retrieval_text
    assert "Metrics" in table.retrieval_text
    assert "BERT | 89.2" in table.retrieval_text
    assert "RoBERTa" in table.retrieval_text
    assert "RoBERTa |" not in table.retrieval_text


def test_appendix_after_references_is_indexed_as_appendix():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "References", "text_level": 1},
        {"type": "ref_text", "text": "[1] Reference"},
        {"type": "text", "text": "A ADDITIONAL DETAILS", "text_level": 2},
        {"type": "text", "text": "Appendix evidence"},
        {"type": "text", "text": "A.1 MORE DETAILS", "text_level": 2},
        {"type": "text", "text": "Nested appendix evidence"},
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    appendix = [item for item in chunks if item.region == "appendix"]
    assert len(appendix) == 2
    assert appendix[0].section_path == ("appendix", "A ADDITIONAL DETAILS")
    assert appendix[1].section_path == ("appendix", "A ADDITIONAL DETAILS", "A.1 MORE DETAILS")


def test_lettered_appendix_headings_are_siblings():
    chunks, _ = build_chunks([
        {"type": "text", "text": "References", "text_level": 1},
        {"type": "ref_text", "text": "[1] Reference"},
        {"type": "text", "text": "A. Training dataset", "text_level": 2},
        {"type": "text", "text": "A evidence"},
        {"type": "text", "text": "B. Optimal cosine cycle length", "text_level": 2},
        {"type": "text", "text": "B evidence"},
        {"type": "text", "text": "D. Details on the scaling analyses", "text_level": 2},
        {"type": "text", "text": "D evidence"},
        {"type": "text", "text": "D.1. Approach 1", "text_level": 2},
        {"type": "text", "text": "D1 evidence"},
        {"type": "text", "text": "C XXX", "text_level": 2},
        {"type": "text", "text": "C evidence"},
        {"type": "text", "text": "C.1 Details", "text_level": 2},
        {"type": "text", "text": "C1 evidence"},
    ], paper_id="2203.15556", canonical_id="2203.15556v1", content_hash="test")
    appendix = [item for item in chunks if item.region == "appendix"]
    assert appendix[0].section_path == ("appendix", "A. Training dataset")
    assert appendix[1].section_path == ("appendix", "B. Optimal cosine cycle length")
    assert appendix[2].section_path == ("appendix", "D. Details on the scaling analyses")
    assert appendix[3].section_path == ("appendix", "D. Details on the scaling analyses", "D.1. Approach 1")
    assert appendix[4].section_path == ("appendix", "C XXX")
    assert appendix[5].section_path == ("appendix", "C XXX", "C.1 Details")


def test_appendix_prompt_blocks_do_not_become_nested_headings():
    chunks, _ = build_chunks([
        {"type": "text", "text": "References", "text_level": 1},
        {"type": "ref_text", "text": "[1] Reference"},
        {"type": "text", "text": "D Generations from LLaMA-I", "text_level": 2},
        {"type": "text", "text": "write a conversation between the sun and pluto", "text_level": 2},
        {"type": "text", "text": "The sun and Pluto discuss their orbits."},
        {"type": "text", "text": "How do I send an HTTP request in Javascript?", "text_level": 2},
        {"type": "text", "text": "Use the Fetch API."},
    ], paper_id="2302.13971", canonical_id="2302.13971v1", content_hash="test")

    appendix = [item for item in chunks if item.region == "appendix"]
    assert len(appendix) == 1
    assert appendix[0].section_path == ("appendix", "D Generations from LLaMA-I")
    assert "How do I send an HTTP request in Javascript?" in appendix[0].text


def test_appendix_keeps_short_unnumbered_headings():
    chunks, _ = build_chunks([
        {"type": "text", "text": "References", "text_level": 1},
        {"type": "text", "text": "A Details", "text_level": 2},
        {"type": "text", "text": "Experimental Setup", "text_level": 2},
        {"type": "text", "text": "The setup details."},
    ], paper_id="2302.13971", canonical_id="2302.13971v1", content_hash="test")

    appendix = [item for item in chunks if item.region == "appendix"]
    assert appendix[0].section_path == ("appendix", "A Details", "Experimental Setup")


def test_acknowledgements_starts_a_new_top_level_section():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "9 Broader Impacts", "text_level": 2},
        {"type": "text", "text": "impact text"},
        {"type": "text", "text": "Acknowledgements", "text_level": 2},
        {"type": "text", "text": "We thank the team."},
    ], paper_id="2305.14314", canonical_id="2305.14314v1", content_hash="test")
    acknowledgement = next(item for item in chunks if item.text == "We thank the team.")
    assert acknowledgement.section_path == ("content", "Acknowledgements")
    assert acknowledgement.retrieval_text == "content\nAcknowledgements\nWe thank the team."


def test_implicit_appendix_after_acknowledgements_starts_at_lettered_heading():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "1 Introduction", "text_level": 2},
        {"type": "text", "text": "introduction text"},
        {"type": "text", "text": "Acknowledgement", "text_level": 2},
        {"type": "text", "text": "We thank the team."},
        {"type": "text", "text": "A1. Detailed Architectures", "text_level": 2},
        {"type": "text", "text": "Architecture evidence"},
        {"type": "text", "text": "A2. Detailed Experimental Settings", "text_level": 2},
        {"type": "text", "text": "Settings evidence"},
        {"type": "text", "text": "A2.1. Image classification", "text_level": 2},
        {"type": "text", "text": "Classification evidence"},
    ], paper_id="2103.14030", canonical_id="2103.14030v2", content_hash="test")
    acknowledgement = next(item for item in chunks if item.text == "We thank the team.")
    architecture = next(item for item in chunks if item.text == "Architecture evidence")
    settings = next(item for item in chunks if item.text == "Settings evidence")
    classification = next(item for item in chunks if item.text == "Classification evidence")
    assert acknowledgement.section_path == ("content", "Acknowledgement")
    assert architecture.section_path == ("appendix", "A1. Detailed Architectures")
    assert settings.section_path == ("appendix", "A2. Detailed Experimental Settings")
    assert classification.section_path == ("appendix", "A2. Detailed Experimental Settings", "A2.1. Image classification")


def test_contributions_stays_content_before_implicit_appendix():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "1 Introduction", "text_level": 2},
        {"type": "text", "text": "introduction text"},
        {"type": "text", "text": "Acknowledgements", "text_level": 2},
        {"type": "text", "text": "We thank the team."},
        {"type": "text", "text": "Contributions", "text_level": 2},
        {"type": "text", "text": "Contribution evidence"},
        {"type": "text", "text": "A Details of Common Crawl Filtering", "text_level": 2},
        {"type": "text", "text": "Appendix A evidence"},
        {"type": "text", "text": "B Details of Model Training", "text_level": 2},
        {"type": "text", "text": "Appendix B evidence"},
    ], paper_id="2005.14165", canonical_id="2005.14165v4", content_hash="test")
    contribution = next(item for item in chunks if item.text == "Contribution evidence")
    appendix_a = next(item for item in chunks if item.text == "Appendix A evidence")
    appendix_b = next(item for item in chunks if item.text == "Appendix B evidence")
    assert contribution.section_path == ("content", "Contributions")
    assert appendix_a.section_path == ("appendix", "A Details of Common Crawl Filtering")
    assert appendix_b.section_path == ("appendix", "B Details of Model Training")


def test_lettered_heading_before_acknowledgements_is_not_implicit_appendix():
    chunks, _ = build_chunks([
        {"type": "text", "text": "Abstract", "text_level": 1},
        {"type": "text", "text": "abstract text"},
        {"type": "text", "text": "A. Method", "text_level": 2},
        {"type": "text", "text": "Method evidence"},
        {"type": "text", "text": "References", "text_level": 2},
    ], paper_id="1706.03762", canonical_id="1706.03762v7", content_hash="test")
    method = next(item for item in chunks if item.text == "Method evidence")
    assert method.region == "content"
    assert method.section_path == ("content", "A. Method")
