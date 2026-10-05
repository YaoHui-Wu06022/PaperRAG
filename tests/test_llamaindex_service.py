from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper_rag.catalog.service import rebuild_catalog
from paper_rag.config import Settings
from paper_rag.llamaindex.embedding import DashScopeEmbedding
from paper_rag.llamaindex.nodes import load_nodes
from paper_rag.llamaindex.service import index_status, retrieve


def make_index_fixture(tmp_path: Path, *, include_second: bool = False) -> Settings:
    settings = Settings.load(tmp_path)
    source = settings.arxiv_data_dir / "1706.03762"
    (source / "mineru").mkdir(parents=True)
    (source / "paper.pdf").write_bytes(b"%PDF-test")
    (source / "metadata.json").write_text(
        json.dumps({
            "base_id": "1706.03762",
            "canonical_id": "1706.03762v7",
            "title": "Attention",
            "abstract": "Transformer attention.",
            "categories": ["cs.CL"],
            "published_at": "2020-01-01T00:00:00Z",
        }),
        encoding="utf-8",
    )
    digest = hashlib.sha256((source / "paper.pdf").read_bytes()).hexdigest()
    (source / "mineru" / "full.md").write_text("Abstract\nattention evidence", encoding="utf-8")
    (source / "mineru" / "manifest.json").write_text(
        json.dumps({
            "canonical_id": "1706.03762v7",
            "source_sha256": digest,
            "model_version": settings.mineru_model_version,
            "language": settings.mineru_language,
        }),
        encoding="utf-8",
    )
    (source / "mineru" / "content_list.json").write_text(
        json.dumps([
            {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "attention evidence", "page_idx": 1},
        ]),
        encoding="utf-8",
    )
    if include_second:
        _add_second_fixture(settings)
    rebuild_catalog(settings)
    return settings


def test_chunk_nodes_preserve_stable_id_and_metadata(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    nodes = load_nodes(settings)
    assert len(nodes) == 1
    assert nodes[0].node_id == nodes[0].metadata["chunk_id"]
    assert nodes[0].metadata["content_text"] == "attention evidence"
    assert nodes[0].metadata["page_start"] == 1


def test_lexical_retrieval_returns_citations_without_milvus(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "attention", mode="lexical")
    assert result["status"] == "ok"
    item = result["data"]["items"][0]
    assert item["source_id"] == "S1"
    assert item["page_start_display"] == 2


def test_single_rag_tool_returns_client_side_context(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "attention", task="fact", limit=2, max_chars=100, mode="lexical")
    assert result["status"] == "ok"
    assert "[S1]" in result["data"]["context_text"]
    assert result["read_only"] is True


def test_embedding_adapter_uses_existing_client(monkeypatch, tmp_path: Path):
    settings = Settings.load(tmp_path)
    monkeypatch.setattr(
        "paper_rag.llamaindex.embedding.DashScopeEmbeddingClient.embed",
        lambda _self, inputs: [[float(len(value)), 1.0] for value in inputs],
    )
    embedding = DashScopeEmbedding(settings)
    assert embedding.get_text_embedding("abc") == [3.0, 1.0]
    assert embedding.get_query_embedding("xy") == [2.0, 1.0]


def test_index_status_is_not_ready_before_rebuild(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = index_status(settings)
    assert result["status"] == "ok"
    assert result["data"]["index_ready"] is False


def test_retrieve_applies_metadata_filters_before_chunk_search(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "attention", filters={"category": "cs.CL"}, task="fact", mode="lexical")
    assert result["status"] == "ok"
    assert result["data"]["papers"][0]["paper_id"] == "1706.03762"


def test_retrieve_rejects_reference_region(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "attention", task="fact", mode="lexical", regions=["reference"])
    assert result["status"] == "invalid_input"


def _add_second_fixture(settings: Settings) -> None:
    source = settings.arxiv_data_dir / "1801.00001"
    (source / "mineru").mkdir(parents=True)
    (source / "paper.pdf").write_bytes(b"%PDF-second")
    (source / "metadata.json").write_text(
        json.dumps({
            "base_id": "1801.00001",
            "canonical_id": "1801.00001v2",
            "title": "Older Attention",
            "abstract": "Attention before 2018.",
            "categories": ["cs.AI"],
            "published_at": "2017-01-01T00:00:00Z",
        }),
        encoding="utf-8",
    )
    digest = hashlib.sha256((source / "paper.pdf").read_bytes()).hexdigest()
    (source / "mineru" / "full.md").write_text("Abstract\nolder evidence", encoding="utf-8")
    (source / "mineru" / "manifest.json").write_text(json.dumps({"canonical_id": "1801.00001v2", "source_sha256": digest, "model_version": settings.mineru_model_version, "language": settings.mineru_language}), encoding="utf-8")
    (source / "mineru" / "content_list.json").write_text(json.dumps([
        {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "older evidence", "page_idx": 1},
    ]), encoding="utf-8")


def test_summary_and_comparison_use_unique_filtered_papers(tmp_path: Path):
    settings = make_index_fixture(tmp_path, include_second=True)

    summary = retrieve(settings, "attention", task="summary", filters={"year_from": "2018"}, limit=4, mode="lexical")
    assert summary["status"] == "ok"
    assert {item["paper_id"] for item in summary["data"]["items"]} == {"1706.03762"}

    comparison = retrieve(settings, "attention", task="comparison", paper_ids=["1706.03762", "1706.03762v7", "1801.00001"], limit=4, mode="lexical")
    assert comparison["status"] == "ok"
    assert {item["paper_id"] for item in comparison["data"]["items"]} == {"1706.03762", "1801.00001"}


def test_context_budget_truncates_source_instead_of_dropping_it(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "attention", task="fact", mode="lexical", max_chars=40)
    assert result["status"] == "ok"
    assert result["data"]["context_text"]
    assert result["data"]["truncated"] is True


def test_filters_validate_year_and_unknown_keys(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    assert retrieve(settings, "attention", filters={"year_from": "20"}, mode="lexical")["status"] == "invalid_input"
    assert retrieve(settings, "attention", filters={"unknown": "x"}, mode="lexical")["status"] == "invalid_input"


def test_reason_respects_final_limit(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "attention", task="reason", mode="lexical", limit=1)
    assert result["status"] == "ok"
    assert result["data"]["count"] <= 1
