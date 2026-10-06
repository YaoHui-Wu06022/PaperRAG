from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

from paper_rag.catalog.service import rebuild_catalog
from paper_rag.catalog.chunks import CHUNK_RULE_VERSION
from paper_rag.config import Settings
from paper_rag.llamaindex.embedding import DashScopeEmbedding
from paper_rag.llamaindex.index import _cache_key_matches
from paper_rag.llamaindex.nodes import load_nodes
from paper_rag.llamaindex import service
from paper_rag.llamaindex.service import index_status, rebuild_index, retrieve
from paper_rag.llamaindex.retrievers import HybridRetriever


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
    assert nodes[0].metadata["retrieval_text_hash"] == hashlib.sha256(nodes[0].text.encode("utf-8")).hexdigest()
    assert nodes[0].metadata["chunk_rule_version"] == CHUNK_RULE_VERSION


def test_catalog_stores_retrieval_hash_and_embedding_cache_schema(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        chunk_columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        cache_columns = {row[1] for row in connection.execute("PRAGMA table_info(embedding_items)")}
        row = connection.execute("SELECT retrieval_text, retrieval_text_hash FROM chunks LIMIT 1").fetchone()
    assert "retrieval_text_hash" in chunk_columns
    assert {"retrieval_text_hash", "embedding_model", "embedding_dimensions", "chunk_rule_version", "milvus_collection"} <= cache_columns
    assert row[1] == hashlib.sha256(row[0].encode("utf-8")).hexdigest()


def test_catalog_sync_carries_forward_embedding_cache(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    connection = sqlite3.connect(settings.paper_catalog_db_path)
    try:
        chunk_id, retrieval_hash = connection.execute("SELECT chunk_id, retrieval_text_hash FROM chunks LIMIT 1").fetchone()
        connection.execute("INSERT INTO embedding_state (key, value) VALUES ('active_collection', 'old_collection')")
        connection.execute("INSERT INTO embedding_items (chunk_id, retrieval_text_hash, embedding_model, embedding_dimensions, chunk_rule_version, milvus_collection, synced_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (chunk_id, retrieval_hash, settings.embedding_model, settings.embedding_dimensions, CHUNK_RULE_VERSION, "old_collection", "now"))
        connection.commit()
    finally:
        connection.close()
    rebuild_catalog(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        assert connection.execute("SELECT value FROM embedding_state WHERE key='active_collection'").fetchone()[0] == "old_collection"
        assert connection.execute("SELECT COUNT(*) FROM embedding_items").fetchone()[0] == 1


def test_incremental_mode_requires_compatible_active_index(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = rebuild_index(settings, mode="incremental")
    assert result["status"] == "rebuild_required"
    assert "manifest_missing" in result["data"]["stale_reasons"]


def test_embedding_cache_key_changes_when_retrieval_text_changes(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    node = load_nodes(settings)[0]
    item = {"chunk_id": node.node_id, "retrieval_text_hash": node.metadata["retrieval_text_hash"], "embedding_model": settings.embedding_model, "embedding_dimensions": settings.embedding_dimensions, "chunk_rule_version": CHUNK_RULE_VERSION}
    assert _cache_key_matches(item, node, settings)
    node.metadata["retrieval_text_hash"] = "changed"
    assert not _cache_key_matches(item, node, settings)


def test_embedding_content_is_retrieval_text_without_metadata(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    node = load_nodes(settings)[0]

    assert node.get_content(metadata_mode=MetadataMode.EMBED) == node.text
    assert node.get_content(metadata_mode=MetadataMode.EMBED) == "abstract\nattention evidence"


def test_lexical_retrieval_returns_citations_without_milvus(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "什么是 Attention？", mode="lexical")
    assert result["status"] == "ok"
    item = result["data"]["evidence"][0]
    assert item["source_id"] == "S1"
    assert item["page_start"] == 1
    assert result["data"]["presentation"]["render_policy"] == "compose"


def test_hybrid_retrieval_marks_client_composition(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "什么是 Attention？", mode="hybrid")
    assert result["data"]["presentation"]["render_policy"] == "compose"


def test_retrieve_returns_only_evidence_and_scores(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "什么是 Attention？", task="fact", limit=2, max_chars=100, mode="lexical")
    assert result["status"] == "ok"
    assert result["data"]["evidence"][0]["source_id"] == "S1"
    assert result["data"]["evidence"][0]["text"] == "attention evidence"
    assert "score" in result["data"]["evidence"][0]
    assert "rrf_score" not in result["data"]["evidence"][0]
    assert "semantic_rank" not in result["data"]["evidence"][0]
    assert "papers" not in result["data"]
    assert "context_text" not in result["data"]
    assert "retrieval_debug" in result["data"]
    assert result["data"]["task"] == "fact"
    assert result["data"]["mode"] == "lexical"
    assert result["data"]["routing"]["task"] == "fact"
    assert "retrieval_text" not in result["data"]["evidence"][0]
    assert "ranking_features" not in result["data"]["evidence"][0]
    assert set(result["data"]["evidence"][0]) == {
        "source_id", "paper_id", "chunk_id", "text", "type", "section_path",
        "page_start", "page_end", "score",
    }
    assert "answer_context_id" not in result["data"]
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
    result = retrieve(settings, "什么是 Attention？", filters={"category": "cs.CL"}, task="fact", mode="lexical")
    assert result["status"] == "ok"
    assert result["data"]["evidence"][0]["paper_id"] == "1706.03762"


def test_retrieve_rejects_reference_region(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "什么是 Attention？", task="fact", mode="lexical", regions=["reference"])
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

    summary = retrieve(settings, "概括 Attention 的方法", task="summary", filters={"year_from": "2018"}, limit=4, mode="lexical")
    assert summary["status"] == "ok"
    assert {item["paper_id"] for item in summary["data"]["evidence"]} == {"1706.03762"}

    comparison = retrieve(settings, "比较 Attention 方法", task="comparison", paper_ids=["1706.03762", "1706.03762v7", "1801.00001"], limit=4, mode="lexical")
    assert comparison["status"] == "ok"
    assert {item["paper_id"] for item in comparison["data"]["evidence"]} == {"1706.03762", "1801.00001"}


def test_retrieve_evidence_keeps_chunk_text(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "什么是 Attention？", task="fact", mode="lexical", max_chars=40)
    assert result["status"] == "ok"
    assert result["data"]["evidence"][0]["text"] == "attention evidence"


def test_filters_validate_year_and_unknown_keys(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    assert retrieve(settings, "什么是 Attention？", filters={"year_from": "20"}, mode="lexical")["status"] == "invalid_input"
    assert retrieve(settings, "什么是 Attention？", filters={"unknown": "x"}, mode="lexical")["status"] == "invalid_input"


def test_reason_respects_final_limit(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "为什么 Attention 有效？", task="reason", mode="lexical", limit=1)
    assert result["status"] == "ok"
    assert len(result["data"]["evidence"]) <= 1


def test_candidate_without_evidence_returns_insufficient_evidence(tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    result = retrieve(settings, "未见过的 FlashAttention 细节", paper_ids=["1706.03762"], task="fact", mode="lexical")

    assert result["status"] == "insufficient_evidence"
    assert result["data"]["evidence"] == []
    assert "no_evidence_chunks" in result["warnings"]
    instruction = result["data"]["presentation"]["agent_instruction"]
    assert instruction["task"] == "fact"
    assert "无法可靠回答该问题" in instruction["system_prompt"]


def test_empty_primary_retrieval_uses_narrow_entity_fallback(monkeypatch, tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    calls: list[object] = []
    item = {
        "chunk_id": "fallback-chunk",
        "paper_id": "1706.03762",
        "canonical_id": "1706.03762v7",
        "ordinal": 1,
        "region": "content",
        "section_path": ["content"],
        "section_label": "content",
        "type": "text",
        "text": "FlashAttention evidence",
        "page_start": 1,
        "page_end": 1,
    }

    def fake_retrieve(*args, **kwargs):
        calls.append(kwargs.get("lexical_query_override"))
        if len(calls) == 1:
            return [], [], {"core_terms": ["FlashAttention"]}
        return [item], [], {"lexical_query": '"flashattention"'}

    monkeypatch.setattr(service, "_retrieve_items", fake_retrieve)
    result = retrieve(settings, "为什么 FlashAttention 能减少 HBM 访问？", paper_ids=["1706.03762"], task="fact", mode="lexical")

    assert result["status"] == "ok"
    assert len(result["data"]["evidence"]) == 1
    assert result["data"]["evidence"][0]["text"] == "FlashAttention evidence"
    assert calls[1].fts_query == '"flashattention"'
    assert len(calls) == 2


def test_retrieve_prepares_lexical_query_once(monkeypatch, tmp_path: Path):
    settings = make_index_fixture(tmp_path)
    original = service.prepare_lexical_query
    calls: list[str] = []

    def wrapped(query, current_settings):
        calls.append(query)
        return original(query, current_settings)

    monkeypatch.setattr(service, "prepare_lexical_query", wrapped)
    result = retrieve(settings, "什么是 Attention？", paper_ids=["1706.03762"], task="fact", mode="lexical")

    assert result["status"] == "ok"
    assert calls == ["什么是 Attention？"]


def test_candidate_discovery_merges_chunk_search_without_constraints(monkeypatch, tmp_path: Path):
    settings = Settings.load(tmp_path)
    record = SimpleNamespace(base_id="2106.09685", title="LoRA", abstract="LoRA trains adapters")
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(service, "search_catalog", lambda *_args, **_kwargs: [record])
    monkeypatch.setattr(
        service,
        "search_chunks",
        lambda *_args, **_kwargs: calls.append(_args) or [],
    )
    prepared = service.prepare_lexical_query("概括 LoRA", settings)

    candidates, debug = service._discover_candidates(settings, prepared, 5, "fact", {})

    assert candidates == ["2106.09685"]
    assert calls
    assert debug["candidate_discovery"]["chunk_search_used"] is True


def test_default_regions_skip_appendix_unless_query_mentions_it():
    assert service._normalize_regions(None, "概括这个方法") == ("abstract", "content")
    assert service._normalize_regions(None, "附录中有什么？") == ("abstract", "content", "appendix")
    assert service._normalize_regions(["appendix"], "概括这个方法") == ("appendix",)


def test_table_reference_sorting_prefers_requested_caption(tmp_path: Path):
    settings = Settings.load(tmp_path)
    retriever = HybridRetriever(settings, mode="lexical", task="fact")
    table_two = TextNode(
        text="Table 2: EfficientNet scaling results.",
        metadata={
            "content_text": "Table 2: EfficientNet scaling results.",
            "section_label": "Results",
            "type": "table",
            "region": "content",
            "paper_id": "1905.11946",
            "rrf_score": 0.01,
            "ranking_features": {"exact_table_ref_hit": True, "exact_table_caption_hit": True, "exact_entity_hit": True, "section_exact_hit": False, "duplicate_penalty": 0.0},
        },
    )
    table_five = TextNode(
        text="Table 5: EfficientNet ablation results.",
        metadata={
            "content_text": "Table 5: EfficientNet ablation results.",
            "section_label": "Results",
            "type": "table",
            "region": "content",
            "paper_id": "1905.11946",
            "rrf_score": 0.99,
            "ranking_features": {"exact_table_ref_hit": False, "exact_table_caption_hit": False, "exact_entity_hit": True, "section_exact_hit": False, "duplicate_penalty": 0.0},
        },
    )
    first = NodeWithScore(node=table_two, score=0.01)
    second = NodeWithScore(node=table_five, score=0.99)

    assert retriever._sort_key(first, "Table 2 比较了什么？") > retriever._sort_key(second, "Table 2 比较了什么？")


def test_candidate_discovery_falls_back_to_exact_method_entity(monkeypatch, tmp_path: Path):
    settings = Settings.load(tmp_path)

    def fake_search(_settings, _query, _filters, _limit, *, fts_query=None):
        if fts_query == '"lora"':
            return [SimpleNamespace(base_id="2106.09685")]
        return []

    monkeypatch.setattr(service, "search_catalog", fake_search)
    monkeypatch.setattr(service, "search_chunks", lambda *_args, **_kwargs: [])
    prepared = service.prepare_lexical_query("概括 LoRA 的核心贡献。", settings)
    candidates, debug = service._discover_candidates(settings, prepared, 5, "summary", {})

    assert candidates == ["2106.09685"]
    assert debug["candidate_discovery"]["entity_hits"] == {"LoRA": ["2106.09685"]}


def test_candidate_discovery_prefers_exact_title_over_abstract_background(monkeypatch, tmp_path: Path):
    settings = Settings.load(tmp_path)
    bert = SimpleNamespace(base_id="1810.04805", title="BERT: Pre-training", abstract="BERT uses two tasks")
    background = SimpleNamespace(base_id="1909.08053", title="Megatron-LM", abstract="BERT-like models")

    def fake_search(_settings, _query, _filters, _limit, *, fts_query=None):
        if fts_query == '"bert"':
            return [bert, background]
        return []

    monkeypatch.setattr(service, "search_catalog", fake_search)
    monkeypatch.setattr(service, "search_chunks", lambda *_args, **_kwargs: [])
    prepared = service.prepare_lexical_query("BERT 预训练使用哪两个任务？", settings)
    candidates, debug = service._discover_candidates(settings, prepared, 5, "fact", {})

    assert candidates == ["1810.04805"]
    assert debug["candidate_discovery"]["candidate_match_source"] == {"1810.04805": "title_exact"}


def test_reason_window_repairs_mid_token_context(monkeypatch, tmp_path: Path):
    chunks = [
        {"chunk_id": "c1", "paper_id": "p", "ordinal": 1, "region": "content", "section_label": "4 Method", "text": "The complete preceding sentence.", "page_start": 1, "page_end": 1},
        {"chunk_id": "c2", "paper_id": "p", "ordinal": 2, "region": "content", "section_label": "4 Method", "text": "ion kernel stores the cache.", "page_start": 1, "page_end": 1},
        {"chunk_id": "c3", "paper_id": "p", "ordinal": 3, "region": "content", "section_label": "4 Method", "text": "The physical blocks can be reused.", "page_start": 1, "page_end": 1},
    ]
    monkeypatch.setattr(service, "list_chunks", lambda *_args, **_kwargs: chunks)
    direct = [{**chunks[1], "type": "image", "score": 0.1, "rrf_score": 0.1}]

    result = service._reason_items(Settings.load(tmp_path), direct, 3, {})

    assert result
    assert result[0]["evidence_role"] == "context"
    assert result[0]["text"].startswith("The complete preceding sentence.")
    assert result[0]["continuity_status"] == "complete"
    assert result[0]["source_chunk_ids"] == ["c1", "c2", "c3"]


def test_reason_reuses_paper_chunk_cache(monkeypatch, tmp_path: Path):
    chunks = [
        {"chunk_id": "c1", "paper_id": "p", "ordinal": 1, "region": "content", "section_label": "Method", "text": "The first sentence.", "page_start": 1, "page_end": 1},
        {"chunk_id": "c2", "paper_id": "p", "ordinal": 2, "region": "content", "section_label": "Method", "text": "The second sentence.", "page_start": 1, "page_end": 1},
    ]
    calls = 0

    def load_chunks(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return chunks

    monkeypatch.setattr(service, "list_chunks", load_chunks)
    direct = [{**chunks[0], "type": "text"}, {**chunks[1], "type": "text"}]

    service._reason_items(Settings.load(tmp_path), direct, 4, {})

    assert calls == 1
