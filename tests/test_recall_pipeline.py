from dataclasses import replace
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from paper_rag.config import Settings
from paper_rag.llamaindex import service, translation
from paper_rag.llamaindex.query_rewriter import QueryRewrite, QueryRewriterClient, QueryRewriterError, _BASE_PROMPT, _parse_rewrite
from paper_rag.llamaindex.retrievers import HybridRetriever
from test_llamaindex_service import make_index_fixture
from test_query_rewriter import FakeResponse


def chunk(key, ordinal, section="4 Method", region="content", paper="p", **extra):
    return {"chunk_id": key, "paper_id": paper, "ordinal": ordinal, "region": region,
            "section_path": [region, section], "section_label": section, "type": "text",
            "text": f"Evidence {key}.", "page_start": ordinal, "page_end": ordinal, **extra}


@pytest.mark.parametrize("payload", [
    {"core_terms": ["注意力"]}, {"entities": [], "phrases": ["注意力"]},
    {"entities": "BERT", "core_terms": []}, {"entities": [], "core_terms": [1]},
    {"entities": [], "core_terms": []}, {"entities": [" "], "core_terms": []},
    {"entities": [], "core_terms": ["注意力"], "filters": {}},
])
def test_rewrite_strict_contract(payload):
    with pytest.raises(QueryRewriterError):
        _parse_rewrite(payload)


def test_rewrite_entities_only_and_cross_array_duplicates():
    result = _parse_rewrite({"entities": ["LoRA", "lora"], "core_terms": ["LoRA"]})
    assert result == QueryRewrite(("LoRA",), ())


@pytest.mark.parametrize("task", ["fact", "reason", "summary", "comparison"])
def test_all_body_tasks_use_shared_prompt(tmp_path, task):
    calls = []
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test")
    def opener(request, timeout):
        import json
        calls.append(json.loads(request.data))
        return FakeResponse({"choices": [{"message": {"content": '{"entities": ["LoRA"], "core_terms": []}'}}]})
    QueryRewriterClient(settings, opener=opener).rewrite("LoRA 的方法是什么？", purpose="body", task=task)
    prompt = calls[0]["messages"][0]["content"]
    assert prompt == _BASE_PROMPT


def test_metadata_uses_shared_prompt_and_filter_context(tmp_path):
    import json
    calls = []
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test")
    def opener(request, timeout):
        calls.append(json.loads(request.data))
        return FakeResponse({"choices": [{"message": {"content": '{"entities": [], "core_terms": ["注意力"]}'}}]})
    filters = {"year_from": "2020", "category": "cs.CV"}
    QueryRewriterClient(settings, opener=opener).rewrite("2020 年以后有哪些计算机视觉论文和注意力相关？", filters=filters)
    assert calls[0]["messages"][0]["content"] == _BASE_PROMPT
    assert json.loads(calls[0]["messages"][1]["content"])["filters"] == filters


def test_rewrite_retains_original_and_translated_fields(tmp_path, monkeypatch):
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test")
    class Rewriter:
        def rewrite(self, query, **kwargs):
            return QueryRewrite(("LoRA", "QLoRA"), ("显存占用",))
    seen = []
    def translate(self, text):
        seen.append(text)
        return "memory usage"
    monkeypatch.setattr(translation.TencentTranslator, "translate", translate)
    prepared = translation.prepare_lexical_query("比较 LoRA 和 QLoRA 的显存占用", settings, purpose="body", task="comparison", rewriter=Rewriter())
    assert seen == ["显存占用"]
    assert prepared.debug()["entities"] == ["LoRA", "QLoRA"]
    assert prepared.debug()["core_terms"] == ["显存占用"]
    assert prepared.debug()["translated_core_terms"] == ["memory usage"]
    assert prepared.with_terms(prepared.core_terms).fts_query == '"memory usage"'


def test_pure_filters_skip_rewriter(tmp_path, monkeypatch):
    settings = make_index_fixture(tmp_path)
    monkeypatch.setattr(QueryRewriterClient, "rewrite", lambda *a, **k: pytest.fail("不应改写空查询"))
    result = service.search(settings, "", {"year_from": "2018"})
    assert result["data"]["count"] == 1


def test_lexical_empty_does_not_gate_semantic(tmp_path, monkeypatch):
    settings = make_index_fixture(tmp_path)
    from paper_rag.llamaindex.nodes import load_nodes
    node = load_nodes(settings)[0]
    calls = []
    monkeypatch.setattr(service, "_get_index_service", lambda s: SimpleNamespace(load=lambda: object()))
    def build(index, ids, regions, top_k):
        calls.append((ids, regions, top_k))
        return SimpleNamespace(retrieve=lambda bundle: [NodeWithScore(node=node, score=0.9)])
    monkeypatch.setattr(service, "_build_semantic_retriever", build)
    result = service.retrieve(settings, "未在词法出现的机制是什么？", task="fact")
    assert result["status"] == "ok"
    assert calls == [(None, ("abstract", "content"), 50)]
    stats = dict(result["data"]["retrieval_debug"]["recall"]["primary"])
    rankings = stats.pop("chunk_rankings")
    assert rankings[0]["chunk_id"] == node.node_id
    assert rankings[0]["lexical_rank"] is None
    assert rankings[0]["semantic_rank"] == 1
    assert rankings[0]["semantic_score"] == 0.9
    assert stats == {"lexical_count": 0, "semantic_count": 1, "fused_count": 1, "lexical_query": result["data"]["retrieval_debug"]["query_debug"]["lexical_query"], "entity_fallback_used": False}


def test_partial_missing_ids_do_not_silently_succeed(tmp_path):
    settings = make_index_fixture(tmp_path)
    result = service.retrieve(settings, "预训练任务有哪些？", paper_ids=["1706.03762", "9999.99999"], task="fact", mode="lexical")
    assert result["status"] == "not_found"
    assert "9999.99999" in result["warnings"][0]


def test_unique_title_resolution_rejects_ambiguous_background(tmp_path, monkeypatch):
    records = [SimpleNamespace(base_id="a", title="LoRA: Low-Rank Adaptation"), SimpleNamespace(base_id="b", title="Using LoRA Efficiently"), SimpleNamespace(base_id="c", title="QLoRA: Quantized Adaptation")]
    monkeypatch.setattr(service, "search_catalog", lambda *a, **k: records)
    prepared = replace(translation.prepare_lexical_query("比较 LoRA 和 QLoRA", Settings.load(tmp_path)), entities=("LoRA", "QLoRA"), translated_entities=("lora", "qlora"))
    targets, debug = service._resolve_targets(Settings.load(tmp_path), prepared, None)
    assert targets == ["c"]
    assert debug[0]["resolution"] == "ambiguous"
    assert debug[1]["resolution"] == "unique_title"


def test_reason_uses_later_seeds_and_keeps_regions(tmp_path, monkeypatch):
    rows = [chunk(str(i), i * 3, section=f"{i} Method") for i in range(8)]
    rows.append(chunk("appendix", 1, section="0 Method", region="appendix"))
    monkeypatch.setattr(service, "list_chunks", lambda *a, **k: rows)
    result = service._reason_items(Settings.load(tmp_path), rows[:8], 8, {}, ("content",))
    assert len(result) == 8
    assert result[5]["source_chunk_ids"] == ["5"]
    assert all("appendix" not in item["source_chunk_ids"] for item in result)


def test_reason_merge_updates_pages_and_resources(tmp_path, monkeypatch):
    rows = [chunk(str(i), i, asset_refs=[f"img{i}"], source_blocks=[i]) for i in range(5)]
    monkeypatch.setattr(service, "list_chunks", lambda *a, **k: rows)
    result = service._reason_items(Settings.load(tmp_path), [rows[1], rows[3]], 8, {}, ("content",))
    assert len(result) == 1
    assert result[0]["source_chunk_ids"] == ["0", "1", "2", "3", "4"]
    assert (result[0]["page_start"], result[0]["page_end"]) == (0, 4)
    assert result[0]["asset_refs"] == [f"img{i}" for i in range(5)]
    assert result[0]["source_blocks"] == list(range(5))


def test_summary_chapter_coverage_not_only_introduction():
    rows = [chunk("a", 0, region="abstract"), chunk("i", 1, section="1 Introduction"), chunk("m", 2), chunk("r", 3, section="5 Results"), chunk("c", 4, section="6 Conclusion"), chunk("ack", 5, section="Acknowledgements")]
    result = service._summary_sources(rows[:2], rows)
    assert [item["chunk_id"] for item in result[:4]] == ["a", "m", "r", "c"]
    assert all(item["chunk_id"] != "ack" for item in result)
    assert result[1]["chapter_supplement"]
    assert result[1]["score"] is None


def test_summary_unknown_chapters_and_comparison_quotas():
    rows = [chunk("a1", 1, section="1 Novel Component"), chunk("a2", 2, section="1 Novel Component"), chunk("b", 3, section="2 New Component")]
    assert [item["chunk_id"] for item in service._summary_sources([], rows)][:2] == ["a1", "b"]
    other = [chunk("x", 1, paper="other")]
    assert [item["paper_id"] for item in service._balanced_sources([rows, other], 4)] == ["p", "other", "p", "p"]


def test_comparison_recalls_per_paper_and_reuses_query_vector(tmp_path, monkeypatch):
    settings = make_index_fixture(tmp_path, include_second=True)
    from paper_rag.llamaindex.nodes import load_nodes
    nodes = load_nodes(settings)
    bundles, scopes = [], []
    embedded = []
    monkeypatch.setattr(service, "_get_index_service", lambda s: SimpleNamespace(load=lambda: object()))
    def build(index, ids, regions, top_k):
        scopes.append(ids)
        def recall(bundle):
            bundles.append(bundle)
            if bundle.embedding is None:
                embedded.append(bundle.query_str)
                bundle.embedding = [0.1]
            return [NodeWithScore(node=node, score=0.9) for node in nodes if node.metadata["paper_id"] in ids]
        return SimpleNamespace(retrieve=recall)
    monkeypatch.setattr(service, "_build_semantic_retriever", build)
    result = service.retrieve(settings, "比较两篇论文的训练方法", paper_ids=["1706.03762", "1801.00001"], task="comparison", limit=4)
    assert scopes == [["1706.03762"], ["1801.00001"]]
    assert embedded == ["比较两篇论文的训练方法"]
    assert bundles[0] is bundles[1]
    assert [item["source_id"] for item in result["data"]["evidence"]] == ["S1", "S2"]


def test_comparison_dimension_fallback_does_not_expand_scope(tmp_path, monkeypatch):
    from paper_rag.llamaindex import retrievers
    settings = Settings.load(tmp_path)
    prepared = translation.prepare_lexical_query("比较 LoRA 和 QLoRA 的显存占用", settings)
    calls = []
    def search(s, query, ids, limit, regions, **kwargs):
        calls.append((ids, kwargs["fts_query"]))
        return []
    monkeypatch.setattr(retrievers, "search_chunks", search)
    retriever = HybridRetriever(settings, mode="lexical", paper_ids=["p"], regions=["content"], lexical_query_override=prepared.with_terms(["memory usage"]), lexical_fallback_query=prepared.with_terms(["LoRA"]))
    retriever.retrieve("比较 LoRA 和 QLoRA 的显存占用")
    assert calls == [(["p"], '"memory usage"'), (["p"], '"lora"')]
    assert retriever.recall_debug["entity_fallback_used"]


def test_tables_and_formulas_are_not_postponed(tmp_path, monkeypatch):
    from paper_rag.llamaindex import retrievers
    monkeypatch.setattr(retrievers, "search_chunks", lambda *a, **k: [])
    nodes = [TextNode(id_=str(i), text="Evidence", metadata={"type": kind, "region": "content", "paper_id": "p"}) for i, kind in enumerate(["table", "equation", "text"])]
    semantic = SimpleNamespace(retrieve=lambda bundle: [NodeWithScore(node=node, score=1.0) for node in nodes])
    retriever = HybridRetriever(Settings.load(tmp_path), semantic)
    assert [item.node.metadata["type"] for item in retriever.retrieve("该方法的实验性能如何？")] == ["table", "equation", "text"]


def test_media_reference_keeps_single_digit_in_fts(tmp_path):
    from paper_rag.lexical import build_fts_query
    assert build_fts_query("", phrases=["Table 2"]) == '"table 2"'
    assert translation._normalize_parts(["Table 2", "Figure 3"])[0] == ["table 2", "figure 3"]


def test_reading_table_comparison_results_is_fact():
    from paper_rag.routing.rules import classify_by_rules
    from paper_rag.routing.schemas import RetrieveRequest, RetrieveTask
    decision = classify_by_rules(RetrieveRequest("EfficientNet 论文的表 2 展示了哪些比较结果"))
    assert decision.task is RetrieveTask.FACT


def test_embedding_failure_is_not_retried_per_paper(tmp_path, monkeypatch):
    settings = make_index_fixture(tmp_path, include_second=True)
    attempts = []
    monkeypatch.setattr(service, "_get_index_service", lambda s: SimpleNamespace(load=lambda: object()))
    def build(index, ids, regions, top_k):
        def fail(bundle):
            attempts.append(bundle.query_str)
            raise RuntimeError("Embedding unavailable")
        return SimpleNamespace(retrieve=fail)
    monkeypatch.setattr(service, "_build_semantic_retriever", build)
    result = service.retrieve(settings, "比较两篇论文的 attention 方法", paper_ids=["1706.03762", "1801.00001"], task="comparison")
    assert len(attempts) == 1
    assert result["data"]["evidence"]
    assert "lexical_fallback" in result["warnings"]


def test_automatic_comparison_chooses_two_then_recalls_separately(tmp_path, monkeypatch):
    settings = make_index_fixture(tmp_path, include_second=True)
    scopes = []
    def recall(s, query, ids, regions, mode, limit, **kwargs):
        scopes.append(ids)
        paper_ids = ids if ids is not None else ["1706.03762", "1801.00001"]
        return [chunk(p, 0, paper=p) for p in paper_ids], [], {"lexical_count": 0, "semantic_count": len(paper_ids), "fused_count": len(paper_ids)}
    monkeypatch.setattr(service, "_retrieve_items", recall)
    result = service.retrieve(settings, "比较相关论文的方法", task="comparison")
    assert scopes == [None, ["1706.03762"], ["1801.00001"]]
    assert result["data"]["retrieval_debug"]["target_paper_ids"] == ["1706.03762", "1801.00001"]


def test_comparison_limit_and_version_duplicates(tmp_path):
    settings = make_index_fixture(tmp_path, include_second=True)
    ids = ["1706.03762", "1801.00001"]
    assert service.retrieve(settings, "比较论文的方法", paper_ids=ids, task="comparison", mode="lexical", limit=1)["status"] == "invalid_input"
    assert service.retrieve(settings, "比较论文的方法", paper_ids=["1706.03762", "1706.03762v7"], task="comparison", mode="lexical")["status"] == "invalid_input"


def test_successful_rewrite_survives_translation_failure(tmp_path, monkeypatch):
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test", bm25_translation_retry_count=0)
    class Rewriter:
        def rewrite(self, query, **kwargs):
            return QueryRewrite(("LoRA",), ("显存占用",))
    def fail(self, text):
        raise RuntimeError("translation unavailable")
    monkeypatch.setattr(translation.TencentTranslator, "translate", fail)
    result = translation.prepare_lexical_query("LoRA 的显存占用如何？", settings, purpose="body", task="fact", rewriter=Rewriter())
    assert result.rewriter_used and not result.rewriter_fallback
    assert result.translation_fallback
    assert result.entities == ("LoRA",)
    assert result.debug()["core_terms"] == ["显存占用"]


def test_comparison_semantic_fallback_keeps_entity_fallback(tmp_path, monkeypatch):
    from paper_rag.llamaindex import retrievers
    settings = Settings.load(tmp_path)
    prepared = translation.prepare_lexical_query("比较两篇论文的显存占用", settings)
    calls = []
    monkeypatch.setattr(retrievers, "search_chunks", lambda *a, **k: calls.append(k["fts_query"]) or [])
    retriever = HybridRetriever(settings, mode="semantic", paper_ids=["p"], lexical_query_override=prepared.with_terms(["memory usage"]), lexical_fallback_query=prepared.with_terms(["LoRA"]))
    retriever.retrieve("比较两篇论文的显存占用")
    assert calls == ['"memory usage"', '"lora"']
    assert "lexical_fallback" in retriever.warnings


def test_same_table_number_prefers_named_paper_without_filtering(tmp_path, monkeypatch):
    from paper_rag.llamaindex import retrievers
    monkeypatch.setattr(retrievers, "search_chunks", lambda *a, **k: [])
    nodes = [TextNode(id_=p, text="Table 2", metadata={"paper_id": p, "region": "content", "type": "table", "content_text": "Table 2"}) for p in ["other", "target"]]
    semantic = SimpleNamespace(retrieve=lambda bundle: [NodeWithScore(node=node, score=1.0) for node in nodes])
    retriever = HybridRetriever(Settings.load(tmp_path), semantic, preferred_paper_ids=["target"])
    result = retriever.retrieve("目标论文的表 2 展示了什么？")
    assert [item.node.metadata["paper_id"] for item in result] == ["target", "other"]
    assert len(result) == 2


def test_entity_generic_paper_suffix_keeps_raw_debug(tmp_path, monkeypatch):
    records = [SimpleNamespace(base_id='e', title='EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks')]
    monkeypatch.setattr(service, 'search_catalog', lambda *a, **k: records)
    prepared = replace(translation.prepare_lexical_query('EfficientNet 论文的表 2 展示了什么？', Settings.load(tmp_path)), entities=('EfficientNet 论文',), translated_entities=('EfficientNet paper',))
    targets, debug = service._resolve_targets(Settings.load(tmp_path), prepared, None)
    assert targets == ['e']
    assert debug[0]['entity'] == 'EfficientNet 论文'


def test_summary_unknown_chapter_does_not_starve_ranked_evidence():
    rows = [chunk('a', 0, region='abstract'), chunk('m', 1), chunk('r', 2, section='5 Results'), chunk('c', 3, section='6 Conclusion'), chunk('i', 4, section='1 Introduction')]
    unknown = [chunk(f'u{n}', 5+n, section='3 New Component') for n in range(4)]
    ranked = rows + [chunk('high', 9, section='5 Results')]
    result = service._summary_sources(ranked, rows + unknown)
    assert [item['chunk_id'] for item in result[5:]] == ['u0', 'high']


def test_comparison_prepares_rewrite_translation_once(tmp_path, monkeypatch):
    settings = make_index_fixture(tmp_path, include_second=True)
    calls = []
    real_prepare = service.prepare_lexical_query
    def prepare(*args, **kwargs):
        calls.append((kwargs['purpose'], kwargs['task']))
        return real_prepare(*args, **kwargs)
    monkeypatch.setattr(service, 'prepare_lexical_query', prepare)
    monkeypatch.setattr(service, '_retrieve_items', lambda s, q, ids, *a, **k: ([chunk(p, 0, paper=p) for p in ids], [], {}))
    result = service.retrieve(settings, '比较两篇论文的训练方法', paper_ids=['1706.03762', '1801.00001'], task='comparison')
    assert {item['paper_id'] for item in result['data']['evidence']} == {'1706.03762', '1801.00001'}
    assert calls == [('body', 'comparison')]


def test_metadata_missing_catalog_has_explicit_status(tmp_path):
    result = service.search(Settings.load(tmp_path), '', {})
    assert result['status'] == 'catalog_not_ready'


def test_retrieval_rankings_are_debug_only_before_reason_windows(tmp_path):
    settings = make_index_fixture(tmp_path)
    result = service.retrieve(settings, "为什么 Attention 有效？", task="reason", mode="lexical")
    assert result["data"]["evidence"]
    rankings = result["data"]["retrieval_debug"]["recall"]["primary"]["chunk_rankings"]
    assert rankings[0]["lexical_rank"] == 1
    assert rankings[0]["semantic_rank"] is None
    assert rankings[0]["chunk_id"] in result["data"]["evidence"][0]["source_chunk_ids"]
    assert all("lexical_rank" not in e and "semantic_rank" not in e and "semantic_score" not in e and "rrf_score" not in e for e in result["data"]["evidence"])
