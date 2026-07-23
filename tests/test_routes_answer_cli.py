import json
from argparse import Namespace

import pytest

from paper_rag.answer.llm import AnswerError
from paper_rag.answer.service import run_ask
from paper_rag.cli.ask import handle_chat, print_evidence_sources
from paper_rag.cli.main import build_parser
from paper_rag.retrieval.dense.milvus_store import SearchResult
from paper_rag.retrieval.evidence import build_content_evidence
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content.planner import GROUP_CONTEXTS_PER_GROUP, merge_group_contexts, plan_body
from paper_rag.retrieval.routes.content.router import build_content_decision
from paper_rag.retrieval.routes.metadata.planner import plan_metadata
from paper_rag.retrieval.routes.reference.planner import plan_reference
from paper_rag.corpus.scope import records_for_scope

from conftest import add_paper, chunk_row, save_manifest, write_json, write_jsonl


def test_metadata_reference_and_content_routes_use_local_scopes(settings):
    resnet = add_paper(
        settings,
        file_hash="resnet-hash",
        paper_id="resnet",
        title="Deep Residual Learning for Image Recognition",
        authors=["Kaiming He"],
        year={"preprint_year": 2015, "publish_year": 2016},
        venue="CVPR",
        chunks=[
            chunk_row(
                "resnet",
                0,
                region="body",
                text="Residual connections ease optimization in very deep networks.",
            ),
            chunk_row(
                "resnet",
                1,
                region="appendix",
                text="Appendix-only residual ablation details.",
                section="Appendix",
            ),
        ],
    )
    vit = add_paper(
        settings,
        file_hash="vit-hash",
        paper_id="vit",
        title="An Image is Worth 16x16 Words",
        authors=["Alexey Dosovitskiy"],
        year={"preprint_year": 2020, "publish_year": 2021},
        venue="ICLR",
    )
    save_manifest(settings, [resnet, vit])
    write_json(
        settings.paper_data_dir / "citation_graph.json",
        {
            "nodes": [
                {"paper_id": "resnet", "title": resnet.title, "author": resnet.author, "year": resnet.year, "venue": resnet.venue},
                {"paper_id": "vit", "title": vit.title, "author": vit.author, "year": vit.year, "venue": vit.venue},
            ],
            "edges": [
                {
                    "source_paper_id": "vit",
                    "target_paper_id": "resnet",
                    "ref_index": 1,
                    "raw_text": "He et al. Deep Residual Learning for Image Recognition. 2016.",
                    "page": 8,
                    "source_block_id": "b000008",
                }
            ],
        },
    )

    metadata = plan_metadata(
        settings,
        RouteDecision(
            route="metadata",
            query="CVPR 有几篇论文？",
            intent="count",
            filters=[{"field": "venue", "op": "=", "value": "CVPR", "negated": False}],
            parse_status="ok",
        ),
        [],
    )
    assert metadata["route"] == "metadata"
    assert metadata["results"]["count"] == 1

    reference = plan_reference(
        settings,
        RouteDecision(
            route="reference",
            query="哪些论文引用了 ResNet？",
            intent="list",
            return_side="source",
            object_filters=[
                {
                    "field": "paper",
                    "op": "=",
                    "value": "Deep Residual Learning for Image Recognition",
                    "negated": False,
                }
            ],
            parse_status="ok",
        ),
        [],
    )
    assert reference["route"] == "reference"
    assert reference["results"]["papers"] == ["An Image is Worth 16x16 Words"]

    fake_store = FakeStore("resnet::chunk_0000")
    content = plan_body(
        settings,
        RouteDecision(
            route="content",
            query="ResNet 的 residual connections 是什么？",
            intent="lookup",
            parser_result={"content_objects": ["residual connections"], "compare_objects": []},
            filters=[
                {
                    "field": "paper",
                    "op": "=",
                    "value": "Deep Residual Learning for Image Recognition",
                    "negated": False,
                }
            ],
            parse_status="ok",
        ),
        [],
        embedder=FakeEmbedder(),
        store=fake_store,
    )
    assert content["route"] == "content"
    assert fake_store.paper_ids == ["resnet"]
    contexts = content["results"]["contexts"]
    assert contexts[0]["chunk_id"] == "resnet::chunk_0000"
    assert "Appendix-only" not in json.dumps(contexts, ensure_ascii=False)


def test_content_per_groups_retrieve_each_group(settings):
    resnet = add_paper(
        settings,
        file_hash="resnet-hash",
        paper_id="resnet",
        title="Deep Residual Learning for Image Recognition",
        chunks=[chunk_row("resnet", 0, region="body", text="residual shortcut architecture")],
    )
    transformer = add_paper(
        settings,
        file_hash="transformer-hash",
        paper_id="transformer",
        title="Attention is All You Need",
        chunks=[chunk_row("transformer", 0, region="body", text="multi-head attention architecture")],
    )
    save_manifest(settings, [resnet, transformer])

    content = plan_body(
        settings,
        RouteDecision(
            route="content",
            query="分别解释 ResNet 和 Transformer 的 architecture",
            intent="lookup",
            parser_result={"content_objects": ["architecture"], "compare_objects": []},
            paper_groups=[
                {
                    "semantic": "",
                    "filters": [
                        {
                            "field": "paper",
                            "op": "=",
                            "value": "Deep Residual Learning for Image Recognition",
                            "negated": False,
                        }
                    ],
                },
                {
                    "semantic": "",
                    "filters": [
                        {
                            "field": "paper",
                            "op": "=",
                            "value": "Attention is All You Need",
                            "negated": False,
                        }
                    ],
                },
            ],
            group_mode="per",
            parse_status="ok",
        ),
        [],
        embedder=FakeEmbedder(),
        store=FakeStoreByPaper({"resnet": "resnet::chunk_0000", "transformer": "transformer::chunk_0000"}),
    )

    groups = content["results"]["groups"]
    assert len(groups) == 2
    assert all(group["context_refs"] for group in groups)
    assert all("contexts" not in group for group in groups)
    assert [context["title"] for context in content["results"]["contexts"][:2]] == [
        "Deep Residual Learning for Image Recognition",
        "Attention is All You Need",
    ]


def test_group_context_merge_budget_scales_for_more_than_two_groups():
    group_results = [
        {"context_units": [{"chunk_id": f"g{group_index}::chunk_{chunk_index}"} for chunk_index in range(4)]}
        for group_index in range(3)
    ]

    limit = max(8, len(group_results) * GROUP_CONTEXTS_PER_GROUP)

    assert limit == 9
    assert len(merge_group_contexts(group_results, limit)) == 9


def test_content_router_reclassifies_unmarked_semantic_scope(settings):
    decision = build_content_decision(
        settings,
        RouteDecision(route="content", query="deepseek用到了什么模型", parse_status="ok"),
        [],
        plan_parser=StaticContentParser({
            "intent": "lookup",
            "paper_semantic": "deepseek",
            "filters": [],
            "paper_groups": [],
            "group_mode": "single",
            "content_objects": ["模型"],
            "compare_objects": [],
        }),
    )

    assert decision.paper_semantic == ""
    assert decision.parser_result["paper_semantic"] == ""
    assert decision.parser_result["content_objects"] == ["deepseek", "模型"]
    assert decision.parser_result["required_terms"] == ["deepseek"]


def test_content_router_keeps_semantic_scope_with_paper_marker(settings):
    decision = build_content_decision(
        settings,
        RouteDecision(route="content", query="deepseek论文用到了什么模型", parse_status="ok"),
        [],
        plan_parser=StaticContentParser({
            "intent": "lookup",
            "paper_semantic": "deepseek",
            "filters": [],
            "paper_groups": [],
            "group_mode": "single",
            "content_objects": ["模型"],
            "compare_objects": [],
        }),
    )

    assert decision.paper_semantic == "deepseek"
    assert decision.parser_result["content_objects"] == ["模型"]


def test_content_required_terms_filter_unrelated_reclassified_hits(settings):
    paper = add_paper(
        settings,
        file_hash="generic-model-hash",
        paper_id="generic-model",
        title="Generic Model Paper",
        chunks=[chunk_row("generic-model", 0, region="body", text="This paper describes a generic model architecture.")],
    )
    save_manifest(settings, [paper])

    content = plan_body(
        settings,
        RouteDecision(
            route="content",
            query="deepseek用到了什么模型",
            intent="lookup",
            parser_result={
                "content_objects": ["deepseek", "模型"],
                "compare_objects": [],
            },
            parse_status="ok",
        ),
        [],
        embedder=FakeEmbedder(),
        store=FakeStore("generic-model::chunk_0000"),
    )

    assert "contexts" not in content["results"]


def test_content_and_groups_report_missing_group_evidence(settings):
    resnet = add_paper(
        settings,
        file_hash="resnet-hash",
        paper_id="resnet",
        title="Deep Residual Learning for Image Recognition",
        chunks=[chunk_row("resnet", 0, region="body", text="self-attention evidence")],
    )
    vit = add_paper(
        settings,
        file_hash="vit-hash",
        paper_id="vit",
        title="An Image is Worth 16x16 Words",
        chunks=[chunk_row("vit", 0, region="body", text="patch embedding evidence")],
    )
    save_manifest(settings, [resnet, vit])

    warnings: list[str] = []
    content = plan_body(
        settings,
        RouteDecision(
            route="content",
            query="ResNet 和 ViT 是否都使用 self-attention？",
            intent="exists",
            parser_result={"content_objects": ["self-attention"], "compare_objects": []},
            paper_groups=[
                {
                    "semantic": "",
                    "filters": [
                        {
                            "field": "paper",
                            "op": "=",
                            "value": "Deep Residual Learning for Image Recognition",
                            "negated": False,
                        }
                    ],
                },
                {
                    "semantic": "",
                    "filters": [
                        {
                            "field": "paper",
                            "op": "=",
                            "value": "An Image is Worth 16x16 Words",
                            "negated": False,
                        }
                    ],
                },
            ],
            group_mode="and",
            parse_status="ok",
        ),
        warnings,
        embedder=FakeEmbedder(),
        store=FakeStoreByPaper({"resnet": "resnet::chunk_0000"}),
    )

    assert [group["exists"] for group in content["results"]["groups"]] == [True, False]
    assert any("缺失分组证据" in warning for warning in warnings)


def test_content_context_text_uses_block_window_without_exposing_expanded_blocks(settings):
    paper = add_paper(
        settings,
        file_hash="paper-hash",
        paper_id="paper",
        title="Window Paper",
        chunks=[chunk_row("paper", 1, region="body", text="hit block text", section="Introduction")],
    )
    write_jsonl(
        settings.paper_data_dir / "paper" / "blocks.jsonl",
        [
            {
                "block_id": "b000000",
                "order": 0,
                "region": "body",
                "type": "paragraph",
                "text": "before context",
                "page": 1,
                "bbox": None,
                "section_id": "sec_introduction",
                "section_path": ["Introduction"],
            },
            {
                "block_id": "b000001",
                "order": 1,
                "region": "body",
                "type": "paragraph",
                "text": "hit block text",
                "page": 1,
                "bbox": None,
                "section_id": "sec_introduction",
                "section_path": ["Introduction"],
            },
            {
                "block_id": "b000002",
                "order": 2,
                "region": "body",
                "type": "paragraph",
                "text": "after context",
                "page": 1,
                "bbox": None,
                "section_id": "sec_introduction",
                "section_path": ["Introduction"],
            },
        ],
    )
    save_manifest(settings, [paper])

    content = plan_body(
        settings,
        RouteDecision(
            route="content",
            query="Window Paper 的 hit block 是什么？",
            intent="lookup",
            parser_result={"content_objects": ["hit block"], "compare_objects": []},
            filters=[{"field": "paper", "op": "=", "value": "Window Paper", "negated": False}],
            parse_status="ok",
        ),
        [],
        embedder=FakeEmbedder(),
        store=FakeStoreByPaper({"paper": "paper::chunk_0001"}),
        debug=False,
    )

    context = content["results"]["contexts"][0]
    assert "before context" in context["text"]
    assert "after context" in context["text"]
    assert "expanded_blocks" not in context


def test_content_evidence_groups_and_debug_use_refs_instead_of_repeated_text():
    route = RouteDecision(route="content", query="q", intent="lookup", group_mode="per", parse_status="ok")
    unit = {
        "chunk_id": "paper::chunk_0001",
        "chunk_text": "long chunk text",
        "score": 0.5,
        "sources": {"bm25": {"rank": 1}},
        "expanded_blocks": [{"block_id": "b1", "type": "paragraph", "page": 1, "text": "long block text"}],
    }

    content = build_content_evidence(
        route,
        status="ok",
        warnings=[],
        scope_records=[{"title": "Window Paper"}],
        context_units=[unit],
        group_results=[{"records": [{"title": "Window Paper"}], "context_units": [unit], "exists": True}],
        debug=True,
    )

    group = content["results"]["groups"][0]
    assert group["context_refs"] == ["paper::chunk_0001"]
    assert "contexts" not in group
    assert "long block text" not in json.dumps(content["debug"], ensure_ascii=False)
    assert content["debug"]["context_units"][0] == {
        "chunk_id": "paper::chunk_0001",
        "score": 0.5,
        "sources": {"bm25": {"rank": 1}},
    }


def test_paper_semantic_strips_chinese_paper_suffix_and_uses_alias(settings):
    bert = add_paper(
        settings,
        file_hash="bert-hash",
        paper_id="bert",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin"],
        year={"preprint_year": 2018, "publish_year": 2019},
        venue="ACL",
    )
    save_manifest(settings, [bert])
    write_json(
        settings.data_dir / "paper_annotations.json",
        {
            "bert-hash": {
                "title": bert.title,
                "aliases": ["BERT"],
                "tags": {"zh": [], "en": []},
            }
        },
    )

    for semantic in ["BERT 论文", "BERT 这篇论文", "BERT 原论文"]:
        records = records_for_scope(settings, semantic, [])
        assert [record["title"] for record in records] == [bert.title]


def test_ask_uses_local_answers_and_falls_back_when_answer_llm_fails(settings):
    metadata_payload = {
        "query": "CVPR 有几篇论文？",
        "route": "metadata",
        "status": "ok",
        "intent": "count",
        "results": {"count": 1},
    }
    local = run_ask(settings, "CVPR 有几篇论文？", planner=lambda *_args, **_kwargs: metadata_payload)
    assert local["answer_mode"] == "local"
    assert "共找到 1 篇" in local["answer"]

    content_payload = {
        "query": "ResNet 的结构是什么？",
        "route": "content",
        "status": "ok",
        "results": {
            "contexts": [
                {
                    "chunk_id": "c1",
                    "title": "ResNet",
                    "section_path": ["Introduction"],
                    "pages": [1],
                    "text": "Residual connections ease optimization.",
                }
            ]
        },
    }
    fallback = run_ask(
        settings,
        "ResNet 的结构是什么？",
        planner=lambda *_args, **_kwargs: content_payload,
        answer_client=FailingAnswerClient(),
    )
    assert fallback["answer_mode"] == "local"
    assert "回答模型调用失败" in fallback["answer"]
    assert any("回答生成失败" in warning for warning in fallback["warnings"])


def test_cli_registers_main_and_probe_subcommands():
    parser = build_parser()

    probe_args = parser.parse_args(["probe", "planner", "--route", "content"])
    ask_args = parser.parse_args(["ask", "ResNet", "--evidence"])
    chat_args = parser.parse_args(["chat", "--mode", "plan"])

    assert probe_args.command == "probe"
    assert probe_args.probe_command == "planner"
    assert callable(probe_args.handler)
    assert ask_args.command == "ask"
    assert ask_args.query == ["ResNet"]
    assert ask_args.evidence is True
    assert chat_args.command == "chat"
    assert chat_args.mode == "plan"


def test_chat_reuses_corpus_skips_empty_input_and_continues_after_error(monkeypatch, capsys):
    corpus = object()
    calls = []
    queries = iter(["", "first", "second", "exit"])

    monkeypatch.setattr("builtins.input", lambda _prompt: next(queries))
    monkeypatch.setattr("paper_rag.cli.ask.Settings.load", lambda _root: object())
    monkeypatch.setattr("paper_rag.cli.ask.CorpusContext", lambda _settings: corpus)

    def fake_run_ask(_settings, query, *, debug, corpus):
        calls.append((query, debug, corpus))
        if query == "first":
            raise ValueError("temporary failure")
        return {"answer": "second answer", "evidence": {"route": "metadata", "results": {}}}

    monkeypatch.setattr("paper_rag.cli.ask.run_ask", fake_run_ask)

    args = Namespace(project_root=None, mode="ask", debug=False, evidence=False)
    assert handle_chat(args) == 0
    assert calls == [("first", False, corpus), ("second", False, corpus)]
    output = capsys.readouterr().out
    assert "本轮执行失败：temporary failure" in output
    assert "second answer" in output


def test_chat_plan_mode_outputs_json_and_rejects_evidence(monkeypatch, capsys):
    corpus = object()
    queries = iter(["query", "退出"])

    monkeypatch.setattr("builtins.input", lambda _prompt: next(queries))
    monkeypatch.setattr("paper_rag.cli.ask.Settings.load", lambda _root: object())
    monkeypatch.setattr("paper_rag.cli.ask.CorpusContext", lambda _settings: corpus)
    monkeypatch.setattr(
        "paper_rag.cli.ask.run_plan",
        lambda _settings, query, *, debug, corpus: {"query": query, "debug": debug, "same_corpus": corpus is not None},
    )

    args = Namespace(project_root=None, mode="plan", debug=True, evidence=False)
    assert handle_chat(args) == 0
    assert '"query": "query"' in capsys.readouterr().out

    parser = build_parser()
    invalid_args = parser.parse_args(["chat", "--mode", "plan", "--evidence"])
    with pytest.raises(SystemExit):
        invalid_args.handler(invalid_args)


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_chat_exits_cleanly_on_eof_or_ctrl_c(monkeypatch, error):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(error))
    monkeypatch.setattr("paper_rag.cli.ask.Settings.load", lambda _root: object())
    monkeypatch.setattr("paper_rag.cli.ask.CorpusContext", lambda _settings: object())

    args = Namespace(project_root=None, mode="ask", debug=False, evidence=False)
    assert handle_chat(args) == 0


def test_chat_exits_cleanly_when_query_is_interrupted(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "query")
    monkeypatch.setattr("paper_rag.cli.ask.Settings.load", lambda _root: object())
    monkeypatch.setattr("paper_rag.cli.ask.CorpusContext", lambda _settings: object())
    monkeypatch.setattr("paper_rag.cli.ask.run_ask", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    args = Namespace(project_root=None, mode="ask", debug=False, evidence=False)
    assert handle_chat(args) == 0


def test_evidence_sources_show_all_content_locations_without_snippets(capsys):
    contexts = [
        {"chunk_id": f"c{index}", "section_path": ["Intro"], "pages": [index], "text": f"context {index}"}
        for index in range(7)
    ]

    print_evidence_sources({"route": "content", "results": {"contexts": contexts}})

    output = capsys.readouterr().out
    assert "chunk: c6" in output
    assert "章节: Intro" in output
    assert "页码: 6" in output
    assert "context 6" not in output


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeStore:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.paper_ids: list[str] | None = None

    def search(self, query_vector: list[float], top_k: int, *, paper_ids: list[str] | None = None):
        self.paper_ids = paper_ids
        return [
            SearchResult(
                score=0.9,
                chunk_id=self.chunk_id,
                paper_id="resnet",
                chunk_index=0,
                title="Deep Residual Learning for Image Recognition",
                section_path_text="Introduction",
                pages_text="1",
                text="Residual connections ease optimization in very deep networks.",
            )
        ]


class FakeStoreByPaper:
    def __init__(self, chunk_by_paper: dict[str, str]) -> None:
        self.chunk_by_paper = chunk_by_paper
        self.paper_id_calls: list[list[str] | None] = []

    def search(self, query_vector: list[float], top_k: int, *, paper_ids: list[str] | None = None):
        self.paper_id_calls.append(paper_ids)
        results = []
        for paper_id in paper_ids or []:
            chunk_id = self.chunk_by_paper.get(paper_id)
            if not chunk_id:
                continue
            results.append(
                SearchResult(
                    score=0.9,
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    chunk_index=int(chunk_id.rsplit("_", 1)[-1]),
                    title=paper_title(paper_id),
                    section_path_text="Introduction",
                    pages_text="1",
                    text=f"{paper_id} dense result",
                )
            )
        return results[:top_k]


class StaticContentParser:
    def __init__(self, parser_result: dict) -> None:
        self.parser_result = parser_result

    def parse_content(self, query: str) -> dict:
        return self.parser_result


def paper_title(paper_id: str) -> str:
    return {
        "resnet": "Deep Residual Learning for Image Recognition",
        "transformer": "Attention is All You Need",
        "vit": "An Image is Worth 16x16 Words",
        "paper": "Window Paper",
    }.get(paper_id, paper_id)


class FailingAnswerClient:
    def complete_answer(self, evidence: dict) -> str:
        raise AnswerError("boom")
