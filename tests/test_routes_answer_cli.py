import json
from argparse import Namespace

import pytest

from paper_rag.answer.llm import AnswerError
from paper_rag.answer.service import run_ask
from paper_rag.cli.ask import handle_chat, print_evidence_sources
from paper_rag.cli.main import build_parser
from paper_rag.retrieval.dense.milvus_store import SearchResult
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content.planner import plan_body
from paper_rag.retrieval.routes.metadata.planner import plan_metadata
from paper_rag.retrieval.routes.reference.planner import plan_reference

from conftest import add_paper, chunk_row, save_manifest, write_json


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
            "version": 1,
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


def test_evidence_sources_show_at_most_five_items(capsys):
    contexts = [{"chunk_id": f"c{index}", "title": "ResNet"} for index in range(7)]

    print_evidence_sources({"route": "content", "results": {"contexts": contexts}})

    output = capsys.readouterr().out
    assert "chunk: c4" in output
    assert "chunk: c5" not in output


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


class FailingAnswerClient:
    def complete_answer(self, evidence: dict) -> str:
        raise AnswerError("boom")
