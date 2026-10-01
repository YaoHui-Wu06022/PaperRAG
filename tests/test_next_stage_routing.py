from paper_rag.answer.service import run_ask
from paper_rag.extraction.schema import QueryExtraction
from paper_rag.retrieval.plan import build_plan_route
from paper_rag.retrieval.routes.common.jev_client import JevDecision, JevDecisionClient, JevError
from paper_rag.retrieval.routes.common.local_parser import LocalRouteParser
from paper_rag.retrieval.routes.content.planner import retrieval_summary


def test_local_parser_separates_resnet_scope_and_compare_objects():
    decision = JevDecision(route="content", content_intent="compare", needs_synthesis=True)
    parsed = LocalRouteParser(
        "在 ResNet 论文中，BasicBlock 和 Bottleneck 在结构与适用深度上有什么差异？",
        decision,
    ).parse_content("")

    assert {item["value"] for item in parsed["filters"] if item.get("field") == "paper"} == {"ResNet"}
    assert parsed["compare_objects"] == ["BasicBlock", "Bottleneck"]
    assert "ResNet" not in parsed["content_objects"]
    assert parsed["content_objects"] == []


def test_jev_failure_only_rules_fallback_for_metadata_reference(monkeypatch, settings):
    def fail(_settings):
        raise JevError("offline", code="jev_unavailable")

    monkeypatch.setattr("paper_rag.retrieval.plan.JevDecisionClient.from_settings", fail)
    metadata = build_plan_route(settings, "Transformer 发表在哪里？", [])
    content = build_plan_route(settings, "ResNet 的结构是什么？", [])

    assert metadata.route == "metadata"
    assert metadata.decision_backend == "rules"
    assert content.route == "unclear"
    assert content.decision_fallback_reason == "jev_unavailable"


def test_jev_probability_only_response_is_checked_against_confidence_threshold(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"route": "content", "probabilities": {"content": 0.42, "metadata": 0.31, "reference": 0.17, "unclear": 0.10}}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    client = JevDecisionClient("https://jev.invalid", "test-key", min_confidence=0.55)

    try:
        client.decide("ResNet 的结构是什么？")
    except JevError as exc:
        assert exc.code == "jev_low_confidence"
    else:
        raise AssertionError("probability-only low-confidence Jev response must be rejected")

    decision = JevDecision.from_payload({"route": "content", "probabilities": {"content": 0.82}})
    assert decision.confidence == 0.82


def test_retrieval_summary_requires_context_and_terms():
    stats = {
        "dense_available": True,
        "bm25_available": True,
        "dense_hits": 3,
        "bm25_hits": 2,
        "fused_hits": 2,
        "unique_papers": {"p1"},
        "required_terms_covered": False,
    }
    summary = retrieval_summary(stats, context_units=[{"chunk_id": "c1"}], scope_records=[{"title": "p1"}])
    assert summary["status"] == "insufficient"
    assert summary["required_terms_covered"] is False


def test_all_content_intents_return_evidence_without_answer_client(settings):
    for intent in ("lookup", "list", "count", "exists", "reason", "compare", "summary"):
        evidence = {
            "query": f"query-{intent}",
            "route": "content",
            "status": "ok",
            "intent": intent,
            "decision": {"complexity": 2},
            "retrieval": {"status": "sufficient"},
            "results": {"contexts": [{"chunk_id": f"{intent}-c1", "text": "evidence"}]},
        }
        run_ask(
            settings,
            evidence["query"],
            planner=lambda *_args, evidence=evidence, **_kwargs: evidence,
        )


def test_extraction_schema_contains_only_structured_query_fields():
    result = QueryExtraction.from_payload({
        "paper_mentions": ["ResNet"],
        "content_objects": ["shortcut"],
        "compare_objects": [],
        "reference_mentions": [],
        "confidence": 0.9,
    })
    assert result.paper_mentions == ["ResNet"]


def test_metadata_and_reference_never_call_answer_client(settings):
    metadata = {
        "query": "论文年份是什么？",
        "route": "metadata",
        "status": "ok",
        "intent": "lookup",
        "results": {"items": [{"title": "Paper", "values": {"year": 2020}}]},
    }
    reference = {
        "query": "谁引用了 Paper？",
        "route": "reference",
        "status": "ok",
        "intent": "list",
        "results": {"papers": ["Citing Paper"]},
    }
    run_ask(settings, metadata["query"], planner=lambda *_args, **_kwargs: metadata)
    run_ask(settings, reference["query"], planner=lambda *_args, **_kwargs: reference)


def test_content_ask_returns_evidence_and_never_calls_answer_client(settings):
    payload = {
        "query": "ResNet 的结构是什么？",
        "route": "content",
        "status": "ok",
        "intent": "lookup",
        "decision": {"complexity": 2},
        "retrieval": {"status": "sufficient"},
        "results": {"contexts": [{"chunk_id": "c1", "title": "ResNet", "text": "shortcut"}]},
    }
    planner = lambda *_args, **_kwargs: payload
    first = run_ask(settings, payload["query"], planner=planner)
    second = run_ask(settings, payload["query"], planner=planner)
    assert first["answer_mode"] == "evidence"
    assert second["answer_mode"] == "evidence"
