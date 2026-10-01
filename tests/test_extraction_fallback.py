from __future__ import annotations

from paper_rag.extraction.schema import QueryExtraction
from paper_rag.extraction.deepseek import DeepSeekExtractionClient
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content import router as content_router


def test_model_parser_uses_one_structured_extraction(monkeypatch, settings):
    def fake_extract():
        return QueryExtraction(
            paper_mentions=["ResNet"],
            content_objects=[],
            compare_objects=["BasicBlock", "Bottleneck"],
            reference_mentions=[],
            confidence=0.95,
        )

    monkeypatch.setattr(content_router.ModelQueryParser, "_extract", lambda self: fake_extract())
    decision = RouteDecision(
        route="content",
        query="ResNet 比较两个 block",
        intent="compare",
        decision_backend="jev",
    )
    result = content_router.build_content_decision(settings, decision, [])
    assert result.parse_status == "ok"
    assert result.parser_result["compare_objects"] == ["BasicBlock", "Bottleneck"]


def test_deepseek_extraction_client_makes_one_request_and_validates_json(monkeypatch):
    calls = []
    client = DeepSeekExtractionClient("https://example.invalid", "key", "deepseek-flash")

    def fake_completion(payload):
        calls.append(payload)
        return {
            "choices": [{
                "message": {
                    "content": (
                        '{"paper_mentions":["ResNet"],"paper_groups":[],'
                        '"author_mentions":[],"year_intervals":[],'
                        '"venue_mentions":[],"source_paper_mentions":[],'
                        '"object_paper_mentions":[],"content_objects":[],'
                        '"compare_objects":[],"reference_mentions":[],'
                        '"reference_side":null,"group_mode":"single",'
                        '"metadata_fields":[],"scope_required":true,"confidence":0.9}'
                    )
                }
            }]
        }

    monkeypatch.setattr(DeepSeekExtractionClient, "_chat_completion", lambda _self, payload: fake_completion(payload))
    result = client.extract_query(
        "ResNet 的结构是什么？",
        route="content",
        intent="lookup",
        complexity=2,
    )
    assert result.paper_mentions == ["ResNet"]
    assert len(calls) == 1
