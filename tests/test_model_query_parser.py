from __future__ import annotations

from dataclasses import replace

from paper_rag.extraction.schema import QueryExtraction
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.common.model_parser import ModelQueryParser


def make_extraction(**overrides):
    payload = {
        "paper_mentions": ["ResNet"],
        "paper_groups": [],
        "author_mentions": [],
        "year_intervals": [],
        "venue_mentions": [],
        "source_paper_mentions": [],
        "object_paper_mentions": [],
        "content_objects": ["identity shortcut", "projection shortcut"],
        "compare_objects": ["identity shortcut", "projection shortcut"],
        "reference_mentions": [],
        "reference_side": None,
        "group_mode": "single",
        "metadata_fields": [],
        "scope_required": True,
        "confidence": 0.95,
    }
    payload.update(overrides)
    return QueryExtraction.from_payload(payload)


def test_model_parser_keeps_complex_objects_and_builds_content_scope(monkeypatch, settings):
    extraction = make_extraction()
    monkeypatch.setattr(
        "paper_rag.extraction.deepseek.extract_query_with_cache",
        lambda *_args, **_kwargs: extraction,
    )
    decision = RouteDecision(
        route="content",
        query="在 ResNet 论文中比较 identity shortcut 和 projection shortcut",
        intent="compare",
        decision_backend="jev",
        complexity=3,
    )
    parser = ModelQueryParser(settings, decision.query, decision)
    result = parser.parse_content(decision.query)
    assert result["compare_objects"] == ["identity shortcut", "projection shortcut"]
    assert result["filters"][0]["value"] == "ResNet"
    assert "Deep" not in result["content_objects"]
    assert result["extraction_debug"]["call_count"] == 1


def test_model_parser_builds_metadata_filters_from_structured_fields(monkeypatch, settings):
    extraction = make_extraction(
        content_objects=[],
        compare_objects=[],
        year_intervals=[[2017, "inf"]],
        venue_mentions=["CVPR", "ICCV"],
        metadata_fields=["title", "venue"],
        scope_required=False,
    )
    monkeypatch.setattr(
        "paper_rag.extraction.deepseek.extract_query_with_cache",
        lambda *_args, **_kwargs: extraction,
    )
    decision = RouteDecision(
        route="metadata",
        query="列出 2017 年以后 CVPR 或 ICCV 论文",
        intent="list",
        decision_backend="jev",
    )
    result = ModelQueryParser(settings, decision.query, decision).parse_metadata(decision.query)
    assert result["return_fields"] == ["title", "venue"]
    assert {item["field"] for item in result["filters"]} == {"paper", "year", "venue"}


def test_content_explicit_scope_without_paper_is_parse_failed(monkeypatch, settings):
    extraction = make_extraction(paper_mentions=[], content_objects=["shortcut"], compare_objects=[])
    monkeypatch.setattr(
        "paper_rag.extraction.deepseek.extract_query_with_cache",
        lambda *_args, **_kwargs: extraction,
    )
    decision = RouteDecision(
        route="content",
        query="在某篇论文中解释 shortcut",
        intent="lookup",
        decision_backend="jev",
    )
    parser = ModelQueryParser(settings, decision.query, decision)
    try:
        parser.parse_content(decision.query)
    except Exception as exc:
        assert "alias_unresolved" in str(exc)
    else:
        raise AssertionError("显式论文 scope 缺失时必须停止检索")


def test_low_confidence_extraction_stops_content(monkeypatch, settings):
    extraction = make_extraction(confidence=0.2)
    monkeypatch.setattr(
        "paper_rag.extraction.deepseek.extract_query_with_cache",
        lambda *_args, **_kwargs: extraction,
    )
    decision = RouteDecision(
        route="content",
        query="ResNet compare shortcut",
        intent="compare",
        decision_backend="jev",
    )
    parser = ModelQueryParser(settings, decision.query, decision)
    try:
        parser.parse_content(decision.query)
    except Exception as exc:
        assert "extraction_low_confidence" in str(exc)
    else:
        raise AssertionError("低置信度抽取时必须停止检索")


def test_extraction_cache_hit_avoids_second_model_call(monkeypatch, settings):
    calls = []
    extraction = make_extraction()

    def fake_extract(*_args, **_kwargs):
        calls.append(1)
        return extraction

    settings = replace(settings, deepseek_base_url="https://example.invalid", deepseek_api_key="key")
    monkeypatch.setattr("paper_rag.extraction.deepseek.DeepSeekExtractionClient.extract_query", fake_extract)
    first = ModelQueryParser(
        settings,
        "在 ResNet 论文中比较两个 shortcut",
        RouteDecision(route="content", query="在 ResNet 论文中比较两个 shortcut", intent="compare", complexity=2),
    )
    first._extract()
    second = ModelQueryParser(
        settings,
        "在 ResNet 论文中比较两个 shortcut",
        RouteDecision(route="content", query="在 ResNet 论文中比较两个 shortcut", intent="compare", complexity=2),
    )
    second_result = second._extract()
    assert len(calls) == 1
    assert second_result.cache_hit is True
