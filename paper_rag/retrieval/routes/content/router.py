"""content 路由：使用统一大模型抽取结果建立正文检索 scope。"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from paper_rag.config import Settings
from paper_rag.corpus.utils import dedupe_text
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.common.model_parser import ModelQueryParser
from paper_rag.retrieval.routes.common.router import apply_paper_scope_year_filters, build_paper_scope_decision

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def build_content_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
    corpus: "CorpusContext | None" = None,
) -> RouteDecision:
    """调用统一抽取器并把结果转换为正文 planner 的 RouteDecision。"""
    enriched = build_paper_scope_decision(
        settings,
        decision,
        warnings,
        parser_factory=lambda current_settings: ModelQueryParser(current_settings, decision.query, decision),
        parser_method="parse_content",
        warning_prefix="content",
        missing_parser_message="query parser 必须提供 parse_content(query)",
        plan_parser=plan_parser,
        corpus=corpus,
    )
    if enriched.parse_status == "parse_failed":
        return enriched
    enriched = normalize_unmarked_entity_scope(enriched)
    return apply_paper_scope_year_filters(settings, enriched, warnings, corpus=corpus)


EXPLICIT_PAPER_SCOPE_MARKERS = (
    "论文",
    "文中",
    "正文",
    "原文",
    "本文",
    "该文",
    "这篇",
    "paper",
    "article",
)


def normalize_unmarked_entity_scope(decision: RouteDecision) -> RouteDecision:
    """把未标记为论文范围的语义移动到正文对象，避免误当论文 scope。"""
    semantic = decision.paper_semantic.strip()
    if not semantic or not should_treat_semantic_as_content_object(decision):
        return decision
    parser_result = dict(decision.parser_result or {})
    content_objects = dedupe_text([semantic, *(parser_result.get("content_objects") or [])])
    parser_result["paper_semantic"] = ""
    parser_result["content_objects"] = content_objects
    parser_result["required_terms"] = dedupe_text([semantic, *(parser_result.get("required_terms") or [])])
    return replace(decision, paper_semantic="", parser_result=parser_result)


def should_treat_semantic_as_content_object(decision: RouteDecision) -> bool:
    if has_explicit_paper_scope_marker(decision.query):
        return False
    if has_paper_filter(decision.filters) or has_group_scope(decision.paper_groups):
        return False
    return decision.group_mode == "single"


def has_explicit_paper_scope_marker(query: str) -> bool:
    text = query.casefold()
    return any(marker in text for marker in EXPLICIT_PAPER_SCOPE_MARKERS)


def has_paper_filter(filters: list[dict[str, Any]]) -> bool:
    return any(filter_item.get("field") == "paper" for filter_item in filters)


def has_group_scope(groups: list[dict[str, Any]]) -> bool:
    return any((group.get("semantic") or group.get("filters")) for group in groups)
