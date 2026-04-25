from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ..common.filters import resolve_paper_year_filters
from ..common.paper_resolver import resolve_parser_papers
from ...top_router import RouteDecision
from .parser import ContentParserClient


CONTENT_ACTION_TERMS = {
    "讲",
    "解释",
    "分析",
    "总结",
    "概括",
    "比较",
    "对比",
    "区别",
    "差异",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "使用",
    "采用",
    "用了",
    "说明",
    "提出",
    "解决",
    "讨论",
    "证明",
    "描述",
}

CONTENT_OBJECT_TERMS = {
    "方法",
    "流程",
    "步骤",
    "机制",
    "原理",
    "作用",
    "影响",
    "实验",
    "结果",
    "性能",
    "结构",
    "特点",
    "贡献",
    "局限",
    "优点",
    "缺点",
    "训练",
    "评估",
    "问题",
    "创新点",
    "改进",
    "优势",
    "不足",
    "限制",
    "算法",
    "模块",
    "框架",
}

CONTENT_ACTION_QUESTION_TERMS = {
    "写",
    "说",
}

CONTENT_QUESTION_TERMS = {
    "什么",
    "哪些",
    "怎样",
    "是否",
    "能否",
    "有没有",
}


def content_route(query: str, tokens: list[str] | None = None) -> RouteDecision | None:
    _ = tokens
    reason = content_entry_reason(query)
    if not reason:
        return None
    return RouteDecision(
        route="content",
        reason=reason,
        intent=None,
        query=query,
    )


def content_entry_reason(query: str) -> str:
    action = first_content_action(query)
    if action:
        return f"匹配到内容询问行为: {action}"
    action_question = first_content_action_question(query)
    question = first_content_question(query)
    if action_question and question == "什么":
        return f"匹配到内容行为和疑问词: {action_question}/{question}"
    object_term = first_content_object(query)
    if object_term and question:
        return f"匹配到内容对象和疑问词: {object_term}/{question}"
    return ""


def first_content_action(query: str) -> str | None:
    for term in sorted(CONTENT_ACTION_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def first_content_object(query: str) -> str | None:
    for term in sorted(CONTENT_OBJECT_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def first_content_action_question(query: str) -> str | None:
    for term in sorted(CONTENT_ACTION_QUESTION_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def first_content_question(query: str) -> str | None:
    for term in sorted(CONTENT_QUESTION_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def build_content_decision(
    settings: Settings,
    decision: RouteDecision,
    query: str,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    try:
        parser = plan_parser or ContentParserClient.from_settings(settings)
        if not hasattr(parser, "parse_content"):
            raise PlanParseError("plan_parser must provide parse_content(query)")
        parser_result = parser.parse_content(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"content_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent=None,
            query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
        )
    resolved = resolve_parser_papers(settings, parser_result)
    parser_result = {**parser_result, "filters": resolved["filters"]}
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        query=query,
        resolved_papers=resolved["resolved_papers"],
        resolved_anchor_papers=resolved["resolved_anchor_papers"],
        alias_matches=resolved["alias_matches"],
        parser_result=parser_result,
        parse_status="ok",
        filters=parser_result["filters"],
        anchors=parser_result["anchors"],
    )
    return apply_anchor_year_filters(settings, enriched, warnings)


def apply_anchor_year_filters(settings: Settings, decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    filters = list(decision.filters)
    resolved_filters = resolve_paper_year_filters(settings, filters, warnings)
    if resolved_filters == filters:
        return decision
    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["filters"] = resolved_filters
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=decision.intent,
        query=decision.query,
        resolved_papers=decision.resolved_papers,
        resolved_anchor_papers=decision.resolved_anchor_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        filters=resolved_filters,
        anchors=decision.anchors,
    )
