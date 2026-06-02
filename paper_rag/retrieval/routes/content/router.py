"""content router：调用 parser，并解析正文检索前的论文范围。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from paper_rag.config import Settings
from paper_rag.retrieval.routes.common.errors import PlanParseError
from paper_rag.corpus.aliases import dedupe_alias_matches
from paper_rag.corpus.records import merge_paper_records
from paper_rag.corpus.resolver import resolve_parser_scope, resolve_year_filter_values
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content.parser import ContentParserClient

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
    """把 content parser result 归一化成 RouteDecision。"""
    query = decision.query
    try:
        parser = plan_parser or ContentParserClient.from_settings(settings)
        if not hasattr(parser, "parse_content"):
            raise PlanParseError("plan_parser must provide parse_content(query)")
        parser_result = parser.parse_content(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"content_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            intent=None,
            query=query,
            resolved_papers=decision.resolved_papers,
            alias_matches=decision.alias_matches,
            parser_result=decision.parser_result,
            parse_status="parse_failed",
            parser_error=str(exc),
            paper_semantic=decision.paper_semantic,
            filters=decision.filters,
            paper_groups=decision.paper_groups,
            group_mode=decision.group_mode,
        )

    parser_result = {
        **parser_result,
        # 顶层若将来带公共 filters，也在这里并入 content scope。
        "filters": [*decision.filters, *parser_result["filters"]],
    }
    resolved = resolve_parser_scope(settings, parser_result, corpus=corpus)
    parser_result = {
        **parser_result,
        "filters": resolved["filters"],
        "paper_groups": resolved["paper_groups"],
    }
    enriched = RouteDecision(
        route=decision.route,
        intent=parser_result["intent"],
        query=query,
        resolved_papers=merge_paper_records(decision.resolved_papers, resolved["resolved_papers"]),
        alias_matches=dedupe_alias_matches([*decision.alias_matches, *resolved["alias_matches"]]),
        parser_result=parser_result,
        parse_status="ok",
        paper_semantic=parser_result["paper_semantic"],
        filters=parser_result["filters"],
        paper_groups=parser_result["paper_groups"],
        group_mode=parser_result["group_mode"],
    )
    return apply_content_year_filters(settings, enriched, warnings, corpus=corpus)


def apply_content_year_filters(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    corpus: "CorpusContext | None" = None,
) -> RouteDecision:
    """解析 content scope 中 year interval 的论文边界。"""
    filters = resolve_year_filter_values(settings, list(decision.filters), warnings, corpus=corpus)
    paper_groups = [
        {**group, "filters": resolve_year_filter_values(settings, list(group.get("filters") or []), warnings, corpus=corpus)}
        for group in decision.paper_groups
    ]
    if filters == decision.filters and paper_groups == decision.paper_groups:
        return decision

    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["filters"] = filters
        parser_result["paper_groups"] = paper_groups
    return RouteDecision(
        route=decision.route,
        intent=decision.intent,
        query=decision.query,
        resolved_papers=decision.resolved_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        paper_semantic=decision.paper_semantic,
        filters=filters,
        paper_groups=paper_groups,
        group_mode=decision.group_mode,
    )
