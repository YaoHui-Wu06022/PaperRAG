"""metadata/content 共享的 paper scope router 逻辑。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

from paper_rag.config import Settings
from paper_rag.corpus.aliases import dedupe_alias_matches
from paper_rag.corpus.records import merge_paper_records
from paper_rag.corpus.resolver import resolve_parser_scope, resolve_scope_year_filters
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.common.errors import PlanParseError

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


ParserFactory = Callable[[Settings], Any]


def build_paper_scope_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    parser_factory: ParserFactory,
    parser_method: str,
    warning_prefix: str,
    missing_parser_message: str,
    include_return_fields: bool = False,
    plan_parser: Any = None,
    corpus: "CorpusContext | None" = None,
) -> RouteDecision:
    """调用 domain parser，并把 metadata/content paper scope 归一化为 RouteDecision。"""
    query = decision.query
    try:
        parser = plan_parser or parser_factory(settings)
        parse = getattr(parser, parser_method, None)
        if parse is None:
            raise PlanParseError(missing_parser_message)
        parser_result = parse(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"{warning_prefix} parser 解析失败：{exc}")
        return paper_scope_parse_failed_decision(
            decision,
            str(exc),
            include_return_fields=include_return_fields,
        )

    parser_result = {
        **parser_result,
        # 顶层若将来带公共 filters，也在这里并入二层 scope。
        "filters": [*decision.filters, *parser_result["filters"]],
    }
    resolved = resolve_parser_scope(settings, parser_result, corpus=corpus)
    parser_result = {
        **parser_result,
        "filters": resolved["filters"],
        "paper_groups": resolved["paper_groups"],
    }
    payload: dict[str, Any] = {
        "route": decision.route,
        "intent": parser_result["intent"],
        "query": query,
        "resolved_papers": merge_paper_records(decision.resolved_papers, resolved["resolved_papers"]),
        "alias_matches": dedupe_alias_matches([*decision.alias_matches, *resolved["alias_matches"]]),
        "parser_result": parser_result,
        "parse_status": "ok",
        "paper_semantic": parser_result["paper_semantic"],
        "filters": parser_result["filters"],
        "paper_groups": parser_result["paper_groups"],
        "group_mode": parser_result["group_mode"],
    }
    if include_return_fields:
        payload["return_fields"] = parser_result["return_fields"]
    return RouteDecision(**payload)


def paper_scope_parse_failed_decision(
    decision: RouteDecision,
    parser_error: str,
    *,
    include_return_fields: bool = False,
) -> RouteDecision:
    """保留已有 scope 字段，把 metadata/content parser 失败包装成 RouteDecision。"""
    payload: dict[str, Any] = {
        "route": decision.route,
        "intent": None,
        "query": decision.query,
        "resolved_papers": decision.resolved_papers,
        "alias_matches": decision.alias_matches,
        "parser_result": decision.parser_result,
        "parse_status": "parse_failed",
        "parser_error": parser_error,
        "paper_semantic": decision.paper_semantic,
        "filters": decision.filters,
        "paper_groups": decision.paper_groups,
        "group_mode": decision.group_mode,
    }
    if include_return_fields:
        payload["return_fields"] = []
    return RouteDecision(**payload)


def apply_paper_scope_year_filters(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    include_return_fields: bool = False,
    corpus: "CorpusContext | None" = None,
) -> RouteDecision:
    """解析 metadata/content paper scope 中 year interval 的论文边界。"""
    filters, paper_groups = resolve_scope_year_filters(
        settings,
        decision.filters,
        decision.paper_groups,
        warnings,
        corpus=corpus,
    )
    if filters == decision.filters and paper_groups == decision.paper_groups:
        return decision

    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["filters"] = filters
        parser_result["paper_groups"] = paper_groups
    payload: dict[str, Any] = {
        "route": decision.route,
        "intent": decision.intent,
        "query": decision.query,
        "resolved_papers": decision.resolved_papers,
        "alias_matches": decision.alias_matches,
        "parser_result": parser_result,
        "parse_status": decision.parse_status,
        "parser_error": decision.parser_error,
        "paper_semantic": decision.paper_semantic,
        "filters": filters,
        "paper_groups": paper_groups,
        "group_mode": decision.group_mode,
    }
    if include_return_fields:
        payload["return_fields"] = decision.return_fields
    return RouteDecision(**payload)
