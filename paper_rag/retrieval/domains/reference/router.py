"""reference router：分别解析 source/object 两侧 scope。"""

from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ...data.aliases_match import dedupe_alias_matches
from ...data.manifest_records import merge_paper_records
from ...data.parser_scope_resolver import resolve_parser_paper_scope, resolve_year_filter_values
from ...route import RouteDecision
from .parser import ReferenceParserClient


def build_reference_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    """把 reference parser result 归一化成 RouteDecision。"""
    query = decision.query
    try:
        parser = plan_parser or ReferenceParserClient.from_settings(settings)
        if not hasattr(parser, "parse_reference"):
            raise PlanParseError("plan_parser must provide parse_reference(query)")
        parser_result = parser.parse_reference(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"reference_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            intent=None,
            query=query,
            resolved_papers=decision.resolved_papers,
            alias_matches=decision.alias_matches,
            parser_result=decision.parser_result,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_side=decision.return_side,
            source_semantic=decision.source_semantic,
            source_filters=decision.source_filters,
            source_groups=decision.source_groups,
            source_mode=decision.source_mode,
            object_semantic=decision.object_semantic,
            object_filters=decision.object_filters,
            object_groups=decision.object_groups,
            object_mode=decision.object_mode,
        )

    parser_result = correct_active_cites_scope(query, parser_result, warnings)

    source_resolved = resolve_parser_paper_scope(settings, {
        "filters": parser_result["source_filters"],
        "paper_groups": parser_result["source_groups"],
    })
    object_resolved = resolve_parser_paper_scope(settings, {
        "filters": parser_result["object_filters"],
        "paper_groups": parser_result["object_groups"],
    })
    parser_result = {
        **parser_result,
        # parser_result 中保留两侧 resolved_papers，debug 时能看 source/object 是否放反。
        "source_filters": source_resolved["filters"],
        "source_groups": source_resolved["paper_groups"],
        "object_filters": object_resolved["filters"],
        "object_groups": object_resolved["paper_groups"],
        "source_resolved_papers": source_resolved["resolved_papers"],
        "object_resolved_papers": object_resolved["resolved_papers"],
    }
    enriched = RouteDecision(
        route=decision.route,
        intent=parser_result["intent"],
        query=query,
        resolved_papers=merge_paper_records(
            decision.resolved_papers,
            source_resolved["resolved_papers"],
            object_resolved["resolved_papers"],
        ),
        alias_matches=dedupe_alias_matches([
            *decision.alias_matches,
            *source_resolved["alias_matches"],
            *object_resolved["alias_matches"],
        ]),
        parser_result=parser_result,
        parse_status="ok",
        return_side=parser_result["return_side"],
        source_semantic=parser_result["source_semantic"],
        source_filters=parser_result["source_filters"],
        source_groups=parser_result["source_groups"],
        source_mode=parser_result["source_mode"],
        object_semantic=parser_result["object_semantic"],
        object_filters=parser_result["object_filters"],
        object_groups=parser_result["object_groups"],
        object_mode=parser_result["object_mode"],
    )
    return apply_reference_year_filters(settings, enriched, warnings)


def correct_active_cites_scope(query: str, parser_result: dict[str, object], warnings: list[str]) -> dict[str, object]:
    """修正“X 引用的论文”这类主动句中 paper=X 被放到 object 侧的情况。"""
    if parser_result.get("return_side") != "object" or parser_result.get("intent") not in {"list", "count"}:
        return parser_result
    if not is_active_cites_query(query) or not source_scope_is_empty(parser_result):
        return parser_result

    object_filters = list(parser_result.get("object_filters") or [])
    moved_filters, remaining_filters = split_positive_paper_equals(object_filters)
    if not moved_filters:
        return parser_result

    warnings.append("reference parser corrected active cites paper scope from object to source")
    return {
        **parser_result,
        "source_filters": [*list(parser_result.get("source_filters") or []), *moved_filters],
        "object_filters": remaining_filters,
    }


def is_active_cites_query(query: str) -> bool:
    """判断表面句式是否像“X 引用/参考了哪些论文”。"""
    compact_query = "".join(str(query or "").split()).casefold()
    return any(pattern in compact_query for pattern in ("引用的", "引用了", "参考文献"))


def source_scope_is_empty(parser_result: dict[str, object]) -> bool:
    """source 侧完全为空时，才允许把 paper 约束从 object 侧挪回来。"""
    return not (
        str(parser_result.get("source_semantic") or "").strip()
        or parser_result.get("source_filters")
        or parser_result.get("source_groups")
    )


def split_positive_paper_equals(filters: list[object]) -> tuple[list[dict[str, object]], list[object]]:
    """拆出非 negated 的 paper=... filter，其它 object 条件保留在 object 侧。"""
    moved: list[dict[str, object]] = []
    remaining: list[object] = []
    for filter_item in filters:
        if isinstance(filter_item, dict) and is_positive_paper_equals(filter_item):
            moved.append(filter_item)
        else:
            remaining.append(filter_item)
    return moved, remaining


def is_positive_paper_equals(filter_item: dict[str, object]) -> bool:
    """只移动 paper=...，不碰 follow/prior、title/year/venue/author 条件。"""
    return (
        filter_item.get("field") == "paper"
        and filter_item.get("op") == "="
        and not bool(filter_item.get("negated", False))
    )


def apply_reference_year_filters(settings: Settings, decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    """解析 source/object 两侧 year interval 中的论文边界。"""
    source_filters = resolve_year_filter_values(settings, list(decision.source_filters), warnings)
    source_groups = [
        {**group, "filters": resolve_year_filter_values(settings, list(group.get("filters") or []), warnings)}
        for group in decision.source_groups
    ]
    object_filters = resolve_year_filter_values(settings, list(decision.object_filters), warnings)
    object_groups = [
        {**group, "filters": resolve_year_filter_values(settings, list(group.get("filters") or []), warnings)}
        for group in decision.object_groups
    ]
    if (
        source_filters == decision.source_filters
        and source_groups == decision.source_groups
        and object_filters == decision.object_filters
        and object_groups == decision.object_groups
    ):
        return decision

    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["source_filters"] = source_filters
        parser_result["source_groups"] = source_groups
        parser_result["object_filters"] = object_filters
        parser_result["object_groups"] = object_groups
    return RouteDecision(
        route=decision.route,
        intent=decision.intent,
        query=decision.query,
        resolved_papers=decision.resolved_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        return_side=decision.return_side,
        source_semantic=decision.source_semantic,
        source_filters=source_filters,
        source_groups=source_groups,
        source_mode=decision.source_mode,
        object_semantic=decision.object_semantic,
        object_filters=object_filters,
        object_groups=object_groups,
        object_mode=decision.object_mode,
    )
