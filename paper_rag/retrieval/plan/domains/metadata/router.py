from __future__ import annotations

from typing import Any

from .....config import Settings
from .....dataprocess.manifest import effective_year
from ....data.aliases import resolve_target_papers
from ...top_router import RouteDecision, route_tokens
from .parser import PlanParseError, PlanParserClient, validate_metadata_parse


METADATA_ENTRY_TERMS = {
    "作者",
    "标题",
    "会议",
    "期刊",
    "题目",
    "发布",
    "发表",
    "年份",
    "哪一年",
    "几年",
    "谁写",
    "谁提出",
    "哪些论文",
    "多少篇",
    "几篇",
}


def metadata_route(query: str, tokens: list[str]) -> RouteDecision | None:
    reason = metadata_entry_reason(query, tokens)
    if not reason:
        return None
    return RouteDecision(
        route="metadata",
        reason=reason,
        intent=None,
        target_query=query,
    )


def build_metadata_decision(
    settings: Settings,
    decision: RouteDecision,
    query: str,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    try:
        parser_result = parse_metadata_query(settings, query, plan_parser)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"metadata_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent="unknown",
            target_query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_field=None,
        )
    if parser_result["intent"] == "unknown":
        warnings.append("metadata parser returned intent=unknown")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent="unknown",
            target_query=query,
            parser_result=parser_result,
            parse_status="unknown",
            return_field=parser_result["return_field"],
            filters=parser_result["filters"],
            anchors=parser_result["anchors"],
        )
    target_queries = metadata_target_queries(parser_result)
    if parser_result["intent"] == "lookup" and not target_queries:
        target_queries = [query]
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        target_query=query,
        target_queries=target_queries,
        parser_result=parser_result,
        parse_status="ok",
        return_field=parser_result["return_field"],
        filters=parser_result["filters"],
        anchors=parser_result["anchors"],
    )
    resolved = resolve_decision_targets(settings, enriched, target_queries)
    return apply_anchor_year_filters(resolved, warnings)


def parse_metadata_query(settings: Settings, query: str, plan_parser=None) -> dict[str, Any]:
    parser = plan_parser or PlanParserClient.from_settings(settings)
    if not hasattr(parser, "parse_metadata"):
        raise PlanParseError("plan_parser must provide parse_metadata(query)")
    result = parser.parse_metadata(query)
    return validate_metadata_parse(result, query)


def metadata_target_queries(parser_result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for anchor in parser_result.get("anchors") or []:
        if isinstance(anchor, dict) and anchor.get("field") == "title":
            text = str(anchor.get("value") or "").strip()
            if text:
                values.append(text)
    for filter_item in parser_result.get("filters") or []:
        if filter_item.get("field") == "title":
            values.extend(flatten_filter_value(filter_item.get("value")))
    for entity in parser_result.get("entities") or []:
        if isinstance(entity, dict) and entity.get("type") == "title":
            text = str(entity.get("text") or "").strip()
            if text:
                values.append(text)
    return unique_nonempty(values)


def resolve_decision_targets(settings: Settings, decision: RouteDecision, target_queries: list[str]) -> RouteDecision:
    target_papers, alias_matches = resolve_target_papers(settings, target_queries)
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=decision.intent,
        target_query=decision.target_query,
        target_queries=target_queries,
        target_papers=target_papers,
        alias_matches=alias_matches,
        parser_result=decision.parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        return_field=decision.return_field,
        filters=decision.filters,
        anchors=decision.anchors,
    )


def apply_anchor_year_filters(decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    filters = list(decision.filters)
    anchor_years = resolved_anchor_years(decision.target_papers)
    resolved_filters = [resolve_anchor_interval_filter(filter_item, anchor_years, warnings) for filter_item in filters]
    resolved_filters = merge_year_interval_filters(resolved_filters)
    if resolved_filters == filters:
        return decision
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=decision.intent,
        target_query=decision.target_query,
        target_queries=decision.target_queries,
        target_papers=decision.target_papers,
        alias_matches=decision.alias_matches,
        parser_result={**(decision.parser_result or {}), "filters": resolved_filters},
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        return_field=decision.return_field,
        filters=resolved_filters,
        anchors=decision.anchors,
    )


def resolved_anchor_years(target_papers: list[dict[str, Any]]) -> list[int]:
    years = [effective_year(paper.get("year")) for paper in target_papers]
    return [year for year in years if year is not None]


def resolve_anchor_interval_filter(filter_item: dict[str, Any], anchor_years: list[int], warnings: list[str]) -> dict[str, Any]:
    if filter_item.get("field") != "year" or filter_item.get("op") != "interval":
        return filter_item
    value = filter_item.get("value")
    if not isinstance(value, list) or "anchor" not in value:
        return filter_item
    if not anchor_years:
        warnings.append("metadata anchor interval could not resolve anchor year")
        return filter_item
    if value == ["anchor", "anchor"]:
        if len(anchor_years) < 2:
            warnings.append("metadata anchor interval requires at least two anchor years")
            return filter_item
        low, high = min(anchor_years), max(anchor_years)
        resolved = [low + 1, high - 1]
    else:
        resolved = list(value)
        if value[0] == "anchor":
            resolved[0] = min(anchor_years) + 1
        if value[1] == "anchor":
            resolved[1] = max(anchor_years) - 1
    return {**filter_item, "value": resolved}


def merge_year_interval_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []
    for filter_item in filters:
        if (
            filter_item.get("field") == "year"
            and filter_item.get("op") == "interval"
            and not filter_item.get("negated")
            and isinstance(filter_item.get("value"), list)
            and len(filter_item["value"]) == 2
        ):
            merged = merge_interval_filter(merged, filter_item)
        else:
            output.append(filter_item)
    if merged is not None:
        output.append(merged)
    return output


def merge_interval_filter(current: dict[str, Any] | None, next_filter: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return dict(next_filter)
    current_lower, current_upper = current["value"]
    next_lower, next_upper = next_filter["value"]
    return {
        **current,
        "value": [
            max_lower_bound(current_lower, next_lower),
            min_upper_bound(current_upper, next_upper),
        ],
    }


def max_lower_bound(left: Any, right: Any) -> Any:
    if is_negative_infinity(left):
        return right
    if is_negative_infinity(right):
        return left
    return max(left, right)


def min_upper_bound(left: Any, right: Any) -> Any:
    if is_positive_infinity(left):
        return right
    if is_positive_infinity(right):
        return left
    return min(left, right)


def is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}


def metadata_entry_reason(query: str, tokens: list[str]) -> str:
    """Only decide whether to enter metadata parsing."""
    term = first_metadata_entry_term(query)
    if term:
        return f"匹配到关键词: {term}"
    return ""


def first_metadata_entry_term(query: str) -> str | None:
    for term in sorted(METADATA_ENTRY_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def flatten_filter_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = " ".join(route_tokens(value))
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result
