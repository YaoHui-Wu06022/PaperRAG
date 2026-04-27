from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import load_payload, validate_paper_filters


TOP_ROUTERS = {"content", "reference", "metadata", "unclear"}
TOP_FILTER_OPS_BY_FIELD = {
    "author": {"contains"},
    "year": {"=", "interval"},
    "venue": {"=", "in"},
    "title": {"contains"},
    "paper": {"=", "in"},
}
PARSER_NAME = "Top"


def validate_top_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    router = payload.get("router")
    if router not in TOP_ROUTERS:
        raise PlanParseError(f"Invalid top router: {router}")
    if "extract_query" not in payload:
        raise PlanParseError("Top parser missing extract_query")
    extract_query = payload.get("extract_query")
    if not isinstance(extract_query, str):
        raise PlanParseError("Top parser extract_query must be a string")
    filters = validate_top_filters(validate_paper_filters(payload.get("filters", []), PARSER_NAME))
    filter_groups = validate_filter_groups(payload.get("filter_groups", []))
    return {
        "router": router,
        "filters": filters,
        "filter_groups": filter_groups,
        "extract_query": extract_query.strip() or fallback_query,
    }


def validate_top_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for filter_item in filters:
        field = filter_item.get("field")
        allowed_ops = TOP_FILTER_OPS_BY_FIELD.get(str(field))
        if allowed_ops is None:
            raise PlanParseError(f"Invalid top filter field: {field}")
        op = filter_item.get("op")
        if op not in allowed_ops:
            raise PlanParseError(f"Invalid top filter op for {field}: {op}")
    return filters


def validate_filter_groups(value: Any) -> list[dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise PlanParseError("Top parser filter_groups must be a list")
    groups: list[dict[str, Any]] = []
    for group in value:
        if not isinstance(group, dict):
            raise PlanParseError("Top parser filter_group must be an object")
        subject = group.get("subject")
        if not isinstance(subject, str):
            raise PlanParseError("Top parser filter_group subject must be a string")
        filters = validate_top_filters(validate_paper_filters(group.get("filters", []), PARSER_NAME))
        groups.append({
            "subject": subject.strip(),
            "filters": filters,
        })
    return groups
