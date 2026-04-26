from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import load_payload, validate_paper_filters


TOP_ROUTERS = {"content", "reference", "metadata", "unclear"}
TOP_FILTER_FIELDS = {"author", "year", "venue"}
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
    filters = validate_paper_filters(payload.get("filters", []), PARSER_NAME)
    for filter_item in filters:
        if filter_item.get("field") not in TOP_FILTER_FIELDS:
            raise PlanParseError(f"Invalid top filter field: {filter_item.get('field')}")
    return {
        "router": router,
        "filters": filters,
        "extract_query": extract_query.strip() or fallback_query,
    }
