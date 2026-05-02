from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import load_payload


TOP_ROUTERS = {"content", "reference", "metadata", "unclear"}
PARSER_NAME = "Top"


def validate_top_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    extra_fields = set(payload) - {"router"}
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise PlanParseError(f"Top parser returned unsupported fields: {fields}")
    router = payload.get("router")
    if router not in TOP_ROUTERS:
        raise PlanParseError(f"Invalid top router: {router}")
    return {"router": router}
