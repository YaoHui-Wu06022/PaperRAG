"""top parser schema：只允许 router 字段。"""

from __future__ import annotations

from typing import Any

from paper_rag.retrieval.routes.common.errors import PlanParseError
from paper_rag.retrieval.routes.common.schema import load_payload


TOP_ROUTERS = {"content", "reference", "metadata", "unclear"}
PARSER_NAME = "Top"


def validate_top_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    """校验 top parser 输出并拒绝旧 filters/query 字段残留。"""
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    extra_fields = set(payload) - {"router"}
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise PlanParseError(f"Top parser 返回了不支持的字段：{fields}")
    router = payload.get("router")
    if router not in TOP_ROUTERS:
        raise PlanParseError(f"不支持的 top router：{router}")
    return {"router": router}
