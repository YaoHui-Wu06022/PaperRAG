"""reference parser schema 校验，统一 source_scope --cites--> object_scope。"""

from __future__ import annotations

from typing import Any

from paper_rag.retrieval.routes.common.errors import PlanParseError
from paper_rag.retrieval.routes.common.schema import (
    load_payload,
    normalize_nullable_enum,
    validate_group_mode,
    validate_paper_filters,
    validate_paper_groups,
    validate_semantic,
)


REFERENCE_INTENTS = {"list", "count", "exists", None}
REFERENCE_RETURN_SIDES = {"source", "object", None}
REFERENCE_FIELDS = {
    "intent",
    "return_side",
    "source_semantic",
    "source_filters",
    "source_groups",
    "source_mode",
    "object_semantic",
    "object_filters",
    "object_groups",
    "object_mode",
}
PARSER_NAME = "Reference"


def validate_reference_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    """校验 reference 输出，并约束 return_side 与 intent 的组合。"""
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    extra_fields = set(payload) - REFERENCE_FIELDS
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise PlanParseError(f"Reference parser 返回了不支持的字段：{fields}")

    intent = normalize_nullable_enum(payload.get("intent"))
    if intent not in REFERENCE_INTENTS:
        raise PlanParseError(f"不支持的 reference intent：{intent}")

    return_side = normalize_nullable_enum(payload.get("return_side"))
    if return_side not in REFERENCE_RETURN_SIDES:
        raise PlanParseError(f"不支持的 reference return_side：{return_side}")
    if intent in {"list", "count"} and return_side not in {"source", "object"}:
        raise PlanParseError("Reference list/count 需要 return_side=source 或 object")
    if intent in {"exists", None} and return_side is not None:
        raise PlanParseError("Reference exists/null 要求 return_side=null")

    source_groups = validate_paper_groups(payload.get("source_groups", []), PARSER_NAME, "source_groups")
    object_groups = validate_paper_groups(payload.get("object_groups", []), PARSER_NAME, "object_groups")
    source_mode = validate_group_mode(payload.get("source_mode", "single"), source_groups, PARSER_NAME, "source_mode")
    object_mode = validate_group_mode(payload.get("object_mode", "single"), object_groups, PARSER_NAME, "object_mode")

    return {
        "intent": intent,
        "return_side": return_side,
        "source_semantic": validate_semantic(payload.get("source_semantic", ""), "Reference source_semantic"),
        "source_filters": validate_paper_filters(payload.get("source_filters", []), f"{PARSER_NAME} source"),
        "source_groups": source_groups,
        "source_mode": source_mode,
        "object_semantic": validate_semantic(payload.get("object_semantic", ""), "Reference object_semantic"),
        "object_filters": validate_paper_filters(payload.get("object_filters", []), f"{PARSER_NAME} object"),
        "object_groups": object_groups,
        "object_mode": object_mode,
    }
