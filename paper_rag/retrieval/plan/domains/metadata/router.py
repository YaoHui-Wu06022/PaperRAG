from __future__ import annotations

import re
from typing import Any

from .....config import Settings
from ....data.aliases import AliasMatch, resolve_target_papers
from ...top_router import RouteDecision, first_matching_term, route_tokens
from .parser import PlanParseError, PlanParserClient, validate_metadata_parse


METADATA_ENTRY_TERMS = {
    "author",
    "authors",
    "conference",
    "date",
    "journal",
    "publication",
    "published",
    "title",
    "venue",
    "year",
}
METADATA_ENTRY_PHRASES = [
    ("who wrote", ["who", "wrote"]),
    ("who are the authors", ["who", "are", "the", "authors"]),
    ("when was published", ["when", "was"], {"published"}),
    ("publication year", ["publication", "year"]),
    ("which journal", ["which", "journal"]),
    ("which conference", ["which", "conference"]),
    ("what is the title", ["what", "is", "the", "title"]),
]
METADATA_LIST_TERMS = {"paper", "papers"}
METADATA_COUNT_TERMS = {"count", "many", "number"}


def metadata_route(query: str, tokens: list[str]) -> RouteDecision | None:
    reason = metadata_entry_reason(tokens)
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
        )
    target_queries = metadata_target_queries(parser_result)
    if parser_result["intent"] == "lookup" and not target_queries:
        target_queries = [parser_result["raw_query"]]
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        target_query=parser_result["raw_query"],
        target_queries=target_queries,
        parser_result=parser_result,
        parse_status="ok",
        return_field=parser_result["return_field"],
        filters=parser_result["filters"],
    )
    return resolve_decision_targets(settings, enriched, target_queries)


def parse_metadata_query(settings: Settings, query: str, plan_parser=None) -> dict[str, Any]:
    parser = plan_parser or PlanParserClient.from_settings(settings)
    if not hasattr(parser, "parse_metadata"):
        raise PlanParseError("plan_parser must provide parse_metadata(query)")
    result = parser.parse_metadata(query)
    return validate_metadata_parse(result, query)


def metadata_target_queries(parser_result: dict[str, Any]) -> list[str]:
    values: list[str] = []
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
    )


def metadata_entry_reason(tokens: list[str]) -> str:
    """Only decide whether to enter metadata parsing."""
    for phrase in METADATA_ENTRY_PHRASES:
        label, sequence, *required_terms = phrase
        required = required_terms[0] if required_terms else set()
        if contains_sequence(tokens, sequence) and required.issubset(set(tokens)):
            return f"matched metadata entry phrase: {label}"
    if (set(tokens) & METADATA_LIST_TERMS) and metadata_has_filter_clue(tokens):
        return "matched metadata entry list/count clue"
    if (set(tokens) & METADATA_COUNT_TERMS) and metadata_has_filter_clue(tokens):
        return "matched metadata entry count clue"
    term = first_matching_term(tokens, METADATA_ENTRY_TERMS)
    if term:
        return f"matched metadata entry term: {term}"
    return ""


def metadata_has_filter_clue(tokens: list[str]) -> bool:
    if any(re.fullmatch(r"(?:19|20)\d{2}", token) for token in tokens):
        return True
    if first_matching_term(tokens, METADATA_ENTRY_TERMS):
        return True
    if contains_sequence(tokens, ["written", "by"]) or contains_sequence(tokens, ["authored", "by"]):
        return True
    return False


def contains_sequence(tokens: list[str], sequence: list[str]) -> bool:
    if len(sequence) > len(tokens):
        return False
    return any(tokens[index:index + len(sequence)] == sequence for index in range(len(tokens) - len(sequence) + 1))


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
