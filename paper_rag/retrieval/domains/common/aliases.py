from __future__ import annotations

from typing import Any

from ....config import Settings
from ...data.aliases import AliasMatch, expand_query_with_aliases, resolve_target_papers
from ...top_router import route_tokens


def target_queries_from_anchors_and_title_filters(parser_result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for anchor in parser_result.get("anchors") or []:
        text = str(anchor or "").strip()
        if text:
            values.append(text)
    for filter_item in parser_result.get("filters") or []:
        if filter_item.get("field") == "title":
            values.extend(flatten_filter_value(filter_item.get("value")))
    return unique_nonempty(values)


def resolve_anchor_papers(settings: Settings, target_queries: list[str]) -> tuple[list[dict[str, Any]], list[AliasMatch]]:
    return resolve_target_papers(settings, target_queries)


def alias_matches_for_unresolved_anchors(settings: Settings, anchor_queries: list[str]) -> list[AliasMatch]:
    matches: list[AliasMatch] = []
    for query in anchor_queries:
        _, query_matches = expand_query_with_aliases(settings, query)
        matches.extend(query_matches)
    return matches


def dedupe_alias_matches(matches: list[AliasMatch]) -> list[AliasMatch]:
    seen: set[tuple[str, str]] = set()
    result: list[AliasMatch] = []
    for match in matches:
        key = (match.alias, match.canonical)
        if key in seen:
            continue
        seen.add(key)
        result.append(match)
    return result


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
