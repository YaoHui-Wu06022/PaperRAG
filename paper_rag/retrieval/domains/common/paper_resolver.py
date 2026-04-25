from __future__ import annotations

from typing import Any

from ....config import Settings
from ...data.aliases import AliasMatch, expand_query_with_aliases, resolve_paper_mentions as resolve_paper_mentions_records
from .text import flatten_filter_value, unique_nonempty


def paper_mentions_from_anchors_and_title_filters(parser_result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for anchor in parser_result.get("anchors") or []:
        text = str(anchor or "").strip()
        if text:
            values.append(text)
    for filter_item in parser_result.get("filters") or []:
        if filter_item.get("field") == "title":
            values.extend(flatten_filter_value(filter_item.get("value")))
    return unique_nonempty(values)


def resolve_paper_mentions(settings: Settings, paper_mentions: list[str]) -> tuple[list[dict[str, Any]], list[AliasMatch]]:
    return resolve_paper_mentions_records(settings, paper_mentions)


def alias_matches_for_unresolved_anchors(settings: Settings, paper_mentions: list[str]) -> list[AliasMatch]:
    matches: list[AliasMatch] = []
    for query in paper_mentions:
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
