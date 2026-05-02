from __future__ import annotations

from copy import deepcopy
from typing import Any

from ....config import Settings
from ...data.aliases import AliasMatch, expand_query_with_aliases
from ...data.aliases import resolve_paper_queries as resolve_manifest_paper_queries
from ....dataprocess.venues import normalize_venue_for_storage
from ...data.text import flatten_filter_value, unique_nonempty


def resolve_parser_papers(
    settings: Settings,
    parser_result: dict[str, Any],
    *,
    fallback_query: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_parser_paper_scope(settings, parser_result)
    resolved_papers = resolved["resolved_papers"]
    alias_matches = list(resolved["alias_matches"])
    if fallback_query and not resolved_papers:
        fallback_papers, fallback_matches = resolve_paper_queries(settings, [fallback_query])
        resolved_papers = merge_papers(resolved_papers, fallback_papers)
        alias_matches.extend(fallback_matches)
    return {
        **resolved,
        "resolved_papers": resolved_papers,
        "alias_matches": dedupe_alias_matches(alias_matches),
    }


def resolve_parser_paper_scope(settings: Settings, parser_result: dict[str, Any]) -> dict[str, Any]:
    filters, filter_matches, filter_papers = resolve_scope_filters(settings, parser_result.get("filters") or [])
    paper_groups: list[dict[str, Any]] = []
    group_matches: list[AliasMatch] = []
    group_papers: list[dict[str, Any]] = []
    for group in parser_result.get("paper_groups") or []:
        if not isinstance(group, dict):
            continue
        group_filters, matches, papers = resolve_scope_filters(settings, group.get("filters") or [])
        group_matches.extend(matches)
        group_papers = merge_papers(group_papers, papers)
        paper_groups.append({
            **group,
            "semantic": str(group.get("semantic") or "").strip(),
            "filters": group_filters,
        })

    resolved_papers = merge_papers(filter_papers, group_papers)
    return {
        "filters": filters,
        "paper_groups": paper_groups,
        "resolved_papers": resolved_papers,
        "alias_matches": dedupe_alias_matches([*filter_matches, *group_matches]),
    }


def resolve_scope_filters(
    settings: Settings,
    filters: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[AliasMatch], list[dict[str, Any]]]:
    resolved_filters: list[dict[str, Any]] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for filter_item in filters:
        item = deepcopy(filter_item)
        field = item.get("field")
        if field == "paper":
            value, matches, papers = resolve_paper_filter_value(settings, item.get("value"))
            item["value"] = value
            alias_matches.extend(matches)
            if not item.get("negated"):
                resolved_papers = merge_papers(resolved_papers, papers)
        elif field == "year" and item.get("op") == "interval":
            value, matches, papers = resolve_interval_paper_bounds(settings, item.get("value"))
            item["value"] = value
            alias_matches.extend(matches)
            resolved_papers = merge_papers(resolved_papers, papers)
        elif field == "title" and item.get("op") in {"=", "in"}:
            value, matches, papers = resolve_title_filter_value(settings, item.get("value"))
            item["value"] = value
            alias_matches.extend(matches)
            if not item.get("negated"):
                resolved_papers = merge_papers(resolved_papers, papers)
        elif field == "venue":
            item["value"] = resolve_venue_filter_value(settings, item.get("value"))
        resolved_filters.append(item)
    return resolved_filters, alias_matches, resolved_papers


def resolve_paper_filter_value(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    values = flatten_filter_value(value)
    titles, matches, papers = resolve_paper_mentions_to_titles(settings, values)
    if not isinstance(value, list):
        return (titles[0] if titles else str(value or "").strip()), matches, papers
    return titles, matches, papers


def resolve_interval_paper_bounds(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    if not isinstance(value, list) or len(value) != 2:
        return value, [], []
    left, left_matches, left_papers = resolve_interval_bound_paper(settings, value[0])
    right, right_matches, right_papers = resolve_interval_bound_paper(settings, value[1])
    return [left, right], [*left_matches, *right_matches], merge_papers(left_papers, right_papers)


def resolve_interval_bound_paper(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    if not isinstance(value, str) or not value.strip():
        return value, [], []
    normalized = value.strip().lower()
    if normalized in {"-inf", "-infinity", "inf", "+inf", "infinity", "+infinity"}:
        return value, [], []
    titles, matches, papers = resolve_paper_mentions_to_titles(settings, [value])
    return (titles[0] if titles else value), matches, papers


def resolve_title_filter_value(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    values = flatten_filter_value(value)
    titles, matches, papers = resolve_paper_mentions_to_titles(settings, values)
    if not isinstance(value, list):
        return (titles[0] if titles else str(value or "").strip()), matches, papers
    return titles, matches, papers


def resolve_venue_filter_value(settings: Settings, value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_venue_for_storage(settings, item) or str(item or "").strip() for item in value]
    return normalize_venue_for_storage(settings, value) or str(value or "").strip()


def resolve_paper_mentions_to_titles(
    settings: Settings,
    mentions: list[str],
) -> tuple[list[str], list[AliasMatch], list[dict[str, Any]]]:
    titles: list[str] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for mention in mentions:
        papers, matches = resolve_paper_queries(settings, [mention])
        alias_matches.extend(matches)
        resolved_papers = merge_papers(resolved_papers, papers)
        if papers:
            titles.extend(str(paper.get("title") or "").strip() for paper in papers if paper.get("title"))
        elif matches:
            titles.extend(match.canonical for match in matches if match.canonical)
        else:
            titles.append(mention)
    return unique_nonempty(titles), alias_matches, resolved_papers


def resolve_paper_queries(settings: Settings, queries: list[str]) -> tuple[list[dict[str, Any]], list[AliasMatch]]:
    return resolve_manifest_paper_queries(settings, queries)


def alias_matches_for_query(settings: Settings, query: str) -> list[AliasMatch]:
    _, matches = expand_query_with_aliases(settings, query)
    return matches


def dedupe_alias_matches(matches: list[AliasMatch]) -> list[AliasMatch]:
    seen: set[tuple[str, str]] = set()
    result: list[AliasMatch] = []
    for match in matches:
        key = (match.alias, match.canonical)
        if key not in seen:
            seen.add(key)
            result.append(match)
    return result


def merge_papers(*paper_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    for paper_list in paper_lists:
        for paper in paper_list:
            key = str(paper.get("paper_id") or paper.get("paper_data_path") or paper.get("title") or "")
            if key and key not in seen:
                seen.add(key)
                papers.append(paper)
    return papers
