from __future__ import annotations

from typing import Any

from ....config import Settings
from ...data.aliases import AliasMatch, expand_query_with_aliases, resolve_paper_queries
from .text import flatten_filter_value, unique_nonempty


def resolve_parser_papers(
    settings: Settings,
    parser_result: dict[str, Any],
    *,
    fallback_query: str | None = None,
) -> dict[str, Any]:
    filters, filter_matches, filter_papers = norm_title_filters(settings, parser_result.get("filters") or [])
    anchors = [str(anchor).strip() for anchor in parser_result.get("anchors") or [] if str(anchor).strip()]
    resolved_anchor_papers: dict[str, list[dict[str, Any]]] = {}
    alias_matches: list[AliasMatch] = []
    for anchor in anchors:
        papers, matches = resolve_paper_queries(settings, [anchor])
        resolved_anchor_papers[anchor] = papers
        alias_matches.extend(matches)
        if not papers:
            alias_matches.extend(alias_matches_for_query(settings, anchor))

    resolved_papers = merge_papers(flatten_resolved_paper_groups(resolved_anchor_papers), filter_papers)
    if fallback_query and not resolved_papers:
        fallback_papers, fallback_matches = resolve_paper_queries(settings, [fallback_query])
        resolved_papers = merge_papers(resolved_papers, fallback_papers)
        alias_matches.extend(fallback_matches)
    alias_matches.extend(filter_matches)
    return {
        "filters": filters,
        "resolved_papers": resolved_papers,
        "resolved_anchor_papers": resolved_anchor_papers,
        "alias_matches": dedupe_alias_matches(alias_matches),
    }


def norm_title_filters(settings: Settings, filters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[AliasMatch], list[dict[str, Any]]]:
    normalized_filters: list[dict[str, Any]] = []
    exact_title_values: dict[bool, list[str]] = {}
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for filter_item in filters:
        if filter_item.get("field") == "title" and filter_item.get("op") in {"=", "in"}:
            values, matches, papers = resolve_title_filter_values(settings, flatten_filter_value(filter_item.get("value")))
            exact_title_values.setdefault(bool(filter_item.get("negated")), []).extend(values)
            alias_matches.extend(matches)
            if not filter_item.get("negated"):
                resolved_papers = merge_papers(resolved_papers, papers)
        else:
            normalized_filters.append(filter_item)
    for negated, values in exact_title_values.items():
        values = unique_nonempty(values)
        if values:
            normalized_filters.append({
                "field": "title",
                "op": "=" if len(values) == 1 else "in",
                "value": values[0] if len(values) == 1 else values,
                "negated": negated,
            })
    return normalized_filters, alias_matches, resolved_papers


def resolve_title_filter_values(settings: Settings, values: list[str]) -> tuple[list[str], list[AliasMatch], list[dict[str, Any]]]:
    titles: list[str] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for value in values:
        papers, matches = resolve_paper_queries(settings, [value])
        alias_matches.extend(matches)
        resolved_papers = merge_papers(resolved_papers, papers)
        if papers:
            titles.extend(str(paper.get("title") or "").strip() for paper in papers if paper.get("title"))
        elif matches:
            titles.extend(match.canonical for match in matches if match.canonical)
        else:
            titles.append(value)
    return unique_nonempty(titles), alias_matches, resolved_papers


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


def flatten_resolved_paper_groups(resolved_groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return merge_papers(*resolved_groups.values())


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
