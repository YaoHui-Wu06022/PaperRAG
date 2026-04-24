from __future__ import annotations

from typing import Any

from ....dataprocess.manifest import effective_year


def resolve_anchor_year_filters(
    filters: list[dict[str, Any]],
    anchor_papers: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    anchor_years = resolved_anchor_years(anchor_papers)
    resolved_filters = [resolve_anchor_interval_filter(filter_item, anchor_years, warnings) for filter_item in filters]
    return merge_year_interval_filters(resolved_filters)


def resolved_anchor_years(anchor_papers: list[dict[str, Any]]) -> list[int]:
    years = [effective_year(paper.get("year")) for paper in anchor_papers]
    return [year for year in years if year is not None]


def resolve_anchor_interval_filter(filter_item: dict[str, Any], anchor_years: list[int], warnings: list[str]) -> dict[str, Any]:
    if filter_item.get("field") != "year" or filter_item.get("op") != "interval":
        return filter_item
    value = filter_item.get("value")
    if not isinstance(value, list) or "anchor" not in value:
        return filter_item
    if not anchor_years:
        warnings.append("anchor interval could not resolve anchor year")
        return filter_item
    if value == ["anchor", "anchor"]:
        if len(anchor_years) < 2:
            warnings.append("anchor interval requires at least two anchor years")
            return filter_item
        low, high = min(anchor_years), max(anchor_years)
        resolved = [low + 1, high - 1]
    else:
        resolved = list(value)
        if value[0] == "anchor":
            resolved[0] = min(anchor_years) + 1
        if value[1] == "anchor":
            resolved[1] = max(anchor_years) - 1
    return {**filter_item, "value": resolved}


def merge_year_interval_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []
    for filter_item in filters:
        if (
            filter_item.get("field") == "year"
            and filter_item.get("op") == "interval"
            and not filter_item.get("negated")
            and isinstance(filter_item.get("value"), list)
            and len(filter_item["value"]) == 2
        ):
            merged = merge_interval_filter(merged, filter_item)
        else:
            output.append(filter_item)
    if merged is not None:
        output.append(merged)
    return output


def merge_interval_filter(current: dict[str, Any] | None, next_filter: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return dict(next_filter)
    current_lower, current_upper = current["value"]
    next_lower, next_upper = next_filter["value"]
    return {
        **current,
        "value": [
            max_lower_bound(current_lower, next_lower),
            min_upper_bound(current_upper, next_upper),
        ],
    }


def max_lower_bound(left: Any, right: Any) -> Any:
    if is_negative_infinity(left):
        return right
    if is_negative_infinity(right):
        return left
    return max(left, right)


def min_upper_bound(left: Any, right: Any) -> Any:
    if is_positive_infinity(left):
        return right
    if is_positive_infinity(right):
        return left
    return min(left, right)


def is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}
