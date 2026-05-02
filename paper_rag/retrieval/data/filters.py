from __future__ import annotations

from typing import Any

from ...config import Settings
from ...dataprocess.manifest import normalize_year
from ...dataprocess.venues import expand_venue_query_terms, expand_venue_record_terms, venue_key, venue_keys_match
from .text import flatten_filter_value, normalized_text_key, route_tokens


def match_record_filters(settings: Settings, record, filters: list[dict[str, Any]]) -> bool:
    year_source = "preprint" if has_arxiv_filter(filters) else "publish"
    return all(match_record_filter(settings, record, filter_item, year_source=year_source) for filter_item in filters)


def match_record_filter(
    settings: Settings,
    record,
    filter_item: dict[str, Any],
    *,
    year_source: str = "publish",
) -> bool:
    matched = match_record_positive_filter(settings, record, filter_item, year_source=year_source)
    return not matched if filter_item.get("negated") else matched


def match_record_positive_filter(
    settings: Settings,
    record,
    filter_item: dict[str, Any],
    *,
    year_source: str = "publish",
) -> bool:
    field = filter_item.get("field")
    op = filter_item.get("op")
    value = filter_item.get("value")
    if field == "year":
        return compare_number(record.year, op, value, year_source=year_source)
    if field == "author":
        return compare_authors(record.author, op, value)
    if field == "venue":
        if matches_arxiv_preprint(record, value, op):
            return True
        return compare_venue(settings, record.venue, op, value)
    if field == "title":
        return compare_text(record.title, op, value)
    return False


def compare_number(actual: Any, op: str, expected: Any, *, year_source: str = "publish") -> bool:
    actual_year = filter_year(actual, year_source)
    if actual_year is None:
        return False
    actual_number = int(actual_year)
    if op == "interval":
        bounds = list(expected) if isinstance(expected, list) else []
        if len(bounds) != 2:
            return False
        lower_bound, upper_bound = bounds
        if not _is_negative_infinity(lower_bound):
            try:
                lower_number = int(lower_bound)
            except (TypeError, ValueError):
                return False
            if actual_number < lower_number:
                return False
        if not _is_positive_infinity(upper_bound):
            try:
                upper_number = int(upper_bound)
            except (TypeError, ValueError):
                return False
            if actual_number > upper_number:
                return False
        return True
    if op == "in":
        return isinstance(expected, list) and actual_number in {int(item) for item in expected}
    expected_number = int(expected)
    if op == "=":
        return actual_number == expected_number
    if op == "contains":
        return str(expected_number) in str(actual_number)
    return False


def _is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def _is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}


def compare_authors(authors: list[str], op: str, expected: Any) -> bool:
    values = flatten_filter_value(expected)
    if not values:
        return False
    if op in {"=", "contains", "in"}:
        return any(matching_author(authors, value) for value in values)
    return False


def compare_text(actual: Any, op: str, expected: Any) -> bool:
    actual_text = str(actual or "")
    actual_key = normalized_text_key(actual_text)
    values = flatten_filter_value(expected)
    if not values:
        return False
    value_keys = [normalized_text_key(value) for value in values]
    if op == "=":
        return actual_key == value_keys[0]
    if op == "in":
        return actual_key in set(value_keys)
    if op == "contains":
        return any(value_key and value_key in actual_key for value_key in value_keys)
    return False


def compare_venue(settings: Settings, actual: Any, op: str, expected: Any) -> bool:
    actual_keys = [venue_key(value) for value in expand_venue_record_terms(settings, actual)]
    actual_keys = [key for key in actual_keys if key]
    expected_keys = [venue_key(value) for value in expand_venue_query_terms(settings, flatten_filter_value(expected))]
    expected_keys = [key for key in expected_keys if key]
    if not actual_keys or not expected_keys:
        return False
    if op in {"=", "in"}:
        return any(venue_keys_match(actual_key, expected_key) for actual_key in actual_keys for expected_key in expected_keys)
    if op == "contains":
        return any(expected_key in actual_key for actual_key in actual_keys for expected_key in expected_keys)
    return False


def matches_arxiv_preprint(record, expected: Any, op: str) -> bool:
    if op not in {"=", "in", "contains"}:
        return False
    actual_year = normalize_year(record.year)
    if actual_year.get("preprint_year") is None:
        return False
    values = [str(value).strip().lower() for value in flatten_filter_value(expected) if str(value).strip()]
    return any(value in {"arxiv", "arxiv preprint"} for value in values)


def has_arxiv_filter(filters: list[dict[str, Any]]) -> bool:
    return any(
        filter_item.get("field") == "venue"
        and not filter_item.get("negated")
        and expected_is_arxiv(filter_item.get("value"))
        for filter_item in filters
    )


def expected_is_arxiv(value: Any) -> bool:
    values = [str(item).strip().lower() for item in flatten_filter_value(value) if str(item).strip()]
    return any(item in {"arxiv", "arxiv preprint"} for item in values)


def filter_year(value: Any, year_source: str) -> int | None:
    year = normalize_year(value)
    if year_source == "preprint":
        return year.get("preprint_year")
    return year.get("publish_year")


def matching_author(authors: list[str], author_query: str) -> str | None:
    query_tokens = route_tokens(author_query)
    if not query_tokens:
        return None
    for author in authors:
        if route_tokens(author) == query_tokens:
            return author
    return None
