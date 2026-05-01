from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ....config import Settings
from ...data.aliases import alias_match_to_dict, expand_query_with_aliases
from ...data.filters import compare_number, match_record_filter
from ...data.manifest_lookup import load_active_manifest_records, to_evidence_manifest_record
from ...evidence import to_evidence_anchor_result, to_evidence_reference_entry
from ...top_router import RouteDecision
from ..common.text import flatten_filter_value, normalized_text_key


def plan_reference(settings: Settings, route: RouteDecision, warnings: list[str]) -> dict[str, Any]:
    if route.parse_status == "parse_failed":
        return {
            **build_reference_evidence_base(route),
            "parse_status": "parse_failed",
            "parser_error": route.parser_error,
            "reference_items": [],
            "citing_papers": [],
            "anchor_results": [],
        }
    if route.parse_status == "unknown_direction":
        return {
            **build_reference_evidence_base(route),
            "parse_status": "unknown_direction",
            "reference_items": [],
            "citing_papers": [],
            "anchor_results": [],
        }
    if not route.anchors:
        warnings.append("reference route missing anchor")
        return build_reference_evidence(route, [], [], parse_status="missing_anchor")
    if route.direction == "cites":
        references, anchor_results = reference_cites_results(settings, route, warnings)
    elif route.direction == "cited_by":
        references, anchor_results = reference_cited_by_results(settings, route)
    else:
        warnings.append("reference route direction is unsupported")
        return build_reference_evidence(route, [], [], parse_status="unknown_direction")
    references = combine_reference_results(references, route.anchor_mode or "per", route.direction, len(route.anchors))
    if not references:
        warnings.append("reference route found no matching references")
    return build_reference_evidence(route, references, anchor_results, count=len(references))


def build_reference_evidence_base(route: RouteDecision) -> dict[str, Any]:
    return {
        "direction": route.direction,
        "anchors": route.anchors,
        "anchor_mode": route.anchor_mode,
        "filters": route.filters,
        "alias_matches": [alias_match_to_dict(match) for match in route.alias_matches],
    }


def build_reference_evidence(
    route: RouteDecision,
    references: list[dict[str, Any]],
    anchor_results: list[dict[str, Any]],
    *,
    parse_status: str = "ok",
    count: int | None = None,
) -> dict[str, Any]:
    evidence = {
        **build_reference_evidence_base(route),
        "parse_status": parse_status,
        "reference_items": [],
        "citing_papers": [],
        "anchor_results": [to_evidence_anchor_result(result) for result in anchor_results],
    }
    public_entries = [to_evidence_reference_entry(entry) for entry in references]
    if route.direction == "cited_by":
        evidence["citing_papers"] = public_entries
    else:
        evidence["reference_items"] = public_entries
    if route.intent == "count":
        evidence["count"] = count if count is not None else len(references)
    return evidence


def reference_cites_results(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    anchor_results: list[dict[str, Any]] = []
    for anchor in route.anchors:
        anchor_value = str(anchor or "").strip()
        resolved_papers = route.resolved_anchor_papers.get(anchor, [])
        anchor_refs: list[dict[str, Any]] = []
        for target in resolved_papers:
            for ref in load_reference_rows(target):
                if match_reference_filters(ref, route.filters, warnings):
                    entry = {
                        "direction": "cites",
                        "anchor_mention": anchor,
                        "anchor_paper": target,
                        "target_paper": None,
                        "reference": ref,
                    }
                    anchor_refs.append(entry)
                    references.append(entry)
        if not resolved_papers:
            warnings.append(f"reference anchor not found locally: {anchor_value}")
        anchor_results.append({
            "anchor_mention": anchor,
            "direction": "cites",
            "resolved_papers": resolved_papers,
            "references": anchor_refs,
            "count": len(anchor_refs),
        })
    return references, anchor_results


def reference_cited_by_results(settings: Settings, route: RouteDecision) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    anchor_results: list[dict[str, Any]] = []
    records = load_active_manifest_records(settings)
    for anchor in route.anchors:
        resolved_papers = route.resolved_anchor_papers.get(anchor, [])
        anchor_terms = [str(target.get("title") or "").strip() for target in resolved_papers if target.get("title")]
        target_keys = {paper_identity_key(target) for target in resolved_papers}
        anchor_refs: list[dict[str, Any]] = []
        seen_citing_papers: set[str] = set()
        for record in records:
            if not all(match_record_filter(settings, record, filter_item) for filter_item in route.filters):
                continue
            paper = to_evidence_manifest_record(record)
            paper["paper_id"] = Path(str(paper.get("paper_data_path") or "")).name if paper.get("paper_data_path") else None
            paper_key = paper_identity_key(paper)
            if paper_key in target_keys:
                continue
            if paper_key in seen_citing_papers:
                continue
            for ref in load_reference_rows(paper):
                if match_reference_terms(ref.get("raw_text"), anchor_terms):
                    entry = {
                        "direction": "cited_by",
                        "anchor_mention": anchor,
                        "citing_paper": paper,
                    }
                    anchor_refs.append(entry)
                    references.append(entry)
                    seen_citing_papers.add(paper_key)
                    break
        anchor_results.append({
            "anchor_mention": anchor,
            "direction": "cited_by",
            "resolved_papers": resolved_papers,
            "references": anchor_refs,
            "count": len(anchor_refs),
        })
    return references, anchor_results


def load_reference_rows(paper: dict[str, Any]) -> list[dict[str, Any]]:
    paper_data_path = paper.get("paper_data_path")
    if not paper_data_path:
        return []
    path = Path(str(paper_data_path)) / "references.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "reference_id": row.get("reference_id"),
                "ref_index": row.get("ref_index"),
                "raw_text": row.get("raw_text"),
                "page": row.get("page"),
                "source_block_id": row.get("source_block_id"),
            })
    return rows


def reference_anchor_terms(settings: Settings, anchor_value: str) -> list[str]:
    expanded_query, matches = expand_query_with_aliases(settings, anchor_value)
    terms = [anchor_value, expanded_query]
    for match in matches:
        terms.extend(match.expanded_terms)
    return unique_reference_terms(terms)


def unique_reference_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = normalized_text_key(text)
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def match_reference_terms(raw_text: Any, terms: list[str]) -> bool:
    raw_key = normalized_text_key(str(raw_text or ""))
    return any((term_key := normalized_text_key(term)) and term_key in raw_key for term in terms)


def paper_identity_key(paper: dict[str, Any]) -> str:
    return str(paper.get("paper_id") or paper.get("paper_data_path") or paper.get("title") or "")


def match_reference_filters(ref: dict[str, Any], filters: list[dict[str, Any]], warnings: list[str]) -> bool:
    for filter_item in filters:
        field = filter_item.get("field")
        if field not in {"title", "year"}:
            warning = f"reference cite filters only support title/year; ignored {field}"
            if warning not in warnings:
                warnings.append(warning)
            continue
        matched = match_reference_positive_filter(ref, filter_item)
        if filter_item.get("negated"):
            matched = not matched
        if not matched:
            return False
    return True


def match_reference_positive_filter(ref: dict[str, Any], filter_item: dict[str, Any]) -> bool:
    raw_text = str(ref.get("raw_text") or "")
    field = filter_item.get("field")
    op = filter_item.get("op")
    expected = filter_item.get("value")
    if field == "title":
        values = flatten_filter_value(expected)
        if op in {"=", "contains", "in"}:
            return any(match_reference_terms(raw_text, [value]) for value in values)
    if field == "year":
        years = reference_years(raw_text)
        if op == "interval":
            return any(compare_number(year, "interval", expected) for year in years)
        if op == "in":
            return isinstance(expected, list) and any(year in {int(item) for item in expected} for year in years)
        if op == "=":
            return any(year == int(expected) for year in years)
        if op == "contains":
            return str(expected) in raw_text
    return False


def reference_years(raw_text: str) -> list[int]:
    return [int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", raw_text)]


def combine_reference_results(
    references: list[dict[str, Any]],
    anchor_mode: str,
    direction: str | None,
    anchor_count: int,
) -> list[dict[str, Any]]:
    if anchor_mode == "per":
        return references
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in references:
        grouped.setdefault(reference_result_key(entry, direction), []).append(entry)
    if anchor_mode == "or":
        return [items[0] for _, items in sorted(grouped.items())]
    if anchor_mode == "and":
        return [items[0] for _, items in sorted(grouped.items()) if len({str(item.get("anchor_mention") or "") for item in items}) >= anchor_count]
    return references


def reference_result_key(entry: dict[str, Any], direction: str | None) -> str:
    if direction == "cited_by":
        paper = entry.get("citing_paper") or {}
        return str(paper.get("paper_id") or paper.get("title") or "")
    ref = entry.get("reference") or {}
    return normalized_text_key(str(ref.get("raw_text") or ""))
