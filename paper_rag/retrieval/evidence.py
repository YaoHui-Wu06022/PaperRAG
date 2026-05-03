from __future__ import annotations

from typing import Any

from ..dataprocess.venues import display_venue
from .data.aliases import alias_match_to_dict
from .route import RouteDecision


COMPACT_EDGE_LIMIT = 3


def build_metadata_evidence(
    settings,
    route: RouteDecision,
    *,
    status: str,
    warnings: list[str],
    records: list[dict[str, Any]] | None = None,
    group_results: list[dict[str, Any]] | None = None,
    count: int | None = None,
    exists: bool | None = None,
    parser_error: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    records = records or []
    results = metadata_results(
        settings,
        route,
        records,
        group_results=group_results,
        count=count,
        exists=exists,
    )
    evidence = build_base_evidence(
        route,
        status=status,
        warnings=warnings,
        plan={
            "return_fields": route.return_fields,
            "scope": compact_scope_terms(route.paper_semantic, route.filters),
            "groups": compact_groups(route.paper_groups),
            "group_mode": route.group_mode if route.group_mode != "single" else "",
        },
        resolved=common_resolved(route),
        results=results,
        parser_error=parser_error,
    )
    if debug:
        evidence["debug"] = route_debug(route, records=records, group_results=group_results)
    return evidence


def build_content_evidence(
    route: RouteDecision,
    *,
    status: str,
    warnings: list[str],
    scope_records: list[dict[str, Any]] | None = None,
    context_units: list[dict[str, Any]] | None = None,
    retrieval_query: dict[str, Any] | None = None,
    group_results: list[dict[str, Any]] | None = None,
    parser_error: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    parser_result = route.parser_result or {}
    scope_records = scope_records or []
    context_units = context_units or []
    results = content_results(context_units, group_results=group_results)
    evidence = build_base_evidence(
        route,
        status=status,
        warnings=warnings,
        plan={
            "scope": compact_scope_terms(route.paper_semantic, route.filters),
            "groups": compact_groups(route.paper_groups),
            "group_mode": route.group_mode if route.group_mode != "single" else "",
            "content_objects": parser_result.get("content_objects") or [],
            "compare_objects": parser_result.get("compare_objects") or [],
            "retrieval_query": compact_retrieval_query(retrieval_query),
        },
        resolved=common_resolved(route),
        results=results,
        parser_error=parser_error,
    )
    if debug:
        evidence["debug"] = route_debug(
            route,
            scope_records=scope_records,
            retrieval_query=retrieval_query,
            group_results=group_results,
            context_units=context_units,
        )
    return evidence


def build_reference_evidence(
    route: RouteDecision,
    *,
    status: str,
    warnings: list[str],
    source_records: list[dict[str, Any]] | None = None,
    object_records: list[dict[str, Any]] | None = None,
    answer_papers: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    group_results: list[dict[str, Any]] | None = None,
    count: int | None = None,
    exists: bool | None = None,
    parser_error: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    source_records = source_records or []
    object_records = object_records or []
    answer_papers = answer_papers or []
    edges = edges or []
    results = reference_results(
        route,
        answer_papers,
        edges,
        group_results=group_results,
        count=count,
        exists=exists,
    )
    evidence = build_base_evidence(
        route,
        status=status,
        warnings=warnings,
        plan={
            "return_side": route.return_side,
            "source_scope": compact_scope_terms(route.source_semantic, route.source_filters),
            "source_groups": compact_groups(route.source_groups),
            "source_mode": route.source_mode if route.source_mode != "single" else "",
            "object_scope": compact_scope_terms(route.object_semantic, route.object_filters),
            "object_groups": compact_groups(route.object_groups),
            "object_mode": route.object_mode if route.object_mode != "single" else "",
        },
        resolved=common_resolved(route),
        results=results,
        parser_error=parser_error,
    )
    if debug:
        evidence["debug"] = route_debug(
            route,
            source_records=source_records,
            object_records=object_records,
            answer_papers=answer_papers,
            edges=edges,
            group_results=group_results,
        )
    return evidence


def build_base_evidence(
    route: RouteDecision,
    *,
    status: str,
    warnings: list[str],
    plan: dict[str, Any],
    resolved: dict[str, Any],
    results: dict[str, Any],
    parser_error: str | None = None,
) -> dict[str, Any]:
    evidence = compact_payload({
        "query": route.query,
        "route": route.route,
        "status": status,
        "intent": route.intent,
        "plan": compact_payload(plan),
        "resolved": compact_payload(resolved),
        "warnings": list(warnings),
    })
    evidence["results"] = compact_payload(results)
    if parser_error:
        evidence["parser_error"] = parser_error
    return evidence


def common_resolved(route: RouteDecision) -> dict[str, Any]:
    aliases = [alias_match_to_dict(match) for match in route.alias_matches]
    return {"aliases": aliases} if aliases else {}


def metadata_results(
    settings,
    route: RouteDecision,
    records: list[dict[str, Any]],
    *,
    group_results: list[dict[str, Any]] | None,
    count: int | None,
    exists: bool | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if route.intent == "count":
        results["count"] = count if count is not None else len(records)
    elif route.intent == "exists":
        results["exists"] = bool(exists)
    else:
        items = [metadata_item(settings, record, route.return_fields) for record in records]
        if items:
            results["items"] = items
    if group_results:
        results["groups"] = [compact_metadata_group(settings, route, group) for group in group_results]
    return results


def compact_metadata_group(settings, route: RouteDecision, group: dict[str, Any]) -> dict[str, Any]:
    records = group.get("records") or []
    result: dict[str, Any] = {
        "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        "count": len(records),
        "exists": bool(records),
    }
    if route.intent not in {"count", "exists"}:
        items = [metadata_item(settings, record, route.return_fields) for record in records]
        if items:
            result["items"] = items
    return compact_payload(result)


def metadata_item(settings, record: dict[str, Any], return_fields: list[str]) -> dict[str, Any]:
    public_record = to_evidence_metadata_record(record)
    public_record["venue"] = display_venue(settings, public_record.get("venue"))
    item: dict[str, Any] = {"title": public_record.get("title")}
    values = {field: public_record.get(field) for field in return_fields if field != "title"}
    if values:
        item["values"] = values
    return compact_payload(item)


def content_results(
    context_units: list[dict[str, Any]],
    *,
    group_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    contexts = [compact_context_unit(unit) for unit in context_units]
    if contexts:
        results["contexts"] = contexts
    if group_results:
        results["groups"] = [compact_record_group(group) for group in group_results]
    return results


def compact_retrieval_query(retrieval_query: dict[str, Any] | None) -> dict[str, Any]:
    if not retrieval_query:
        return {}
    return compact_payload({
        "dense_query": retrieval_query.get("dense_query"),
        "bm25_queries": retrieval_query.get("bm25_queries") or [],
    })


def compact_context_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return compact_payload({
        "chunk_id": unit.get("chunk_id"),
        "title": unit.get("title"),
        "section_path": unit.get("section_path"),
        "pages": unit.get("pages"),
        "text": unit.get("chunk_text") or unit.get("text"),
    })


def compact_record_group(group: dict[str, Any]) -> dict[str, Any]:
    records = group.get("records") or []
    return compact_payload({
        "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        "papers": [paper_label(record) for record in records if paper_label(record)],
        "count": len(records),
        "exists": bool(records),
    })


def reference_results(
    route: RouteDecision,
    answer_papers: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    group_results: list[dict[str, Any]] | None,
    count: int | None,
    exists: bool | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if route.intent == "count":
        results["count"] = count if count is not None else len(answer_papers)
    elif route.intent == "exists":
        results["exists"] = bool(exists)
        compact_edges = [compact_reference_edge(edge) for edge in edges[:COMPACT_EDGE_LIMIT]]
        if compact_edges:
            results["edges"] = compact_edges
    else:
        papers = [paper_label(paper) for paper in answer_papers if paper_label(paper)]
        compact_edges = [compact_reference_edge(edge) for edge in edges]
        if papers:
            results["papers"] = papers
        if compact_edges:
            results["edges"] = compact_edges
    if group_results:
        results["groups"] = [compact_reference_group(group) for group in group_results]
    return results


def compact_reference_group(group: dict[str, Any]) -> dict[str, Any]:
    papers = [paper_label(paper) for paper in group.get("answer_papers") or [] if paper_label(paper)]
    edges = group.get("edges") or []
    return compact_payload({
        "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        "papers": papers,
        "count": len(papers),
        "exists": bool(edges),
    })


def compact_scope_terms(semantic: str, filters: list[dict[str, Any]]) -> list[str]:
    terms = [str(semantic or "").strip()]
    terms.extend(format_filter(filter_item) for filter_item in filters)
    return [term for term in terms if term]


def compact_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for group in groups:
        payload = compact_payload({
            "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        })
        if payload:
            compacted.append(payload)
    return compacted


def format_filter(filter_item: dict[str, Any]) -> str:
    field = str(filter_item.get("field") or "").strip()
    op = str(filter_item.get("op") or "").strip()
    value = filter_item.get("value")
    prefix = "not " if filter_item.get("negated") else ""
    if field == "year" and op == "interval" and isinstance(value, list) and len(value) >= 2:
        return f"{prefix}year=[{format_value(value[0])},{format_value(value[1])}]"
    if op == "in":
        return f"{prefix}{field} in {format_value(value)}"
    if op == "contains":
        return f"{prefix}{field} contains {format_value(value)}"
    if op in {"follow", "prior"}:
        return f"{prefix}{field} {op} {format_value(value)}"
    if op:
        return f"{prefix}{field}{op}{format_value(value)}"
    return f"{prefix}{field} {format_value(value)}".strip()


def format_value(value: Any) -> str:
    if isinstance(value, list):
        return "/".join(format_value(item) for item in value)
    if isinstance(value, dict):
        return "/".join(f"{key}:{format_value(item)}" for key, item in value.items())
    return str(value)


def route_debug(route: RouteDecision, **extra: Any) -> dict[str, Any]:
    debug = {
        "parser_result": route.parser_result,
        "parse_status": route.parse_status,
        "parser_error": route.parser_error,
        "route_decision": {
            "return_fields": route.return_fields,
            "paper_semantic": route.paper_semantic,
            "filters": route.filters,
            "paper_groups": route.paper_groups,
            "group_mode": route.group_mode,
            "return_side": route.return_side,
            "source_semantic": route.source_semantic,
            "source_filters": route.source_filters,
            "source_groups": route.source_groups,
            "source_mode": route.source_mode,
            "object_semantic": route.object_semantic,
            "object_filters": route.object_filters,
            "object_groups": route.object_groups,
            "object_mode": route.object_mode,
        },
    }
    debug.update(extra)
    return debug


def to_evidence_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [paper for paper in (to_evidence_paper(paper) for paper in papers) if paper is not None]


def to_evidence_paper(paper: dict[str, Any] | None) -> dict[str, Any] | None:
    if not paper:
        return None
    result = {
        "title": paper.get("title"),
        "author": paper.get("author"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
    }
    matched_alias = paper.get("matched_alias")
    if matched_alias:
        result["matched_alias"] = matched_alias
    return result


def compact_reference_edge(entry: dict[str, Any]) -> dict[str, Any]:
    edge = entry.get("edge") or {}
    return compact_payload({
        "source": paper_label(entry.get("source_paper")),
        "object": paper_label(entry.get("object_paper")),
        "ref": edge.get("raw_text") or edge.get("ref_index"),
        "page": edge.get("page"),
        "block": edge.get("source_block_id"),
    })


def paper_label(paper: dict[str, Any] | None) -> str:
    if not paper:
        return ""
    return str(paper.get("title") or paper.get("paper_id") or paper.get("_record_key") or "").strip()


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not is_empty_value(value)
    }


def is_empty_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def to_evidence_reference_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = {
        "anchor_mention": entry.get("anchor_mention"),
    }
    if entry.get("reference") is not None:
        result["reference"] = entry.get("reference")
    if entry.get("anchor_terms"):
        result["anchor_terms"] = entry.get("anchor_terms")
    if entry.get("anchor_paper"):
        result["anchor_paper"] = to_evidence_paper(entry.get("anchor_paper"))
    if entry.get("citing_paper"):
        result["citing_paper"] = to_evidence_paper(entry.get("citing_paper"))
    if entry.get("target_paper"):
        result["target_paper"] = to_evidence_paper(entry.get("target_paper"))
    return result


def to_evidence_anchor_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_mention": result.get("anchor_mention"),
        "resolved_papers": to_evidence_papers(result.get("resolved_papers") or []),
        "count": result.get("count"),
    }


def to_evidence_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": record.get("title"),
        "author": record.get("author"),
        "year": record.get("year"),
        "venue": record.get("venue"),
        "pdf_path": record.get("pdf_path"),
    }


def to_evidence_metadata_records(resolved_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_evidence_metadata_record(paper) for paper in resolved_papers]
