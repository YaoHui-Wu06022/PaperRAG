"""统一构建 planner 输出的 composer/debug evidence。"""

from __future__ import annotations

from typing import Any

from ..dataprocess.venues import display_venue
from .data.aliases_match import alias_match_to_dict
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
    """构建 metadata route 的默认压缩 evidence，debug 时附加完整中间态。"""
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
    """构建 content route evidence，默认只暴露检索 query 和精简 contexts。"""
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
    """构建 reference route evidence，默认输出答案论文和精简边证据。"""
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
    """三条 route 共享的 evidence 骨架，负责移除空字段。"""
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
    """生成三条 route 共用的 resolved 摘要，目前只保留 alias 命中。"""
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
    """按 metadata intent 压缩 records/count/exists/group results。"""
    results: dict[str, Any] = {}
    if route.intent == "count":
        results["count"] = count if count is not None else len(records)
        items = [metadata_item(settings, record, route.return_fields) for record in records]
        if items:
            results["items"] = items
    elif route.intent == "exists":
        results["exists"] = bool(exists)
        if not exists:
            actual_items = metadata_actual_items(settings, route)
            if actual_items:
                results["actual"] = actual_items
    else:
        items = [metadata_item(settings, record, route.return_fields) for record in records]
        if items:
            results["items"] = items
    if group_results:
        results["groups"] = [compact_metadata_group(settings, route, group) for group in group_results]
    return results


def metadata_actual_items(settings, route: RouteDecision) -> list[dict[str, Any]]:
    """metadata exists=false 时展示被判断论文的真实字段。"""
    if not route.resolved_papers:
        return []
    fields = metadata_actual_fields(route.filters)
    return [metadata_item(settings, record, fields) for record in route.resolved_papers]


def metadata_actual_fields(filters: list[dict[str, Any]]) -> list[str]:
    """从失败的 metadata filters 中推断需要展示的真实字段。"""
    fields: list[str] = []
    for filter_item in filters:
        field = filter_item.get("field")
        if field in {"author", "year", "venue", "title"} and field not in fields:
            fields.append(field)
    return fields or ["title"]


def compact_metadata_group(settings, route: RouteDecision, group: dict[str, Any]) -> dict[str, Any]:
    """把 metadata group 命中压缩成 scope/count/items。"""
    records = group.get("records") or []
    result: dict[str, Any] = {
        "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        "count": len(records),
        "exists": bool(records),
    }
    if route.intent != "exists":
        items = [metadata_item(settings, record, route.return_fields) for record in records]
        if items:
            result["items"] = items
    return compact_payload(result)


def metadata_item(settings, record: dict[str, Any], return_fields: list[str]) -> dict[str, Any]:
    """把单条 metadata record 转为 composer 需要的 item。"""
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
    """压缩 content 命中的 contexts 和 group records。"""
    results: dict[str, Any] = {}
    contexts = [compact_context_unit(unit) for unit in context_units]
    if contexts:
        results["contexts"] = contexts
    if group_results:
        results["groups"] = [compact_record_group(group) for group in group_results]
    return results


def compact_retrieval_query(retrieval_query: dict[str, Any] | None) -> dict[str, Any]:
    """默认 evidence 只展示 dense_query 和 bm25_queries。"""
    if not retrieval_query:
        return {}
    return compact_payload({
        "dense_query": retrieval_query.get("dense_query"),
        "bm25_queries": retrieval_query.get("bm25_queries") or [],
    })


def compact_context_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """裁剪 context_unit，隐藏 expanded blocks、scores 等 debug 信息。"""
    return compact_payload({
        "chunk_id": unit.get("chunk_id"),
        "title": unit.get("title"),
        "section_path": unit.get("section_path"),
        "pages": unit.get("pages"),
        "text": unit.get("chunk_text") or unit.get("text"),
    })


def compact_record_group(group: dict[str, Any]) -> dict[str, Any]:
    """把 content group 的论文 records 压成标题列表和计数。"""
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
    """按 reference intent 压缩答案论文、边证据和分组结果。"""
    results: dict[str, Any] = {}
    if route.intent == "count":
        results["count"] = count if count is not None else len(answer_papers)
        papers = [paper_label(paper) for paper in answer_papers if paper_label(paper)]
        compact_edges = [compact_reference_edge(edge) for edge in edges[:COMPACT_EDGE_LIMIT]]
        if papers:
            results["papers"] = papers
        if compact_edges:
            results["edges"] = compact_edges
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
    """把 reference group 输出压成 scope/papers/count/exists。"""
    papers = [paper_label(paper) for paper in group.get("answer_papers") or [] if paper_label(paper)]
    edges = group.get("edges") or []
    return compact_payload({
        "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        "papers": papers,
        "count": len(papers),
        "exists": bool(edges),
    })


def compact_scope_terms(semantic: str, filters: list[dict[str, Any]]) -> list[str]:
    """把 semantic + filter dict 转成短文本 scope 列表。"""
    terms = [str(semantic or "").strip()]
    terms.extend(format_filter(filter_item) for filter_item in filters)
    return [term for term in terms if term]


def compact_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 paper_groups 压缩为仅含 scope 的列表。"""
    compacted: list[dict[str, Any]] = []
    for group in groups:
        payload = compact_payload({
            "scope": compact_scope_terms(group.get("semantic") or "", group.get("filters") or []),
        })
        if payload:
            compacted.append(payload)
    return compacted


def format_filter(filter_item: dict[str, Any]) -> str:
    """把一个结构化 filter 格式化成 composer 可读短文本。"""
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
    """把 filter value 转成稳定的短文本。"""
    if isinstance(value, list):
        return "/".join(format_value(item) for item in value)
    if isinstance(value, dict):
        return "/".join(f"{key}:{format_value(item)}" for key, item in value.items())
    return str(value)


def route_debug(route: RouteDecision, **extra: Any) -> dict[str, Any]:
    """生成 debug 模式下完整 route/parser/result 中间态。"""
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


def compact_reference_edge(entry: dict[str, Any]) -> dict[str, Any]:
    """把 citation edge 压缩成 source/object/ref/page/block。"""
    edge = entry.get("edge") or {}
    return compact_payload({
        "source": paper_label(entry.get("source_paper")),
        "object": paper_label(entry.get("object_paper")),
        "ref": edge.get("raw_text") or edge.get("ref_index"),
        "page": edge.get("page"),
        "block": edge.get("source_block_id"),
    })


def paper_label(paper: dict[str, Any] | None) -> str:
    """取 composer 中展示论文时使用的短标签。"""
    if not paper:
        return ""
    return str(paper.get("title") or paper.get("paper_id") or paper.get("_record_key") or "").strip()


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """移除空值字段，控制默认 evidence 体积。"""
    return {
        key: value
        for key, value in payload.items()
        if not is_empty_value(value)
    }


def is_empty_value(value: Any) -> bool:
    """判断某个字段是否应该从默认 evidence 中省略。"""
    return value is None or value == "" or value == [] or value == {}


def to_evidence_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    """裁剪 metadata record，隐藏 hash 和内部路径。"""
    return {
        "title": record.get("title"),
        "author": record.get("author"),
        "year": record.get("year"),
        "venue": record.get("venue"),
        "pdf_path": record.get("pdf_path"),
    }
