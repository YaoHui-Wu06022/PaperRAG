from __future__ import annotations

from typing import Any

from ....config import Settings
from ...data.aliases import alias_match_to_dict
from ...evidence import to_evidence_paper, to_evidence_papers
from ...route import RouteDecision
from ...data.citation_scope import load_citation_graph
from ...data.paper_scope import combined_semantic, record_key_from_dict, records_for_scope, unique_records


def plan_reference(settings: Settings, route: RouteDecision, warnings: list[str]) -> dict[str, Any]:
    if route.parse_status == "parse_failed":
        return {
            **build_reference_evidence_base(route),
            "parse_status": "parse_failed",
            "parser_error": route.parser_error,
            "answer_papers": [],
            "edges": [],
        }

    graph = load_citation_graph(settings)
    if not graph:
        warnings.append("reference route requires data/paper_data/citation_graph.json; run paper-rag ingest first")
        return {
            **build_reference_evidence_base(route),
            "parse_status": "graph_missing",
            "answer_papers": [],
            "edges": [],
        }

    source_scope = scope_result(settings, route.source_semantic, route.source_filters, route.source_groups, route.source_mode)
    object_scope = scope_result(settings, route.object_semantic, route.object_filters, route.object_groups, route.object_mode)
    evidence = build_reference_evidence(settings, route, graph, source_scope, object_scope)
    if route.intent == "count":
        evidence["count"] = len(evidence["answer_papers"])
    if route.intent == "exists":
        evidence["exists"] = bool(evidence["edges"])
    if not evidence["answer_papers"] and route.intent != "exists":
        warnings.append("reference route found no matching citation edges")
    if not evidence["edges"]:
        warnings.append("reference route found no matching citation edges")
    return evidence


def build_reference_evidence_base(route: RouteDecision) -> dict[str, Any]:
    return {
        "intent": route.intent,
        "return_side": route.return_side,
        "source_scope": {
            "semantic": route.source_semantic,
            "filters": route.source_filters,
            "groups": route.source_groups,
            "mode": route.source_mode,
        },
        "object_scope": {
            "semantic": route.object_semantic,
            "filters": route.object_filters,
            "groups": route.object_groups,
            "mode": route.object_mode,
        },
        "alias_matches": [alias_match_to_dict(match) for match in route.alias_matches],
    }


def build_reference_evidence(
    settings: Settings,
    route: RouteDecision,
    graph: dict[str, Any],
    source_scope: dict[str, Any],
    object_scope: dict[str, Any],
) -> dict[str, Any]:
    source_nodes = node_index(graph, source_scope["records"])
    object_nodes = node_index(graph, object_scope["records"])
    edges = matching_edges(graph, source_nodes, object_nodes)
    answer_papers = answer_papers_for_edges(edges, route.return_side)
    group_results = (
        build_group_results(settings, route, graph)
        if route.source_mode != "single" or route.object_mode != "single"
        else []
    )
    if group_results:
        answer_papers, edges = fold_group_results(route, group_results, answer_papers, edges)
    evidence: dict[str, Any] = {
        **build_reference_evidence_base(route),
        "parse_status": "ok",
        "source_records": to_evidence_papers(source_scope["records"]),
        "object_records": to_evidence_papers(object_scope["records"]),
        "answer_papers": to_evidence_papers(answer_papers),
        "edges": [to_evidence_edge(edge) for edge in edges],
    }
    if group_results:
        evidence["group_results"] = group_results
    return evidence


def build_group_results(settings: Settings, route: RouteDecision, graph: dict[str, Any]) -> list[dict[str, Any]]:
    if route.source_mode != "single":
        return side_group_results(settings, route, graph, "source")
    if route.object_mode != "single":
        return side_group_results(settings, route, graph, "object")
    return []


def side_group_results(settings: Settings, route: RouteDecision, graph: dict[str, Any], side: str) -> list[dict[str, Any]]:
    shared_semantic = route.source_semantic if side == "source" else route.object_semantic
    shared_filters = route.source_filters if side == "source" else route.object_filters
    groups = route.source_groups if side == "source" else route.object_groups
    results: list[dict[str, Any]] = []
    for group in groups:
        semantic = combined_semantic(shared_semantic, group.get("semantic") or "")
        filters = [*shared_filters, *(group.get("filters") or [])]
        if side == "source":
            source_scope = scope_result(settings, semantic, filters, [], "single")
            object_scope = scope_result(settings, route.object_semantic, route.object_filters, route.object_groups, route.object_mode)
        else:
            source_scope = scope_result(settings, route.source_semantic, route.source_filters, route.source_groups, route.source_mode)
            object_scope = scope_result(settings, semantic, filters, [], "single")
        source_nodes = node_index(graph, source_scope["records"])
        object_nodes = node_index(graph, object_scope["records"])
        edges = matching_edges(graph, source_nodes, object_nodes)
        answer_papers = answer_papers_for_edges(edges, route.return_side)
        results.append({
            "semantic": group.get("semantic") or "",
            "filters": group.get("filters") or [],
            "answer_papers": to_evidence_papers(answer_papers),
            "edges": [to_evidence_edge(edge) for edge in edges],
            "count": len(answer_papers),
            "exists": bool(edges),
        })
    return results


def scope_result(
    settings: Settings,
    semantic: str,
    filters: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    if mode == "per":
        group_records = [
            records_for_scope(settings, combined_semantic(semantic, group.get("semantic") or ""), [*filters, *(group.get("filters") or [])], mode)
            for group in groups
        ]
        records = unique_records([record for group in group_records for record in group])
    elif mode == "or":
        records = unique_records([
            record
            for group in groups
            for record in records_for_scope(
                settings,
                combined_semantic(semantic, group.get("semantic") or ""),
                [*filters, *(group.get("filters") or [])],
                mode,
            )
        ])
    elif mode == "and":
        records = unique_records([
            record
            for group in groups
            for record in records_for_scope(
                settings,
                combined_semantic(semantic, group.get("semantic") or ""),
                [*filters, *(group.get("filters") or [])],
                mode,
            )
        ])
    else:
        records = records_for_scope(settings, semantic, filters, mode)
    return {
        "semantic": semantic,
        "filters": filters,
        "groups": groups,
        "mode": mode,
        "records": records,
    }


def node_index(graph: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nodes_by_id = {
        str(node.get("paper_id") or ""): node
        for node in graph.get("nodes") or []
        if str(node.get("paper_id") or "").strip()
    }
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        paper_id = record_key_from_dict(record)
        if paper_id and paper_id in nodes_by_id:
            index[paper_id] = {
                "paper_id": paper_id,
                "title": record.get("title") or nodes_by_id[paper_id].get("title"),
                "author": record.get("author") or nodes_by_id[paper_id].get("author"),
                "year": record.get("year") or nodes_by_id[paper_id].get("year"),
                "venue": record.get("venue") or nodes_by_id[paper_id].get("venue"),
                "pdf_path": record.get("pdf_path"),
                "paper_data_path": record.get("paper_data_path"),
            }
    return index


def matching_edges(
    graph: dict[str, Any],
    source_nodes: dict[str, dict[str, Any]],
    object_nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for edge in graph.get("edges") or []:
        source_paper_id = str(edge.get("source_paper_id") or "")
        object_paper_id = str(edge.get("target_paper_id") or "")
        if source_paper_id not in source_nodes or object_paper_id not in object_nodes:
            continue
        edges.append({
            "source_paper": source_nodes[source_paper_id],
            "object_paper": object_nodes[object_paper_id],
            "edge": edge,
        })
    return edges


def answer_papers_for_edges(edges: list[dict[str, Any]], return_side: str | None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    side_key = "source_paper" if return_side == "source" else "object_paper"
    for edge in edges:
        paper = edge.get(side_key) or {}
        paper_id = str(paper.get("paper_id") or "")
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            papers.append(paper)
    return papers


def fold_group_results(
    route: RouteDecision,
    group_results: list[dict[str, Any]],
    answer_papers: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = grouped_side_mode(route)
    if mode == "per":
        return answer_papers, edges
    if mode == "or":
        return unique_papers_from_groups(group_results), unique_edges_from_groups(group_results)
    if mode == "and":
        intersection_ids = intersect_group_answer_ids(group_results)
        filtered_papers = [paper for paper in unique_papers_from_groups(group_results) if paper_id(paper) in intersection_ids]
        filtered_edges = [
            edge
            for edge in unique_edges_from_groups(group_results)
            if edge_answer_paper_id(edge, route.return_side) in intersection_ids
        ]
        return filtered_papers, filtered_edges
    return answer_papers, edges


def grouped_side_mode(route: RouteDecision) -> str:
    if route.source_mode != "single":
        return route.source_mode
    if route.object_mode != "single":
        return route.object_mode
    return "single"


def unique_papers_from_groups(group_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    for group in group_results:
        for paper in group.get("answer_papers") or []:
            key = paper_id(paper)
            if key and key not in seen:
                seen.add(key)
                papers.append(paper)
    return papers


def unique_edges_from_groups(group_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, Any]] = set()
    edges: list[dict[str, Any]] = []
    for group in group_results:
        for edge in group.get("edges") or []:
            key = (
                paper_id(edge.get("source_paper")),
                paper_id(edge.get("object_paper")),
                edge.get("ref_index"),
            )
            if key not in seen:
                seen.add(key)
                edges.append(edge)
    return edges


def intersect_group_answer_ids(group_results: list[dict[str, Any]]) -> set[str]:
    answer_sets = [
        {paper_id(paper) for paper in group.get("answer_papers") or [] if paper_id(paper)}
        for group in group_results
    ]
    if not answer_sets:
        return set()
    return set.intersection(*answer_sets)


def paper_id(paper: dict[str, Any] | None) -> str:
    if not paper:
        return ""
    return str(paper.get("paper_id") or record_key_from_dict(paper) or "")


def edge_answer_paper_id(edge: dict[str, Any], return_side: str | None) -> str:
    if return_side == "source":
        return paper_id(edge.get("source_paper"))
    return paper_id(edge.get("object_paper"))


def to_evidence_edge(entry: dict[str, Any]) -> dict[str, Any]:
    edge = entry.get("edge") or {}
    return {
        "source_paper": to_evidence_paper(entry.get("source_paper")),
        "object_paper": to_evidence_paper(entry.get("object_paper")),
        "ref_index": edge.get("ref_index"),
        "raw_text": edge.get("raw_text"),
        "page": edge.get("page"),
        "source_block_id": edge.get("source_block_id"),
        "match_type": edge.get("match_type"),
    }
