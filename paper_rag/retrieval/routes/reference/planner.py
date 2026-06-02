"""reference planner：基于本地 citation graph 执行引用关系查询。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paper_rag.config import Settings
from paper_rag.retrieval.evidence import build_reference_evidence
from paper_rag.retrieval.route import RouteDecision
from paper_rag.corpus.citation_index import load_citation_graph
from paper_rag.corpus.records import dedupe_paper_records, paper_record_key
from paper_rag.corpus.scope import combined_semantic, records_for_scope

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def plan_reference(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
    *,
    debug: bool = False,
    corpus: "CorpusContext | None" = None,
) -> dict[str, Any]:
    """执行 source_scope --cites--> object_scope 查询。"""
    if route.parse_status == "parse_failed":
        return build_reference_evidence(
            route,
            status="parse_failed",
            warnings=warnings,
            answer_papers=[],
            edges=[],
            parser_error=route.parser_error,
            debug=debug,
        )

    graph = corpus.citation_graph if corpus else load_citation_graph(settings)
    if not graph:
        # reference 不再临时扫描 references.jsonl，图缺失时明确提示先 ingest。
        warnings.append("reference route requires data/paper_data/citation_graph.json; run paper-rag ingest first")
        return build_reference_evidence(
            route,
            status="graph_missing",
            warnings=warnings,
            answer_papers=[],
            edges=[],
            debug=debug,
        )

    source_scope = scope_result(settings, route.source_semantic, route.source_filters, route.source_groups, route.source_mode, corpus=corpus)
    object_scope = scope_result(settings, route.object_semantic, route.object_filters, route.object_groups, route.object_mode, corpus=corpus)
    source_nodes = node_index(graph, source_scope["records"])
    object_nodes = node_index(graph, object_scope["records"])
    edges = matching_edges(graph, source_nodes, object_nodes)
    answer_papers = answer_papers_for_edges(edges, route.return_side)
    group_results = (
        build_group_results(settings, route, graph, corpus=corpus)
        if route.source_mode != "single" or route.object_mode != "single"
        else []
    )
    if group_results:
        answer_papers, edges = fold_group_results(route, group_results, answer_papers, edges)
    count = len(answer_papers) if route.intent == "count" else None
    exists = bool(edges) if route.intent == "exists" else None
    if not answer_papers and route.intent != "exists":
        warnings.append("reference route found no matching citation edges")
    if not edges:
        warnings.append("reference route found no matching citation edges")
    return build_reference_evidence(
        route,
        status="ok",
        warnings=warnings,
        source_records=source_scope["records"],
        object_records=object_scope["records"],
        answer_papers=answer_papers,
        edges=edges,
        group_results=group_results or None,
        count=count,
        exists=exists,
        debug=debug,
    )


def build_group_results(
    settings: Settings,
    route: RouteDecision,
    graph: dict[str, Any],
    *,
    corpus: "CorpusContext | None" = None,
) -> list[dict[str, Any]]:
    """根据有分组的一侧构建逐组引用查询结果。"""
    if route.source_mode != "single":
        return side_group_results(settings, route, graph, "source", corpus=corpus)
    if route.object_mode != "single":
        return side_group_results(settings, route, graph, "object", corpus=corpus)
    return []


def side_group_results(
    settings: Settings,
    route: RouteDecision,
    graph: dict[str, Any],
    side: str,
    *,
    corpus: "CorpusContext | None" = None,
) -> list[dict[str, Any]]:
    """对 source 或 object 一侧的 groups 逐组执行引用匹配。"""
    shared_semantic = route.source_semantic if side == "source" else route.object_semantic
    shared_filters = route.source_filters if side == "source" else route.object_filters
    groups = route.source_groups if side == "source" else route.object_groups
    if not groups:
        return []
    fixed_source_scope = None
    fixed_object_scope = None
    if side == "source":
        fixed_object_scope = scope_result(
            settings,
            route.object_semantic,
            route.object_filters,
            route.object_groups,
            route.object_mode,
            corpus=corpus,
        )
    else:
        fixed_source_scope = scope_result(
            settings,
            route.source_semantic,
            route.source_filters,
            route.source_groups,
            route.source_mode,
            corpus=corpus,
        )
    results: list[dict[str, Any]] = []
    for group in groups:
        semantic = combined_semantic(shared_semantic, group.get("semantic") or "")
        filters = [*shared_filters, *(group.get("filters") or [])]
        if side == "source":
            source_scope = scope_result(settings, semantic, filters, [], "single", corpus=corpus)
            object_scope = fixed_object_scope
        else:
            source_scope = fixed_source_scope
            object_scope = scope_result(settings, semantic, filters, [], "single", corpus=corpus)
        source_nodes = node_index(graph, source_scope["records"])
        object_nodes = node_index(graph, object_scope["records"])
        edges = matching_edges(graph, source_nodes, object_nodes)
        answer_papers = answer_papers_for_edges(edges, route.return_side)
        results.append({
            "semantic": group.get("semantic") or "",
            "filters": group.get("filters") or [],
            "answer_papers": answer_papers,
            "edges": edges,
        })
    return results


def scope_result(
    settings: Settings,
    semantic: str,
    filters: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    mode: str,
    *,
    corpus: "CorpusContext | None" = None,
) -> dict[str, Any]:
    """把某一侧 scope 转成候选 records，并保留原始 scope 摘要。"""
    if mode in {"per", "or", "and"}:
        records = dedupe_paper_records([
            record
            for group in groups
            for record in records_for_scope(
                settings,
                combined_semantic(semantic, group.get("semantic") or ""),
                [*filters, *(group.get("filters") or [])],
                mode,
                corpus=corpus,
            )
        ])
    else:
        records = records_for_scope(settings, semantic, filters, mode, corpus=corpus)
    return {
        "semantic": semantic,
        "filters": filters,
        "groups": groups,
        "mode": mode,
        "records": records,
    }


def node_index(graph: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把 records 对齐到 citation graph nodes，供 edge 过滤使用。"""
    nodes_by_id = {
        str(node.get("paper_id") or ""): node
        for node in graph.get("nodes") or []
        if str(node.get("paper_id") or "").strip()
    }
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        paper_id = paper_record_key(record)
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
    """找出 source_nodes 到 object_nodes 之间存在的 citation graph 边。"""
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
    """根据 return_side 从命中边中抽取答案端论文。"""
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
    """按 per/or/and 聚合 group 结果。"""
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
    """返回当前 reference 查询中真正启用分组的一侧 mode。"""
    if route.source_mode != "single":
        return route.source_mode
    if route.object_mode != "single":
        return route.object_mode
    return "single"


def unique_papers_from_groups(group_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从多个 group 的 answer_papers 中保序去重。"""
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
    """从多个 group 的 edges 中按 source/object/ref_index 去重。"""
    seen: set[tuple[str, str, Any]] = set()
    edges: list[dict[str, Any]] = []
    for group in group_results:
        for edge in group.get("edges") or []:
            key = (
                paper_id(edge.get("source_paper")),
                paper_id(edge.get("object_paper")),
                (edge.get("edge") or {}).get("ref_index"),
            )
            if key not in seen:
                seen.add(key)
                edges.append(edge)
    return edges


def intersect_group_answer_ids(group_results: list[dict[str, Any]]) -> set[str]:
    """计算所有 group 共同命中的答案论文 id。"""
    answer_sets = [
        {paper_id(paper) for paper in group.get("answer_papers") or [] if paper_id(paper)}
        for group in group_results
    ]
    if not answer_sets:
        return set()
    return set.intersection(*answer_sets)


def paper_id(paper: dict[str, Any] | None) -> str:
    """从 graph paper 或 manifest record 中取统一 paper id。"""
    if not paper:
        return ""
    return str(paper.get("paper_id") or paper_record_key(paper) or "")


def edge_answer_paper_id(edge: dict[str, Any], return_side: str | None) -> str:
    """取一条 edge 在当前 return_side 下对应的答案论文 id。"""
    if return_side == "source":
        return paper_id(edge.get("source_paper"))
    return paper_id(edge.get("object_paper"))
