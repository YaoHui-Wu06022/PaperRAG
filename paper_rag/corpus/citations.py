"""基于本地 citation graph 计算 paper follow/prior 范围。"""

from __future__ import annotations

import json
from typing import Any

from paper_rag.config import Settings
from .filters import compare_text


def load_citation_graph(settings: Settings) -> dict[str, Any] | None:
    """读取本地 citation_graph.json。"""
    path = settings.paper_data_dir / "citation_graph.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def citation_scope_paper_ids(settings: Settings, titles: list[str], relation: str) -> set[str]:
    """根据 paper follow/prior 关系返回候选论文 id。"""
    graph = load_citation_graph(settings)
    if not graph:
        return set()
    target_ids = paper_ids_for_titles(graph, titles)
    if not target_ids:
        return set()
    edges = graph.get("edges") or []
    if relation == "follow":
        # follow: 当前论文在目标论文之后，即它引用了目标论文。
        return {
            str(edge.get("source_paper_id"))
            for edge in edges
            if edge.get("target_paper_id") in target_ids and edge.get("source_paper_id")
        }
    if relation == "prior":
        # prior: 当前论文在目标论文之前，即被目标论文引用。
        return {
            str(edge.get("target_paper_id"))
            for edge in edges
            if edge.get("source_paper_id") in target_ids and edge.get("target_paper_id")
        }
    return set()


def record_matches_citation_scope(settings: Settings, paper_id: str, titles: list[str], relation: str) -> bool:
    """判断某个 paper_id 是否位于 citation scope 中。"""
    if not paper_id:
        return False
    return paper_id in citation_scope_paper_ids(settings, titles, relation)


def paper_ids_for_titles(graph: dict[str, Any], titles: list[str]) -> set[str]:
    """根据论文标题在 citation graph nodes 中找到 paper_id。"""
    ids: set[str] = set()
    for node in graph.get("nodes") or []:
        if any(compare_text(node.get("title"), "=", title) for title in titles):
            paper_id = str(node.get("paper_id") or "").strip()
            if paper_id:
                ids.add(paper_id)
    return ids
