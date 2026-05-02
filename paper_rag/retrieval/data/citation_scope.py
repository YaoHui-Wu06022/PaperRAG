from __future__ import annotations

import json
from typing import Any

from ...config import Settings
from .filters import compare_text


def load_citation_graph(settings: Settings) -> dict[str, Any] | None:
    path = settings.paper_data_dir / "citation_graph.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def citation_scope_paper_ids(settings: Settings, titles: list[str], relation: str) -> set[str]:
    graph = load_citation_graph(settings)
    if not graph:
        return set()
    target_ids = paper_ids_for_titles(graph, titles)
    if not target_ids:
        return set()
    edges = graph.get("edges") or []
    if relation == "follow":
        return {
            str(edge.get("source_paper_id"))
            for edge in edges
            if edge.get("target_paper_id") in target_ids and edge.get("source_paper_id")
        }
    if relation == "prior":
        return {
            str(edge.get("target_paper_id"))
            for edge in edges
            if edge.get("source_paper_id") in target_ids and edge.get("target_paper_id")
        }
    return set()


def record_matches_citation_scope(settings: Settings, paper_id: str, titles: list[str], relation: str) -> bool:
    if not paper_id:
        return False
    return paper_id in citation_scope_paper_ids(settings, titles, relation)


def paper_ids_for_titles(graph: dict[str, Any], titles: list[str]) -> set[str]:
    ids: set[str] = set()
    for node in graph.get("nodes") or []:
        if any(compare_text(node.get("title"), "=", title) for title in titles):
            paper_id = str(node.get("paper_id") or "").strip()
            if paper_id:
                ids.add(paper_id)
    return ids
