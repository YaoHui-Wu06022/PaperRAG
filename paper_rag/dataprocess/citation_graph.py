from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from .manifest import Manifest, ManifestRecord, normalize_year


CITATION_GRAPH_VERSION = 1
CITATION_GRAPH_FILENAME = "citation_graph.json"


@dataclass(frozen=True)
class CitationGraphBuildResult:
    path: Path
    node_count: int
    edge_count: int


def build_citation_graph(settings: Settings, manifest: Manifest) -> CitationGraphBuildResult:
    graph_path = citation_graph_path(settings)
    papers = active_graph_papers(manifest)
    title_index = [
        (
            paper,
            normalized_title_key(str(paper.get("title") or "")),
            first_author_surname(paper.get("author") or []),
            year_candidates(paper),
        )
        for paper in papers
        if normalized_title_key(str(paper.get("title") or ""))
    ]
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, int]] = set()
    for source in papers:
        source_id = str(source["paper_id"])
        for reference in load_reference_rows(source):
            ref_index = int(reference.get("ref_index") or 0)
            raw_key = normalized_title_key(str(reference.get("raw_text") or ""))
            raw_tokens = set(normalized_tokens(str(reference.get("raw_text") or "")))
            if not raw_key:
                continue
            for target, title_key, author_surname, years in title_index:
                target_id = str(target["paper_id"])
                if source_id == target_id:
                    continue
                if not matches_local_citation(raw_key, raw_tokens, title_key, author_surname, years):
                    continue
                edge_key = (source_id, target_id, ref_index)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    "source_paper_id": source_id,
                    "target_paper_id": target_id,
                    "ref_index": ref_index,
                    "raw_text": reference.get("raw_text"),
                    "page": reference.get("page"),
                    "source_block_id": reference.get("source_block_id"),
                    "match_type": "canonical_title",
                })
    graph = {
        "version": CITATION_GRAPH_VERSION,
        "nodes": [
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "author": paper.get("author"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
            }
            for paper in papers
        ],
        "edges": edges,
    }
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CitationGraphBuildResult(graph_path, len(papers), len(edges))


def citation_graph_path(settings: Settings) -> Path:
    return settings.paper_data_dir / CITATION_GRAPH_FILENAME


def active_graph_papers(manifest: Manifest) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for record in manifest.records.values():
        paper = graph_paper_from_record(record)
        if paper:
            papers.append(paper)
    papers.sort(key=lambda paper: str(paper.get("paper_id") or ""))
    return papers


def graph_paper_from_record(record: ManifestRecord) -> dict[str, Any] | None:
    if record.status != "active" or not record.title or not record.paper_data_path:
        return None
    paper_data_path = Path(record.paper_data_path)
    return {
        "paper_id": paper_data_path.name,
        "title": record.title,
        "author": record.author,
        "year": normalize_year(record.year),
        "venue": record.venue,
        "paper_data_path": record.paper_data_path,
    }


def load_reference_rows(paper: dict[str, Any]) -> list[dict[str, Any]]:
    paper_data_path = paper.get("paper_data_path")
    if not paper_data_path:
        return []
    references_path = Path(str(paper_data_path)) / "references.jsonl"
    if not references_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with references_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "ref_index": row.get("ref_index"),
                "raw_text": row.get("raw_text"),
                "page": row.get("page"),
                "source_block_id": row.get("source_block_id"),
            })
    return rows


def normalized_title_key(value: str) -> str:
    return " ".join(normalized_tokens(value))


def normalized_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def contains_normalized_title(raw_key: str, title_key: str) -> bool:
    return bool(title_key and f" {title_key} " in f" {raw_key} ")


def matches_local_citation(
    raw_key: str,
    raw_tokens: set[str],
    title_key: str,
    first_author: str,
    years: set[int],
) -> bool:
    if not contains_normalized_title(raw_key, title_key):
        return False
    if not first_author or first_author not in raw_tokens:
        return False
    return any(str(year) in raw_tokens for year in years)


def first_author_surname(authors: Any) -> str:
    if not isinstance(authors, list) or not authors:
        return ""
    tokens = [token for token in normalized_tokens(str(authors[0])) if not token.isdigit()]
    return tokens[-1] if tokens else ""


def year_candidates(paper: dict[str, Any]) -> set[int]:
    years: set[int] = set()
    year = normalize_year(paper.get("year"))
    for key in ("preprint_year", "publish_year"):
        value = year.get(key)
        if value:
            years.add(int(value))
    return years
