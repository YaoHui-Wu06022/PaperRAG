from __future__ import annotations

from typing import Any


def to_evidence_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_evidence_paper(paper) for paper in papers]


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
