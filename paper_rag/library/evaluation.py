"""Evidence-span retrieval evaluation; semantic answer grades are external."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .common import LibraryError, atomic_text, digest, parse_evidence
from .contracts import (
    EvalRequest,
    SearchRequest,
    PapersRequest,
    CitationRequest,
    ReadRequest,
)
from .reading import get_evidence, papers, citations, read
from .search import search


def overlaps(left: str, right: str) -> bool:
    a, b = parse_evidence(left), parse_evidence(right)
    return a[:3] == b[:3] and max(a[3], b[3]) < min(a[4], b[4])


def evaluate(cat, request: EvalRequest):
    source = Path(request.cases_path).read_text(encoding="utf-8-sig")
    benchmark = json.loads(source)
    if benchmark.get("corpus_version") != cat.snapshot():
        raise LibraryError(
            "snapshot_mismatch", "Benchmark must name the current corpus version"
        )
    cases = benchmark.get("cases", [])
    if not cases:
        raise LibraryError("invalid_benchmark", "No reviewed evaluation cases")
    groups, ids, results = {}, set(), []
    for case in cases:
        for group in [case["group"], *case.get("source_groups", [])]:
            if groups.setdefault(group, case["split"]) != case["split"]:
                raise LibraryError(
                    "split_leakage",
                    "Shared source evidence cannot cross tune/test splits",
                )
    for case in cases:
        if not case.get("reviewed") or case["id"] in ids:
            raise LibraryError(
                "invalid_benchmark", "Every case needs unique ID and source review"
            )
        ids.add(case["id"])
        split, group = case["split"], case["group"]
        if split not in {"tune", "test"} or groups.setdefault(group, split) != split:
            raise LibraryError(
                "split_leakage", "Evidence groups cannot cross tune/test splits"
            )
        truth = case["evidence_ids"]
        for ref in truth:
            ev = get_evidence(cat, ref)
            if ev["historical"] or ev["withdrawn"]:
                raise LibraryError(
                    "stale_benchmark", "Ground truth references inactive evidence"
                )
        query = SearchRequest(
            query=case["query"],
            mode=request.mode,
            limit=min(50, request.top_k),
            candidate_limit=max(40, request.top_k),
            regions=case.get("regions", ["abstract", "body"]),
            filters=case.get("filters", {}),
        )
        response = search(cat, query)
        hits = response["items"]
        while len(hits) < request.top_k and response["next_cursor"]:
            query = query.model_copy(update={"cursor": response["next_cursor"]})
            response = search(cat, query)
            hits.extend(response["items"])
        hits = hits[: request.top_k]
        found, ranks, gains, invalid = set(), [], [], 0
        for rank, hit in enumerate(hits, 1):
            for ref in hit["evidence_ids"]:
                try:
                    ev = get_evidence(cat, ref)
                    invalid += int(ev["historical"] or ev["withdrawn"])
                except LibraryError:
                    invalid += 1
            matched = {
                n
                for n, gold in enumerate(truth)
                if any(overlaps(gold, ref) for ref in hit["evidence_ids"])
            }
            novel = matched - found
            gains.append(int(bool(novel)))
            found.update(matched)
            if matched:
                ranks.append(rank)
        ideal = sum(1 / math.log2(n + 2) for n in range(min(len(truth), len(hits))))
        results.append(
            {
                "id": case["id"],
                "split": split,
                "hit": int(bool(found)),
                "evidence_recall": len(found) / len(truth) if truth else None,
                "mrr": 1 / min(ranks) if ranks else 0,
                "ndcg": (
                    sum(g / math.log2(n + 2) for n, g in enumerate(gains)) / ideal
                    if ideal
                    else None
                ),
                "invalid_citations": invalid,
                "degraded": response["degraded"],
                "warnings": response["warnings"],
                "retrieval_ms": response["timings"]["total_ms"],
                "unanswerable": not truth,
                "hits": hits,
            }
        )
    summary = {}
    for split in ("tune", "test"):
        rows = [r for r in results if r["split"] == split and not r["unanswerable"]]
        summary[split] = {
            key: sum(r[key] or 0 for r in rows) / len(rows) if rows else None
            for key in ("hit", "evidence_recall", "mrr", "ndcg")
        }
    report = {
        "schema_version": 1,
        "corpus_version": cat.snapshot(),
        "benchmark_hash": digest(source),
        "mode": request.mode,
        "top_k": request.top_k,
        "summary": summary,
        "cases": results,
        "metric_definition": "Relevance requires overlap of versioned block character spans, never title or document-only matches. Recall counts gold spans with overlap; semantic support is separately reviewed.",
        "degraded_cases": sum(r["degraded"] for r in results),
        "release_blocked": any(
            r["invalid_citations"] or r["degraded"] for r in results
        ),
        "semantic_answer_quality": "requires_host_agent_and_source_review",
    }
    tool_results = []
    for case in benchmark.get("tool_cases", []):
        if not case.get("reviewed"):
            raise LibraryError(
                "invalid_benchmark", "Tool cases also require source review"
            )
        if case["command"] == "papers_count":
            data = papers(
                cat, PapersRequest.model_validate(case["request"]), count=True
            )
            passed = data["count"] == case["expected_count"]
        elif case["command"] == "citations":
            req = CitationRequest.model_validate(case["request"])
            items = []
            while True:
                data = citations(cat, req)
                items.extend(data["items"])
                if data["next_offset"] is None:
                    break
                req = req.model_copy(update={"offset": data["next_offset"]})
            passed = any(
                item["target_document_id"] == case["expected_target"]
                and case["expected_quote"] in item["raw_text"]
                for item in items
            )
        elif case["command"] == "read":
            req = ReadRequest.model_validate(case["request"])
            text = []
            while True:
                data = read(cat, req)
                text.extend(item["text"] for item in data["fragments"])
                if not data["next_cursor"]:
                    break
                req = req.model_copy(update={"cursor": data["next_cursor"]})
            passed = digest("".join(text)) == case["expected_hash"]
        else:
            raise LibraryError("invalid_benchmark", "Unknown tool case command")
        tool_results.append(
            {"id": case["id"], "command": case["command"], "passed": passed}
        )
    report["tool_cases"] = tool_results
    report["release_blocked"] |= any(not row["passed"] for row in tool_results)
    if request.traces_path:
        report["agent_traces"] = json.loads(
            Path(request.traces_path).read_text(encoding="utf-8-sig")
        )
        report["trace_note"] = (
            "Externally supplied traces; retrieval metrics do not grade answer support."
        )
    if request.output_path:
        atomic_text(
            Path(request.output_path),
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
    return report
