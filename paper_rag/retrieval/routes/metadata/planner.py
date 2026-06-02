"""metadata planner：根据论文 scope 查询 manifest records。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paper_rag.config import Settings
from paper_rag.corpus.records import dedupe_paper_records
from paper_rag.corpus.scope import combined_semantic, records_for_scope
from paper_rag.retrieval.evidence import build_metadata_evidence
from paper_rag.retrieval.route import RouteDecision

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def plan_metadata(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
    *,
    debug: bool = False,
    corpus: "CorpusContext | None" = None,
) -> dict[str, Any]:
    """执行 metadata 查询，并交给 evidence builder 输出 composer/debug 结果。"""
    if route.parse_status == "parse_failed":
        return build_metadata_evidence(
            settings,
            route,
            status="parse_failed",
            warnings=warnings,
            records=[],
            parser_error=route.parser_error,
            debug=debug,
        )

    group_results: list[dict[str, Any]] | None = None
    exists: bool | None = None
    if route.group_mode in {"per", "and"}:
        # per/and 需要保留每个 group 的独立结果，方便 evidence 展示。
        group_results = metadata_per_group_results(settings, route, corpus=corpus)
        records = dedupe_paper_records([record for group in group_results for record in group["records"]])
        if route.group_mode == "and":
            exists = all(bool(group["records"]) for group in group_results)
    else:
        records = metadata_scope_records(settings, route, corpus=corpus)
        if route.intent == "exists":
            exists = bool(records)

    count = len(records) if route.intent == "count" else None
    if not records:
        warnings.append("metadata route found no matching manifest records")
    return build_metadata_evidence(
        settings,
        route,
        status="ok",
        warnings=warnings,
        records=records,
        group_results=group_results,
        count=count,
        exists=exists,
        debug=debug,
    )


def metadata_per_group_results(
    settings: Settings,
    route: RouteDecision,
    *,
    corpus: "CorpusContext | None" = None,
) -> list[dict[str, Any]]:
    """逐个执行 paper_groups，保留每组命中的 records。"""
    results: list[dict[str, Any]] = []
    for group in route.paper_groups:
        semantic = combined_semantic(route.paper_semantic, group.get("semantic") or "")
        filters = [*route.filters, *(group.get("filters") or [])]
        records = records_for_scope(settings, semantic, filters, route.group_mode, corpus=corpus)
        results.append({
            "semantic": group.get("semantic") or "",
            "filters": group.get("filters") or [],
            "records": records,
        })
    return results


def metadata_scope_records(
    settings: Settings,
    route: RouteDecision,
    *,
    corpus: "CorpusContext | None" = None,
) -> list[dict[str, Any]]:
    """执行 metadata scope 查询，返回扁平去重后的 records。"""
    if route.group_mode == "or":
        records = [
            record
            for group in route.paper_groups
            for record in records_for_scope(
                settings,
                combined_semantic(route.paper_semantic, group.get("semantic") or ""),
                [*route.filters, *(group.get("filters") or [])],
                route.group_mode,
                corpus=corpus,
            )
        ]
        return dedupe_paper_records(records)
    return records_for_scope(settings, route.paper_semantic, route.filters, route.group_mode, corpus=corpus)
