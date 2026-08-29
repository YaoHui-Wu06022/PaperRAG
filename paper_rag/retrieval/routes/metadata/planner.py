"""metadata planner：根据论文 scope 查询 manifest records。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paper_rag.config import Settings
from paper_rag.corpus.scope import append_scope_fallback_warnings, resolve_scope_records
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
    append_scope_fallback_warnings(warnings, route.filters, route.paper_groups)
    records, resolved_groups = resolve_scope_records(
        settings,
        route.paper_semantic,
        route.filters,
        route.paper_groups,
        route.group_mode,
        corpus=corpus,
    )
    if route.group_mode in {"per", "and"}:
        group_results = resolved_groups
        if route.group_mode == "and":
            exists = all(bool(group["records"]) for group in group_results)
    elif route.intent == "exists":
        exists = bool(records)

    count = len(records) if route.intent == "count" else None
    if not records:
        warnings.append("metadata 路由没有匹配到 manifest 记录")
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
