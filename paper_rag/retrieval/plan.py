"""检索 plan 的顶层薄编排：只分路由，再交给各 domain 执行。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.routes.common.errors import PlanParseError
from paper_rag.retrieval.routes.common.parser_client import TopParserClient
from paper_rag.retrieval.routes.content.planner import plan_body
from paper_rag.retrieval.routes.content.router import build_content_decision
from paper_rag.retrieval.routes.metadata.planner import plan_metadata
from paper_rag.retrieval.routes.metadata.router import build_metadata_decision
from paper_rag.retrieval.routes.reference.planner import plan_reference
from paper_rag.retrieval.routes.reference.router import build_reference_decision
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.timing import Timings, attach_timings


def run_plan(
    settings: Settings,
    query: str,
    *,
    debug: bool = False,
    top_parser=None,
    corpus: CorpusContext | None = None,
    timings: Timings | None = None,
) -> dict[str, Any]:
    """执行一次完整 plan：top router -> domain router -> domain planner。"""
    warnings: list[str] = []
    corpus = corpus or CorpusContext(settings)
    timings = timings or Timings(debug)
    with timings.measure("top_parser"):
        route = build_plan_route(settings, query, warnings, top_parser=top_parser)
    if route.route == "metadata":
        with timings.measure("domain_parser"):
            decision = build_metadata_decision(settings, route, warnings, corpus=corpus)
        with timings.measure("scope"):
            evidence = plan_metadata(settings, decision, warnings, debug=debug, corpus=corpus)
        return attach_timings(evidence, timings)
    if route.route == "reference":
        with timings.measure("domain_parser"):
            decision = build_reference_decision(settings, route, warnings, corpus=corpus)
        with timings.measure("scope"):
            evidence = plan_reference(settings, decision, warnings, debug=debug, corpus=corpus)
        return attach_timings(evidence, timings)
    if route.route == "content":
        with timings.measure("domain_parser"):
            decision = build_content_decision(settings, route, warnings, corpus=corpus)
        evidence = plan_body(settings, decision, warnings, debug=debug, corpus=corpus, timings=timings)
        return attach_timings(evidence, timings)
    return attach_timings(unclear_plan(query, route, warnings, debug=debug), timings)


def build_plan_route(
    settings: Settings,
    query: str,
    warnings: list[str],
    *,
    top_parser=None,
) -> RouteDecision:
    """调用 top parser，把 query 分类成 metadata/reference/content/unclear。"""
    try:
        parser = top_parser or TopParserClient.from_settings(settings)
        if not hasattr(parser, "parse_top"):
            raise PlanParseError("top_parser 必须提供 parse_top(query)")
        parser_result = parser.parse_top(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"top parser 解析失败：{exc}")
        return RouteDecision(
            route="unclear",
            query=query,
            parser_result=None,
            parse_status="parse_failed",
            parser_error=str(exc),
        )

    route_name = parser_result["router"]
    return RouteDecision(
        route=route_name,
        query=query,
        parser_result=parser_result,
        parse_status="ok",
    )


def unclear_plan(
    query: str,
    route: RouteDecision,
    warnings: list[str],
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """把 top 层失败或 unclear 结果包装成统一 evidence 骨架。"""
    evidence: dict[str, Any] = {
        "query": query,
        "route": "unclear",
        "status": "parse_failed" if route.parse_status == "parse_failed" else "unclear",
        "results": {},
        "warnings": warnings or ["top parser 返回了不明确的路由"],
    }
    if route.parser_error:
        evidence["parser_error"] = route.parser_error
    if debug:
        evidence["debug"] = {
            "parser_result": route.parser_result,
            "parse_status": route.parse_status,
            "parser_error": route.parser_error,
        }
    return evidence
