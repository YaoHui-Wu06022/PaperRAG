"""检索 plan 的顶层薄编排：只分路由，再交给各 domain 执行。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.routes.common.jev_client import JevDecisionClient, JevError, fallback_decision
from paper_rag.retrieval.routes.common.model_parser import ModelQueryParser
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
    corpus: CorpusContext | None = None,
    timings: Timings | None = None,
) -> dict[str, Any]:
    """执行一次完整 plan：Jev decision -> 结构化抽取 -> 本地校验 -> domain planner。"""
    warnings: list[str] = []
    corpus = corpus or CorpusContext(settings)
    timings = timings or Timings(debug)
    with timings.measure("jev_decision"):
        route = build_plan_route(settings, query, warnings)
    domain_parser = ModelQueryParser(settings, query, decision_from_route(route))
    if route.route == "metadata":
        with timings.measure("domain_parser"):
            decision = build_metadata_decision(settings, route, warnings, plan_parser=domain_parser, corpus=corpus)
        with timings.measure("scope"):
            evidence = plan_metadata(settings, decision, warnings, debug=debug, corpus=corpus)
        if debug:
            evidence.setdefault("debug", {})["embedding_profile"] = getattr(settings, "embedding_profile", "qwen_v4")
        return attach_timings(evidence, timings)
    if route.route == "reference":
        with timings.measure("domain_parser"):
            decision = build_reference_decision(settings, route, warnings, plan_parser=domain_parser, corpus=corpus)
        with timings.measure("scope"):
            evidence = plan_reference(settings, decision, warnings, debug=debug, corpus=corpus)
        if debug:
            evidence.setdefault("debug", {})["embedding_profile"] = getattr(settings, "embedding_profile", "qwen_v4")
        return attach_timings(evidence, timings)
    if route.route == "content":
        with timings.measure("domain_parser"):
            decision = build_content_decision(settings, route, warnings, plan_parser=domain_parser, corpus=corpus)
        evidence = plan_body(settings, decision, warnings, debug=debug, corpus=corpus, timings=timings)
        if debug:
            evidence.setdefault("debug", {})["embedding_profile"] = getattr(settings, "embedding_profile", "qwen_v4")
        return attach_timings(evidence, timings)
    return attach_timings(unclear_plan(query, route, warnings, debug=debug), timings)


def build_plan_route(
    settings: Settings,
    query: str,
    warnings: list[str],
) -> RouteDecision:
    """调用 Jev 一次性判断有限集合路由，失败时保守 rules fallback。"""
    try:
        decision = JevDecisionClient.from_settings(settings).decide(query)
    except (JevError, OSError, ValueError) as exc:
        fallback = fallback_decision(query, str(exc))
        reason = getattr(exc, "code", "jev_unavailable")
        if reason not in {"jev_low_confidence", "jev_unavailable", "jev_invalid_response"}:
            reason = "jev_unavailable"
        # Jev 是正常路由的唯一来源。故障时只允许 metadata/reference
        # 使用保守规则；content 不猜测 route，直接返回 unclear。
        fallback_route = fallback.route if fallback.route in {"metadata", "reference"} else "unclear"
        if fallback_route == "unclear":
            warnings.append(f"{reason}：Jev 不可用，content 路由停止检索：{exc}")
        else:
            warnings.append(f"{reason}：Jev 不可用，使用 {fallback_route} rules fallback：{exc}")
        return RouteDecision(
            route=fallback_route,
            query=query,
            parser_result={"router": fallback_route, "backend": "rules", "jev_error": reason},
            parse_status="ok" if fallback_route != "unclear" else "unclear",
            decision_backend="rules",
            decision_fallback=True,
            decision_fallback_reason=reason,
            decision_policy_version=fallback.policy_version,
            needs_synthesis=fallback.needs_synthesis if fallback_route != "unclear" else False,
            complexity=fallback.complexity if fallback_route != "unclear" else None,
            intent=getattr(fallback, f"{fallback_route}_intent", None) if fallback_route in {"metadata", "reference"} else None,
            return_side=fallback.reference_side if fallback_route == "reference" else None,
        )
    return RouteDecision(
        route=decision.route,
        query=query,
        parser_result={"router": decision.route, "backend": decision.backend},
        parse_status="ok",
        decision_backend=decision.backend,
        decision_confidence=decision.confidence,
        decision_probabilities=decision.probabilities,
        decision_fallback=decision.fallback,
        decision_policy_version=decision.policy_version,
        needs_synthesis=decision.needs_synthesis,
        complexity=decision.complexity,
        intent=getattr(decision, f"{decision.route}_intent", None) if decision.route in {"metadata", "reference", "content"} else None,
        return_side=decision.reference_side,
    )


def decision_from_route(route: RouteDecision):
    """把 route 元数据转换成结构化抽取器使用的轻量 JevDecision。"""
    from paper_rag.retrieval.routes.common.jev_client import JevDecision

    return JevDecision(
        route=route.route,
        metadata_intent=route.intent if route.route == "metadata" else None,
        reference_intent=route.intent if route.route == "reference" else None,
        reference_side=route.return_side,
        content_intent=route.intent if route.route == "content" else None,
        needs_synthesis=route.needs_synthesis,
        complexity=route.complexity or 1,
        confidence=route.decision_confidence,
        probabilities=route.decision_probabilities,
        backend=route.decision_backend,
        fallback=route.decision_fallback,
        policy_version=route.decision_policy_version,
    )


def unclear_plan(
    query: str,
    route: RouteDecision,
    warnings: list[str],
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """把 Jev/rules 无法判定的结果包装成统一 evidence 骨架。"""
    evidence: dict[str, Any] = {
        "query": query,
        "route": "unclear",
        "status": "parse_failed" if route.parse_status == "parse_failed" else "unclear",
        "results": {},
        "warnings": warnings or ["top parser 返回了不明确的路由"],
        "decision": {
            "backend": route.decision_backend,
            "route": route.route,
            "confidence": route.decision_confidence,
            "fallback": route.decision_fallback,
            "fallback_reason": route.decision_fallback_reason,
            "policy_version": route.decision_policy_version,
        },
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
