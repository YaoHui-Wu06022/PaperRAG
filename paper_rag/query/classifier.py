"""组合本地规则和 Jev 的查询意图分类器。"""

from __future__ import annotations

from paper_rag.config import Settings
from paper_rag.query.jev import JevClient, JevError
from dataclasses import replace

from paper_rag.query.rules import classify_by_rules, match_rule_intents
from paper_rag.query.schemas import QueryIntent, QueryIntentDecision, QueryRequest


def classify_query(
    settings: Settings,
    request: QueryRequest,
    *,
    client: JevClient | None = None,
) -> QueryIntentDecision:
    """非空 query 优先交给 Jev，失败后使用高精度规则安全回退。"""

    if not str(request.query or "").strip():
        return QueryIntentDecision(
            intent=QueryIntent.CLARIFY,
            needs_clarification=True,
            matched_intents=(QueryIntent.CLARIFY,),
            candidate_handlers=(QueryIntent.CLARIFY.value,),
        )
    try:
        decision = (client or JevClient(settings)).classify(request.query, request.paper_ids)
        if decision.needs_clarification and decision.intent != QueryIntent.CLARIFY:
            return replace(
                decision,
                intent=QueryIntent.CLARIFY,
                candidate_handlers=(decision.intent.value, QueryIntent.CLARIFY.value),
                matched_intents=(decision.intent, QueryIntent.CLARIFY),
            )
        return decision
    except JevError as exc:
        rule_decision = classify_by_rules(request.query)
        if rule_decision is not None:
            return replace(rule_decision, fallback_used=True, error=str(exc))
        matched = match_rule_intents(request.query)
        return QueryIntentDecision(
            intent=QueryIntent.CLARIFY,
            needs_clarification=True,
            provider="rules",
            fallback_used=True,
            error=str(exc),
            matched_intents=matched or (QueryIntent.CLARIFY,),
            candidate_handlers=(QueryIntent.CLARIFY.value,),
        )
