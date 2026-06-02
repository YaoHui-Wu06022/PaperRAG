"""content router：调用 parser，并解析正文检索前的论文范围。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paper_rag.config import Settings
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.common.parser_client import ContentParserClient
from paper_rag.retrieval.routes.common.router import build_paper_scope_decision, apply_paper_scope_year_filters

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def build_content_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
    corpus: "CorpusContext | None" = None,
) -> RouteDecision:
    """把 content parser result 归一化成 RouteDecision。"""
    enriched = build_paper_scope_decision(
        settings,
        decision,
        warnings,
        parser_factory=ContentParserClient.from_settings,
        parser_method="parse_content",
        warning_prefix="content",
        missing_parser_message="plan_parser 必须提供 parse_content(query)",
        plan_parser=plan_parser,
        corpus=corpus,
    )
    if enriched.parse_status == "parse_failed":
        return enriched
    return apply_paper_scope_year_filters(settings, enriched, warnings, corpus=corpus)
