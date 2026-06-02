"""metadata router：调用 parser，并解析 paper/venue/year scope。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paper_rag.config import Settings
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.common.parser_client import MetadataParserClient
from paper_rag.retrieval.routes.common.router import build_paper_scope_decision, apply_paper_scope_year_filters

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def build_metadata_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
    corpus: "CorpusContext | None" = None,
) -> RouteDecision:
    """把 metadata parser result 归一化成 RouteDecision。"""
    enriched = build_paper_scope_decision(
        settings,
        decision,
        warnings,
        parser_factory=MetadataParserClient.from_settings,
        parser_method="parse_metadata",
        warning_prefix="metadata",
        missing_parser_message="plan_parser 必须提供 parse_metadata(query)",
        include_return_fields=True,
        plan_parser=plan_parser,
        corpus=corpus,
    )
    if enriched.parse_status == "parse_failed":
        return enriched
    return apply_paper_scope_year_filters(
        settings,
        enriched,
        warnings,
        include_return_fields=True,
        corpus=corpus,
    )
