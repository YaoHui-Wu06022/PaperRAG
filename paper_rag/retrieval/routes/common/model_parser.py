"""基于统一大模型抽取结果的三条路由 parser。"""

from __future__ import annotations

from typing import Any

from paper_rag.extraction import ExtractionError, QueryExtraction
from paper_rag.extraction import deepseek as extraction_service
from paper_rag.retrieval.routes.common.errors import PlanParseError
from paper_rag.retrieval.routes.common.local_parser import LocalRouteParser
from paper_rag.retrieval.routes.content.schema import validate_content_parse
from paper_rag.retrieval.routes.metadata.schema import validate_metadata_parse
from paper_rag.retrieval.routes.reference.schema import validate_reference_parse


class ModelQueryParser:
    MIN_EXTRACTION_CONFIDENCE = 0.60
    """让模型理解 query，本地只把结果转换为安全的 planner payload。"""

    def __init__(self, settings, query: str, decision):
        self.settings = settings
        self.query = str(query or "").strip()
        self.decision = decision
        self._result: QueryExtraction | None = None
        self._error: Exception | None = None
        self._extraction_debug: dict[str, Any] | None = None

    def _extract(self) -> QueryExtraction:
        if self._result is not None:
            return self._result
        try:
            result = extraction_service.extract_query_with_cache(
                self.settings,
                self.query,
                route=self.decision.route,
                intent=self._intent,
                complexity=self.decision.complexity,
            )
            if result.confidence < self.MIN_EXTRACTION_CONFIDENCE:
                raise ValueError("extraction_low_confidence")
            self._result = result
            self._extraction_debug = self._debug_payload(result)
            return result
        except (ExtractionError, OSError, ValueError) as exc:
            self._error = exc
            code = "extraction_low_confidence" if "confidence" in str(exc) else "extraction_failed"
            self._extraction_debug = {
                "backend": "deepseek",
                "model": getattr(self.settings, "deepseek_model", None),
                "cache_hit": False,
                "confidence": None,
                "call_count": 1,
                "error": code,
            }
            raise PlanParseError(code) from exc

    def parse_metadata(self, query: str) -> dict[str, Any]:
        try:
            extraction = self._extract()
        except PlanParseError:
            return self._fallback(query, "metadata")
        payload = self._metadata_payload(extraction)
        result = validate_metadata_parse(payload, query)
        return self._attach_extraction(result, extraction)

    def parse_reference(self, query: str) -> dict[str, Any]:
        try:
            extraction = self._extract()
        except PlanParseError:
            return self._fallback(query, "reference")
        payload = self._reference_payload(extraction)
        result = validate_reference_parse(payload, query)
        return self._attach_extraction(result, extraction)

    def parse_content(self, query: str) -> dict[str, Any]:
        extraction = self._extract()
        if extraction.scope_required and not (extraction.paper_mentions or extraction.paper_groups):
            raise PlanParseError("alias_unresolved")
        payload = {
            "intent": self._intent,
            "paper_semantic": "",
            "filters": self._filters(extraction),
            "paper_groups": self._groups(extraction),
            "group_mode": self._group_mode(extraction),
            "content_objects": extraction.content_objects,
            "compare_objects": extraction.compare_objects,
        }
        result = validate_content_parse(payload, query)
        return self._attach_extraction(result, extraction)

    def _metadata_payload(self, extraction: QueryExtraction) -> dict[str, Any]:
        intent = self._intent
        fields = extraction.metadata_fields
        if intent == "list" and not fields:
            fields = ["title"]
        if intent == "lookup" and not fields:
            raise PlanParseError("metadata_fields_missing")
        return {
            "intent": intent,
            "return_fields": fields,
            "paper_semantic": "",
            "filters": self._filters(extraction),
            "paper_groups": self._groups(extraction),
            "group_mode": self._group_mode(extraction),
        }

    @property
    def _intent(self) -> str | None:
        return getattr(self.decision, "intent", None) or getattr(
            self.decision,
            f"{self.decision.route}_intent",
            None,
        )

    def _reference_payload(self, extraction: QueryExtraction) -> dict[str, Any]:
        side = extraction.reference_side or self.decision.return_side
        generic_mentions = extraction.paper_mentions or extraction.reference_mentions
        if side in {"source", "object"}:
            source_mentions = extraction.source_paper_mentions or (
                generic_mentions if side == "object" else []
            )
            object_mentions = extraction.object_paper_mentions or (
                generic_mentions if side == "source" else []
            )
        else:
            source_mentions, object_mentions = [], []
        source_filters = self._filters(extraction, paper_mentions=source_mentions)
        object_filters = self._filters(extraction, paper_mentions=object_mentions)
        source_groups = self._groups(extraction, paper_mentions=source_mentions)
        object_groups = self._groups(extraction, paper_mentions=object_mentions)
        return {
            "intent": self._intent,
            "return_side": side,
            "source_semantic": "",
            "source_filters": source_filters,
            "source_groups": source_groups,
            "source_mode": self._group_mode(extraction, source_groups),
            "object_semantic": "",
            "object_filters": object_filters,
            "object_groups": object_groups,
            "object_mode": self._group_mode(extraction, object_groups),
        }

    def _filters(
        self,
        extraction: QueryExtraction,
        *,
        paper_mentions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [
            {"field": "paper", "op": "=", "value": mention, "negated": False}
            for mention in (paper_mentions if paper_mentions is not None else extraction.paper_mentions)
        ]
        filters.extend(
            {"field": "author", "op": "contains", "value": mention, "negated": False}
            for mention in extraction.author_mentions
        )
        filters.extend(
            {"field": "year", "op": "interval", "value": interval, "negated": False}
            for interval in extraction.year_intervals
        )
        if extraction.venue_mentions:
            if len(extraction.venue_mentions) == 1:
                filters.append({"field": "venue", "op": "=", "value": extraction.venue_mentions[0], "negated": False})
            else:
                filters.append({"field": "venue", "op": "in", "value": extraction.venue_mentions, "negated": False})
        return filters

    def _groups(
        self,
        extraction: QueryExtraction,
        *,
        paper_mentions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not extraction.paper_groups:
            return []
        allowed = set(paper_mentions) if paper_mentions is not None else None
        groups: list[dict[str, Any]] = []
        for group in extraction.paper_groups:
            mentions = [mention for mention in group if allowed is None or mention in allowed]
            if mentions:
                groups.append({
                    "semantic": "",
                    "filters": [
                        {"field": "paper", "op": "=", "value": mention, "negated": False}
                        for mention in mentions
                    ],
                })
        return groups

    @staticmethod
    def _group_mode(extraction: QueryExtraction, groups: list[dict[str, Any]] | None = None) -> str:
        groups = groups if groups is not None else extraction.paper_groups
        if not groups:
            return "single"
        return extraction.group_mode if extraction.group_mode != "single" else "per"

    def _attach_extraction(self, payload: dict[str, Any], extraction: QueryExtraction) -> dict[str, Any]:
        payload["extraction_debug"] = self._debug_payload(extraction)
        return payload

    def _debug_payload(self, extraction: QueryExtraction) -> dict[str, Any]:
        return {
            "backend": "deepseek",
            "model": getattr(self.settings, "deepseek_model", None),
            "cache_hit": extraction.cache_hit,
            "confidence": extraction.confidence,
            "call_count": 0 if extraction.cache_hit else 1,
            "result": extraction.to_payload(),
        }

    def _fallback(self, query: str, route: str) -> dict[str, Any]:
        """模型不可用时只给 metadata/reference 使用最小规则兜底。"""
        fallback = LocalRouteParser(query, self.decision)
        if route == "metadata":
            result = fallback.parse_metadata(query)
        else:
            result = fallback.parse_reference(query)
        result["extraction_debug"] = {
            "backend": "rules",
            "model": None,
            "cache_hit": False,
            "confidence": None,
            "call_count": 1,
            "error": type(self._error).__name__ if self._error else "extraction_failed",
        }
        return result
