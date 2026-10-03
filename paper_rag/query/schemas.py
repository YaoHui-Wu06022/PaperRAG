"""查询意图和查询请求的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    """Paper RAG 首期支持的固定查询意图。"""

    PAPER_DISCOVERY = "paper_discovery"
    PAPER_SUMMARY = "paper_summary"
    PAPER_COMPARISON = "paper_comparison"
    PAPER_CONTENT = "paper_content"
    METADATA_LOOKUP = "metadata_lookup"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class QueryRequest:
    """提交给查询服务的自然语言请求。"""

    query: str
    paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryIntentDecision:
    """Jev 或本地规则返回的意图判断。"""

    intent: QueryIntent
    clarification_probability: float | None = None
    needs_clarification: bool = False
    provider: str = "rules"
    fallback_used: bool = False
    error: str | None = None
    candidate_handlers: tuple[str, ...] = field(default_factory=tuple)
    matched_intents: tuple[QueryIntent, ...] = field(default_factory=tuple)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "clarification_probability": self.clarification_probability,
            "needs_clarification": self.needs_clarification,
            "provider": self.provider,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "candidate_handlers": list(self.candidate_handlers),
            "matched_intents": [intent.value for intent in self.matched_intents],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class QueryResult:
    """只读查询的结构化结果。"""

    decision: QueryIntentDecision
    query: str
    items: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.decision.to_dict(),
            "items": self.items,
            "context": self.context,
            "message": self.message,
            "capabilities": self.capabilities,
            "read_only": True,
        }
