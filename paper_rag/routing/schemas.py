"""正文检索链路的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RouteIntent(str, Enum):
    """正文链路的唯一顶层意图。"""

    RETRIEVE = "retrieve"


class RetrieveTask(str, Enum):
    """JEV 在正文检索链路内细分的任务。"""

    FACT = "fact"
    REASON = "reason"
    SUMMARY = "summary"
    COMPARISON = "comparison"


@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrieveDecision:
    route_intent: RouteIntent
    task: RetrieveTask
    provider: str = "rules"
    fallback_used: bool = False
    confidence: float | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_intent": self.route_intent.value,
            "task": self.task.value,
            "provider": self.provider,
            "fallback_used": self.fallback_used,
            "confidence": self.confidence,
        }


__all__ = ["RetrieveDecision", "RetrieveRequest", "RetrieveTask", "RouteIntent"]
