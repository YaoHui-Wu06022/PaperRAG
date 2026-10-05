"""正文检索链路的 JEV 分类入口。"""

from __future__ import annotations

from paper_rag.config import Settings
from paper_rag.routing.jev import JevClient, JevError
from paper_rag.routing.rules import classify_by_rules
from paper_rag.routing.schemas import RetrieveDecision, RetrieveRequest


def classify_retrieve(settings: Settings, query: str, paper_ids: list[str] | None = None, *, client: JevClient | None = None) -> RetrieveDecision:
    """只有调用方已选择正文检索后，才执行 JEV 或本地正文任务分类。"""

    request = RetrieveRequest(str(query or "").strip(), tuple(str(value) for value in (paper_ids or []) if str(value)))
    if not request.query:
        return classify_by_rules(request)
    try:
        return (client or JevClient(settings)).classify(request)
    except JevError as exc:
        result = classify_by_rules(request)
        return RetrieveDecision(result.route_intent, result.task, provider="rules", fallback_used=True, confidence=result.confidence, warning=str(exc))


__all__ = ["classify_retrieve"]
