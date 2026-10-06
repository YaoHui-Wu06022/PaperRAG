"""正文链路使用的 Jev System One 分类客户端。"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from paper_rag.config import Settings
from paper_rag.http import HttpRequestError, JsonHttpClient
from paper_rag.routing.schemas import RetrieveDecision, RetrieveRequest, RetrieveTask, RouteIntent


class JevError(RuntimeError):
    """Jev 请求或响应解析失败。"""


class JevClient:
    """只在已经进入正文检索链路后判断正文任务。"""

    def __init__(self, settings: Settings, *, opener: Callable[..., Any] | None = None, sleeper: Callable[[float], None] | None = None):
        self.settings = settings
        self.http = JsonHttpClient(opener=opener, **({"sleeper": sleeper} if sleeper else {}))

    def classify(self, request: RetrieveRequest) -> RetrieveDecision:
        if not self.settings.jev_enabled:
            raise JevError("Jev 路由已禁用")
        if not self.settings.jev_api_key:
            raise JevError("JEV_API_KEY 未配置")
        payload = {
            "model": self.settings.jev_model,
            "state": {"query": request.query[:7000], "paper_ids": list(request.paper_ids), "route_intent": RouteIntent.RETRIEVE.value},
            "questions": {
                "retrieve_task": {
                    "type": "choice",
                    "instructions": "只在已经进入正文检索后判断正文任务，不选择 MCP 工具",
                    "criteria": {task.value: task.value for task in RetrieveTask},
                }
            },
        }
        response = self._request(payload)
        task = _parse_task(response)
        return RetrieveDecision(
            RouteIntent.RETRIEVE,
            task,
            provider="jev",
            confidence=_parse_confidence(response),
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            value = self.http.post_json(
                self.settings.jev_base_url,
                payload,
                headers={
                    "Authorization": f"Bearer {self.settings.jev_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                timeout=self.settings.jev_timeout_seconds,
                retries=self.settings.jev_retry_count,
                error_prefix="Jev ",
            )
        except HttpRequestError as exc:
            raise JevError(str(exc)) from exc
        if not isinstance(value, dict):
            raise JevError("Jev 返回格式不是 JSON 对象")
        return value


def _parse_task(payload: dict[str, Any]) -> RetrieveTask:
    value = _find(payload, "retrieve_task")
    if isinstance(value, dict):
        value = next((value[key] for key in ("choice", "value", "label", "answer", "result") if key in value), value)
    try:
        return RetrieveTask(str(value))
    except (TypeError, ValueError) as exc:
        raise JevError("Jev 返回了未知正文任务") from exc


def _parse_confidence(payload: dict[str, Any]) -> float | None:
    """读取 JEV 返回的置信度，不为缺失字段伪造默认值。"""

    value = _find(payload, "confidence")
    if isinstance(value, dict):
        value = next((value[key] for key in ("value", "score", "probability") if key in value), None)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _find(payload: Any, name: str) -> Any:
    if isinstance(payload, dict):
        if name in payload:
            return payload[name]
        for value in payload.values():
            found = _find(value, name)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find(value, name)
            if found is not None:
                return found
    return None


__all__ = ["JevClient", "JevError"]
