"""正文链路使用的 Jev System One 分类客户端。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

from paper_rag.config import Settings
from paper_rag.routing.schemas import RetrieveDecision, RetrieveRequest, RetrieveTask, RouteIntent


class JevError(RuntimeError):
    """Jev 请求或响应解析失败。"""


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class JevClient:
    """只在已经进入正文检索链路后判断正文任务。"""

    def __init__(self, settings: Settings, *, opener: Callable[..., Any] | None = None, sleeper: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper

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
        request = urllib.request.Request(
            self.settings.jev_base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.jev_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": uuid.uuid4().hex,
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.jev_retry_count + 1):
            try:
                with self.opener(request, timeout=self.settings.jev_timeout_seconds) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    raw = response.read().decode("utf-8", errors="replace")
                    if status < 200 or status >= 300:
                        if status in _RETRYABLE_STATUS:
                            raise OSError(f"HTTP {status}")
                        raise JevError(f"Jev 请求失败：HTTP {status}")
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise JevError("Jev 返回格式不是 JSON 对象")
                    return value
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.settings.jev_retry_count:
                self.sleeper(2**attempt)
        raise JevError(f"Jev 请求重试失败：{last_error}") from last_error


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
