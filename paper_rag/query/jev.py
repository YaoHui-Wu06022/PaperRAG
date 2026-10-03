"""Jev System One 查询意图客户端。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

from paper_rag.config import Settings
from paper_rag.query.schemas import QueryIntent, QueryIntentDecision


class JevError(RuntimeError):
    """Jev 请求或响应解析失败。"""


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class JevResponse:
    """Jev 原始响应中提取出的分类结果。"""

    intent: QueryIntent
    clarification_probability: float | None
    needs_clarification: bool


class JevClient:
    """调用 Jev System One 的最小 HTTP 客户端。"""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper

    def classify(self, query: str, paper_ids: tuple[str, ...] = ()) -> QueryIntentDecision:
        """提交 query 并解析固定意图集合。"""

        if not self.settings.jev_enabled:
            raise JevError("Jev 路由已禁用")
        if not self.settings.jev_api_key:
            raise JevError("JEV_API_KEY 未配置")
        state = _build_state(query, paper_ids)
        payload = {
            "model": self.settings.jev_model,
            "state": state,
            "questions": _questions(),
        }
        raw = self._request(payload)
        parsed = _parse_response(raw, self.settings.jev_route_probability_threshold)
        return QueryIntentDecision(
            intent=parsed.intent,
            clarification_probability=parsed.clarification_probability,
            needs_clarification=parsed.needs_clarification,
            provider="jev",
            candidate_handlers=(parsed.intent.value,),
            matched_intents=(parsed.intent,),
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.jev_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": uuid.uuid4().hex,
        }
        request = urllib.request.Request(
            self.settings.jev_base_url,
            data=body,
            method="POST",
            headers=headers,
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.jev_retry_count + 1):
            try:
                with self.opener(request, timeout=self.settings.jev_timeout_seconds) as response:
                    status = _response_status(response)
                    raw = response.read().decode("utf-8", errors="replace")
                    if status < 200 or status >= 300:
                        if status in _RETRYABLE_STATUS:
                            raise _RetryableJevError(f"HTTP {status}")
                        raise JevError(f"Jev 请求失败：HTTP {status}")
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise JevError("Jev 返回的 JSON 无法解析") from exc
                    if not isinstance(value, dict):
                        raise JevError("Jev 返回格式不是 JSON 对象")
                    return value
            except _RetryableJevError as exc:
                last_error = exc
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS:
                    raise JevError(f"Jev 请求失败：HTTP {exc.code}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self.settings.jev_retry_count:
                self.sleeper(2**attempt)
        raise JevError(f"Jev 请求重试失败：{last_error}") from last_error


class _RetryableJevError(JevError):
    pass


def _build_state(query: str, paper_ids: tuple[str, ...]) -> dict[str, Any]:
    """构造不超过 Jev API 限制的最小状态。"""

    state = {"query": str(query), "paper_ids": list(paper_ids)}
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= 8000:
        return state
    state["query"] = str(query)[:7500]
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > 8000:
        state["query"] = str(query)[:7000]
        state["paper_ids"] = []
    return state


def _questions() -> dict[str, Any]:
    return {
        "intent": {
            "type": "choice",
            "instructions": "只判断用户对论文知识库的 query 意图，不选择 MCP 工具，也不判断下载、解析、任务或索引操作",
            "criteria": {
                "paper_discovery": "查找和筛选论文",
                "paper_summary": "总结论文",
                "paper_comparison": "比较论文、方法或实验结果",
                "paper_content": "回答论文正文中的具体问题",
                "metadata_lookup": "查询作者、日期、分类或版本",
                "clarify": "信息不足，需要用户补充",
                "unsupported": "当前系统无法处理",
            },
        },
        "needs_clarification": {
            "type": "noul",
            "instructions": "判断当前 query 是否需要先向用户澄清",
        },
    }


def _parse_response(payload: dict[str, Any], probability_threshold: float = 0.65) -> JevResponse:
    intent_value = _find_choice(payload, "intent")
    try:
        intent = QueryIntent(str(intent_value))
    except (TypeError, ValueError) as exc:
        raise JevError("Jev 返回了未知查询意图") from exc
    clarification = _find_question_value(payload, "needs_clarification")
    probability = _number_from(clarification)
    needs = bool(clarification) if isinstance(clarification, bool) else bool(
        probability is not None and probability >= probability_threshold
    )
    return JevResponse(intent, probability, needs)


def _find_choice(payload: Any, name: str) -> Any:
    value = _find_question_value(payload, name)
    if isinstance(value, dict):
        for key in ("choice", "value", "label", "answer", "result"):
            if key in value:
                return value[key]
    if value is not None:
        return value
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, dict) and name in choices:
        return choices[name]
    raise JevError("Jev 返回缺少 intent")


def _find_question_value(payload: Any, name: str) -> Any:
    if isinstance(payload, dict):
        if name in payload:
            return payload[name]
        for value in payload.values():
            found = _find_question_value(value, name)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_question_value(value, name)
            if found is not None:
                return found
    return None


def _number_from(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("probability", "prob", "score"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    return None


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    return int(status if status is not None else response.getcode())


__all__ = ["JevClient", "JevError"]
