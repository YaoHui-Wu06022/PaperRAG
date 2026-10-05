"""使用 OpenAI 兼容 Chat Completions 提取 BM25 核心词和完整短语。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from paper_rag.config import Settings


class QueryRewriterError(RuntimeError):
    """查询改写请求或响应解析失败。"""


@dataclass(frozen=True)
class QueryRewrite:
    """Query Rewriter 允许返回的最小结构。"""

    core_terms: tuple[str, ...]


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class QueryRewriterClient:
    """调用 OpenAI 兼容 Chat Completions，使用独立配置和响应契约。"""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper

    def rewrite(self, query: str) -> QueryRewrite:
        if not self.settings.query_rewriter_enabled:
            raise QueryRewriterError("Query Rewriter 已禁用")
        if not self.settings.query_rewriter_api_key:
            raise QueryRewriterError("QUERY_REWRITER_API_KEY 未配置")
        payload = {
            "model": self.settings.query_rewriter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是论文库 BM25 查询改写器。只返回 JSON 对象，且只能包含一个字段 core_terms。"
                        "core_terms 是需要保持完整语义的检索短语"
                    ),
                },
                {"role": "user", "content": str(query or "")[:7000]},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = self._request(payload)
        return _parse_rewrite(_response_content(response))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            _chat_completions_url(self.settings.query_rewriter_base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.query_rewriter_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": uuid.uuid4().hex,
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.query_rewriter_retry_count + 1):
            try:
                with self.opener(request, timeout=self.settings.query_rewriter_timeout_seconds) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    raw = response.read().decode("utf-8", errors="replace")
                    if status < 200 or status >= 300:
                        if status in _RETRYABLE_STATUS:
                            raise OSError(f"HTTP {status}")
                        raise QueryRewriterError(f"Query Rewriter 请求失败：HTTP {status}")
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise QueryRewriterError("Query Rewriter 返回格式不是 JSON 对象")
                    return value
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.settings.query_rewriter_retry_count:
                self.sleeper(2**attempt)
        raise QueryRewriterError(f"Query Rewriter 请求重试失败：{last_error}") from last_error


def _response_content(payload: dict[str, Any]) -> dict[str, Any]:
    """提取 Chat Completions 的 message.content 并解析 JSON。"""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise QueryRewriterError("Chat Completions 缺少 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise QueryRewriterError("Chat Completions 缺少 message.content")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.casefold().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QueryRewriterError("Query Rewriter 返回内容不是 JSON") from exc
    if not isinstance(value, dict):
        raise QueryRewriterError("Query Rewriter JSON 不是对象")
    return value


def _chat_completions_url(base_url: str) -> str:
    value = str(base_url or "").rstrip("/")
    return value if value.endswith("/chat/completions") else f"{value}/chat/completions"


def _parse_rewrite(payload: dict[str, Any]) -> QueryRewrite:
    core_terms = _parse_terms(_find(payload, "core_terms"), "core_terms")
    if not core_terms:
        raise QueryRewriterError("Query Rewriter 没有返回核心检索短语")
    return QueryRewrite(core_terms)


def _parse_terms(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, dict):
        value = next((value[key] for key in ("value", "items", "list", "result") if key in value), None)
    if not isinstance(value, list):
        raise QueryRewriterError(f"Query Rewriter 的 {name} 不是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise QueryRewriterError(f"Query Rewriter 的 {name} 包含非法项")
        text = " ".join(item.split())
        key = text.casefold()
        if key not in seen:
            result.append(text)
            seen.add(key)
    return tuple(result)


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


__all__ = ["QueryRewrite", "QueryRewriterClient", "QueryRewriterError"]
