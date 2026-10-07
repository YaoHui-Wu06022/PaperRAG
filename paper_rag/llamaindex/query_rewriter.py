"""使用 OpenAI 兼容 Chat Completions 提取 BM25 核心词和完整短语。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from paper_rag.config import Settings
from paper_rag.http import HttpRequestError, JsonHttpClient


class QueryRewriterError(RuntimeError):
    """查询改写请求或响应解析失败。"""


@dataclass(frozen=True)
class QueryRewrite:
    """Query Rewriter 允许返回的最小结构。"""

    entities: tuple[str, ...]
    core_terms: tuple[str, ...]


_BASE_PROMPT = (
    "你是论文库查询关键词抽取器。只返回 JSON，且仅包含 entities 和 core_terms 两个字符串数组。"
    "所有返回内容都必须直接从原句中抽取，禁止改写。"
    "先抽取 entities，再抽取 core_terms。"
    "entities 表示‘明确要查询的命名对象’，只包含原句中明确出现的论文名、方法名、模型名、算法名或具有专名性质的技术名称，不包含泛化领域词。"
    "core_terms 表示‘具体要查询什么’，只包含原句中希望检索或回答的核心主题、属性、机制、指标、实验结果或比较维度。"
    "core_terms 应遵循最小充分原则：只保留完成检索所必需的最具体词或短语。"
    "如果删除某个 core_term 后仍然能够准确表达用户要检索的内容，则删除该 core_term。"
    "已通过 filters 表达的年份、作者、分类、状态等条件不要再放入 entities 或 core_terms。"
    "core_terms 不得与 entities 重复。"
    "允许一个数组为空，但不能同时为空。"
)


class QueryRewriterClient:
    """调用 OpenAI 兼容 Chat Completions，使用独立配置和响应契约。"""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings
        self.http = JsonHttpClient(opener=opener, **({"sleeper": sleeper} if sleeper else {}))

    def rewrite(self, query: str, *, purpose: str = "metadata", task: str | None = None, filters: dict[str, Any] | None = None) -> QueryRewrite:
        # 用途和任务仅校验调用上下文，所有查询使用同一版抽取提示词。
        if purpose not in {"metadata", "body"} or (
            purpose == "body" and task not in {"fact", "reason", "summary", "comparison"}
        ):
            raise ValueError("invalid query rewrite purpose or task")
        if not self.settings.query_rewriter_enabled:
            raise QueryRewriterError("Query Rewriter 已禁用")
        if not self.settings.query_rewriter_api_key:
            raise QueryRewriterError("QUERY_REWRITER_API_KEY 未配置")
        payload = {
            "model": self.settings.query_rewriter_model,
            "messages": [
                {
                    "role": "system",
                    "content": _BASE_PROMPT,
                },
                {"role": "user", "content": json.dumps({"query": str(query or "")[:7000], "filters": filters or {}}, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = self._request(payload)
        return _parse_rewrite(_response_content(response))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            value = self.http.post_json(
                _chat_completions_url(self.settings.query_rewriter_base_url),
                payload,
                headers={
                    "Authorization": f"Bearer {self.settings.query_rewriter_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                timeout=self.settings.query_rewriter_timeout_seconds,
                retries=self.settings.query_rewriter_retry_count,
                error_prefix="Query Rewriter ",
            )
        except HttpRequestError as exc:
            raise QueryRewriterError(str(exc)) from exc
        if not isinstance(value, dict):
            raise QueryRewriterError("Query Rewriter 返回格式不是 JSON 对象")
        return value


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
    # 严格区分对象和取证目标，不兼容旧单字段响应。
    if set(payload) != {"entities", "core_terms"}:
        raise QueryRewriterError("Query Rewriter 必须仅返回 entities 和 core_terms 字段")
    entities = _parse_terms(payload.get("entities"), "entities")
    core_terms = _parse_terms(payload.get("core_terms"), "core_terms")
    entity_keys = {value.casefold() for value in entities}
    core_terms = tuple(value for value in core_terms if value.casefold() not in entity_keys)
    if not entities and not core_terms:
        raise QueryRewriterError("Query Rewriter 没有返回核心检索短语")
    return QueryRewrite(entities, core_terms)


def _parse_terms(value: Any, name: str) -> tuple[str, ...]:
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


__all__ = ["QueryRewrite", "QueryRewriterClient", "QueryRewriterError"]
