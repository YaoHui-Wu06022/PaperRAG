"""OpenAI-compatible parser client，供 top/metadata/reference/content 共用。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from paper_rag.config import Settings
from paper_rag.retrieval.routes.common.errors import PlanParseError


@dataclass(frozen=True)
class PlanParserClient:
    """封装 chat/completions 请求和 JSON 内容提取。"""

    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: int = 30

    @classmethod
    def from_settings(cls, settings: Settings) -> "PlanParserClient":
        """从 Settings 构造 parser client。"""
        return cls(
            base_url=settings.plan_parser_base_url,
            api_key=settings.plan_parser_api_key,
            model=settings.plan_parser_model,
            timeout_seconds=settings.plan_parser_timeout_seconds,
        )

    def complete_json(self, system_prompt: str, query: str) -> str:
        """发送一次 parser 请求，优先要求 JSON object 格式返回。"""
        if not self.base_url or not self.api_key or not self.model:
            raise PlanParseError("PLAN_PARSER_BASE_URL, PLAN_PARSER_API_KEY or PLAN_PARSER_MODEL is missing")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            data = self.chat_completion(payload)
        except PlanParseError as exc:
            # 部分 OpenAI-compatible 服务不支持 response_format，失败时退回普通 JSON prompt。
            if "response_format" not in str(exc):
                raise
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            data = self.chat_completion(fallback_payload)
        return chat_completion_content(data)

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行底层 HTTP chat/completions 请求。"""
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Paper_RAG/0.1 plan-parser",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise PlanParseError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PlanParseError(str(exc)) from exc
        return data


def chat_completion_content(data: dict[str, Any]) -> str:
    """从 chat/completions 响应中取第一条 message.content。"""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PlanParseError("Plan parser response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise PlanParseError("Plan parser choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise PlanParseError("Plan parser response missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise PlanParseError("Plan parser response missing content")
    return content.strip()
