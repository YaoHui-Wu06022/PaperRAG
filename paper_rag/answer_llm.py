"""content route 的 LLM 回答生成器。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings


ANSWER_SYSTEM_PROMPT = """
你是 Paper_RAG 的回答生成器
你只能根据提供的 evidence 回答用户问题，不要编造 evidence 中没有的信息
如果 evidence 为空、路由不明确、解析失败或没有找到相关结果，需要直接说明目前没有足够证据
回答使用中文，保持简洁。必要时引用论文标题、页码、section 或引用边作为依据
"""


class AnswerClientProtocol(Protocol):
    """回答 LLM client 协议，方便测试时注入 fake client。"""

    def complete_answer(self, evidence: dict[str, Any]) -> str:
        """根据 composer evidence 生成最终回答。"""
        ...


@dataclass(frozen=True)
class AnswerComposerClient:
    """OpenAI-compatible answer composer client。"""

    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: int = 60
    temperature: float = 0.2

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnswerComposerClient":
        """从 Settings 构造 answer client。"""
        return cls(
            base_url=settings.answer_base_url,
            api_key=settings.answer_api_key,
            model=settings.answer_model,
            timeout_seconds=settings.answer_timeout_seconds,
            temperature=settings.answer_temperature,
        )

    def complete_answer(self, evidence: dict[str, Any]) -> str:
        """发送回答生成请求。"""
        if not self.base_url or not self.api_key or not self.model:
            raise AnswerError("ANSWER_BASE_URL, ANSWER_API_KEY or ANSWER_MODEL is missing")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": build_answer_user_prompt(evidence)},
            ],
            "temperature": self.temperature,
            "enable_thinking": False,
        }
        data = self.chat_completion(payload)
        return chat_completion_content(data)

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行底层 HTTP chat/completions 请求。"""
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Paper_RAG/0.1 answer-composer",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise AnswerError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AnswerError(str(exc)) from exc
        return data


class AnswerError(RuntimeError):
    """回答生成失败。"""


def build_answer_user_prompt(evidence: dict[str, Any]) -> str:
    """把 composer evidence 放进回答 LLM 的用户消息。"""
    return (
        "请根据下面的 evidence 回答其中的 query\n"
        "如果 evidence 不足，请明确说明不能确定，并指出缺少什么证据\n\n"
        "evidence:\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def chat_completion_content(data: dict[str, Any]) -> str:
    """从 chat/completions 响应中取第一条 message.content。"""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AnswerError("Answer response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise AnswerError("Answer choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise AnswerError("Answer response missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise AnswerError("Answer response missing content")
    return content.strip()
