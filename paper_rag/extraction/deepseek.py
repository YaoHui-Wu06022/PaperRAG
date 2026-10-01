"""使用 DeepSeek 对用户 query 做一次结构化语义抽取。"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_rag.extraction.schema import ExtractionSchemaError, QueryExtraction


EXTRACTION_PROMPT_VERSION = "v2"
EXTRACTION_SYSTEM_PROMPT = """
你是 Paper_RAG 的查询语义抽取器，不是路由器，也不是最终回答器。
Jev 已经给出了 route、intent 和 complexity，你只从用户 query 中抽取明确出现的论文实体、作者、年份、venue、正文对象和引用方向。
不要根据论文常识补充 query 中没有出现的实体，不要生成 route、filters、最终检索计划或 final_answer。
输出且只能输出 JSON object，字段必须完整：
paper_mentions、paper_groups、author_mentions、year_intervals、venue_mentions、
source_paper_mentions、object_paper_mentions、content_objects、compare_objects、
reference_mentions、reference_side、group_mode、metadata_fields、scope_required、confidence。
paper_groups 是二维字符串列表；year_intervals 使用 [起始, 结束]，无界使用 "inf" 或 "-inf"。
scope_required 表示 query 是否明确要求限定某篇或某几篇论文；没有明确论文时为 false。
metadata_fields 只能使用 author、year、venue、title。
reference_side 只能使用 source、object 或 null。
无法确定的字段返回空列表或 null，并降低 confidence。
""".strip()


class ExtractionError(RuntimeError):
    """结构化抽取请求或响应失败。"""


@dataclass(frozen=True)
class DeepSeekExtractionClient:
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: int = 60
    temperature: float = 0.0
    min_confidence: float = 0.60

    @classmethod
    def from_settings(cls, settings) -> "DeepSeekExtractionClient":
        if not getattr(settings, "deepseek_base_url", "") or not getattr(settings, "deepseek_api_key", None):
            raise ExtractionError("DeepSeek 抽取器未配置")
        return cls(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            timeout_seconds=settings.deepseek_timeout_seconds,
            temperature=0.0,
            min_confidence=0.60,
        )

    def extract_query(
        self,
        query: str,
        *,
        route: str,
        intent: str | None,
        complexity: int | None,
        partial: dict[str, Any] | None = None,
    ) -> QueryExtraction:
        if not self.api_key or not self.model:
            raise ExtractionError("缺少 DeepSeek 抽取配置")
        user_payload = {
            "query": query,
            "route": route,
            "intent": intent,
            "complexity": complexity,
            "partial_parse": partial or {},
        }
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }
        data = self._chat_completion(request_payload)
        payload = _parse_json_object(_response_content(data))
        try:
            result = QueryExtraction.from_payload(payload)
        except ExtractionSchemaError as exc:
            raise ExtractionError(str(exc)) from exc
        if result.confidence < self.min_confidence:
            raise ExtractionError("DeepSeek 抽取置信度不足")
        return result

    def extract(
        self,
        query: str,
        *,
        route: str,
        parser_error: str = "",
        partial: dict[str, Any] | None = None,
        intent: str | None = None,
        complexity: int | None = None,
    ) -> QueryExtraction:
        """保留统一入口，parser_error 只作为局部上下文传递给模型。"""
        partial_payload = dict(partial or {})
        if parser_error:
            partial_payload["parser_error"] = parser_error
        return self.extract_query(
            query,
            route=route,
            intent=intent,
            complexity=complexity,
            partial=partial_payload,
        )

    def _chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Paper_RAG/0.1 query-extractor",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise ExtractionError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ExtractionError(str(exc)) from exc


def extract_query_with_cache(
    settings,
    query: str,
    *,
    route: str,
    intent: str | None,
    complexity: int | None,
    partial: dict[str, Any] | None = None,
) -> QueryExtraction:
    """执行一次 query 抽取，并使用版本化 JSONL 缓存。"""
    path = getattr(settings, "extraction_cache_path", None)
    key = _cache_key(query, route, intent, complexity, partial)
    cached = _load_cache(path, key)
    if cached is not None:
        return QueryExtraction.from_payload(cached).with_cache_hit(True)
    result = DeepSeekExtractionClient.from_settings(settings).extract_query(
        query,
        route=route,
        intent=intent,
        complexity=complexity,
        partial=partial,
    )
    _save_cache(path, key, result)
    return result


def extract_with_cache(
    settings,
    query: str,
    *,
    route: str,
    parser_error: str = "",
    partial: dict[str, Any] | None = None,
    intent: str | None = None,
    complexity: int | None = None,
) -> QueryExtraction:
    """旧调用点使用的统一包装，实际仍走 v2 query 抽取协议。"""
    payload = dict(partial or {})
    if parser_error:
        payload["parser_error"] = parser_error
    return extract_query_with_cache(
        settings,
        query,
        route=route,
        intent=intent,
        complexity=complexity,
        partial=payload,
    )


def _cache_key(
    query: str,
    route: str,
    intent: str | None,
    complexity: int | None,
    partial: dict[str, Any] | None,
) -> str:
    payload = {
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "query": query,
        "route": route,
        "intent": intent,
        "complexity": complexity,
        "partial": partial or {},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cache(path: Path | None, key: str) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("key") == key and isinstance(item.get("result"), dict):
                return item["result"]
    except (OSError, ValueError, TypeError):
        return None
    return None


def _save_cache(path: Path | None, key: str, result: QueryExtraction) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "result": result.to_payload(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        return


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError("DeepSeek 抽取结果不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise ExtractionError("DeepSeek 抽取结果必须是 JSON object")
    return payload


def _response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ExtractionError("DeepSeek 抽取响应缺少 choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ExtractionError("DeepSeek 抽取响应缺少 message.content")
    return message["content"].strip()
