from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ...config import Settings


METADATA_INTENTS = {"lookup", "list", "count", "unknown"}
METADATA_RETURN_FIELDS = {"author", "year", "venue", "title", None}
METADATA_FILTER_FIELDS = {"author", "year", "venue", "title"}
METADATA_FILTER_OPS = {"=", "in", "contains", "between"}


class PlanParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanParserClient:
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: int = 30

    @classmethod
    def from_settings(cls, settings: Settings) -> "PlanParserClient":
        return cls(
            base_url=settings.plan_parser_base_url,
            api_key=settings.plan_parser_api_key,
            model=settings.plan_parser_model,
            timeout_seconds=settings.plan_parser_timeout_seconds,
        )

    def parse_metadata(self, query: str) -> dict[str, Any]:
        if not self.base_url or not self.api_key or not self.model:
            raise PlanParseError("PLAN_PARSER_BASE_URL, PLAN_PARSER_API_KEY or PLAN_PARSER_MODEL is missing")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": metadata_parser_system_prompt()},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            data = self.chat_completion(payload)
        except PlanParseError as exc:
            if "response_format" not in str(exc):
                raise
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            data = self.chat_completion(fallback_payload)
        return validate_metadata_parse(chat_completion_content(data), query)

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
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


def metadata_parser_system_prompt() -> str:
    return """You are a metadata query parser.
Parse the user query into JSON only. Do not answer the question.

Schema:
{
  "router": "metadata",
  "intent": "lookup|list|count|null",
  "return_field": "author|year|venue|title|null",
  "filters": [{"field":"author|year|venue|title","op":"=|in|contains|between","negated":false|true,"value":""}],
  "raw_query": ""
}

Rules:
- Use "lookup" when the query asks for a metadata field value.
- Use "list" when the query asks for papers matching metadata conditions.
- Use "count" when the query asks for the number of papers matching metadata conditions.
- Use "negated": true for the filter negated.
- "not on arXiv" must be:
  {"field":"venue","op":"contains","value":"arXiv","negated":true}
- Use a numeric list like [2015, 2019] for "value" of "between".
- raw_query must equal the input query
- if uncertain, use null
"""


def chat_completion_content(data: dict[str, Any]) -> str:
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


def validate_metadata_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Plan parser returned invalid JSON: {exc}") from exc
    else:
        payload = dict(content)
    if not isinstance(payload, dict):
        raise PlanParseError("Plan parser JSON root must be an object")
    if payload.get("router") != "metadata":
        raise PlanParseError("Metadata parser router must be metadata")
    intent = payload.get("intent")
    if intent not in METADATA_INTENTS:
        raise PlanParseError(f"Invalid metadata intent: {intent}")
    return_field = payload.get("return_field")
    if return_field is None and "target_field" in payload:
        return_field = payload.get("target_field")
    if return_field == "null":
        return_field = None
    if return_field not in METADATA_RETURN_FIELDS:
        raise PlanParseError(f"Invalid metadata return_field: {return_field}")
    filters = payload.get("filters")
    if not isinstance(filters, list):
        raise PlanParseError("Metadata filters must be a list")
    normalized_filters = [validate_metadata_filter(filter_item) for filter_item in filters]
    raw_query = payload.get("raw_query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        raw_query = fallback_query
    return {
        "router": "metadata",
        "intent": intent,
        "return_field": return_field,
        "filters": normalized_filters,
        "raw_query": raw_query,
    }


def validate_metadata_filter(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanParseError("Metadata filter must be an object")
    field = value.get("field")
    if field not in METADATA_FILTER_FIELDS:
        raise PlanParseError(f"Invalid metadata filter field: {field}")
    op = value.get("op")
    if op not in METADATA_FILTER_OPS:
        raise PlanParseError(f"Invalid metadata filter op: {op}")
    if "value" not in value:
        raise PlanParseError("Metadata filter missing value")
    negated = value.get("negated")
    if not isinstance(negated, bool):
        raise PlanParseError("Metadata filter negated must be true or false")
    normalized_value = normalize_metadata_filter_value(op, value.get("value"))
    return {
        "field": field,
        "op": op,
        "value": normalized_value,
        "negated": negated,
    }


def normalize_metadata_filter_value(op: str, value: Any) -> Any:
    if op != "between":
        return value
    if isinstance(value, list):
        normalized: list[int] = []
        for item in value:
            try:
                normalized.append(int(str(item).strip()))
            except (TypeError, ValueError):
                raise PlanParseError("Metadata between filter values must be numeric")
        if len(normalized) >= 2:
            return normalized[:2]
        raise PlanParseError("Metadata between filter requires two numeric values")
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("to", "-").split("-") if part.strip()]
        if len(parts) >= 2:
            try:
                return [int(parts[0]), int(parts[1])]
            except ValueError as exc:
                raise PlanParseError("Metadata between filter values must be numeric") from exc
    raise PlanParseError("Metadata between filter requires a numeric range")
