"""Jev 有限集合决策客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from paper_rag.retrieval.routes.common.jev_policy import POLICY_VERSION, decision_questions, rule_fallback


class JevError(RuntimeError):
    """Jev 请求或响应不可用，并携带稳定的降级原因。"""

    def __init__(self, message: str, *, code: str = "jev_unavailable") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class JevDecision:
    route: str
    metadata_intent: str | None = None
    reference_intent: str | None = None
    reference_side: str | None = None
    content_intent: str | None = None
    needs_synthesis: bool = False
    complexity: int = 1
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    backend: str = "jev"
    fallback: bool = False
    policy_version: str = POLICY_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "JevDecision":
        raw_answers = payload.get("answers")
        if isinstance(raw_answers, list):
            answers = {}
            for item in raw_answers:
                if isinstance(item, dict):
                    key = item.get("question") or item.get("id") or item.get("name")
                    if key:
                        answers[str(key)] = item.get("answer", item.get("value", item))
        else:
            answers = raw_answers if isinstance(raw_answers, dict) else payload
        if isinstance(answers.get("answers"), dict):
            answers = answers["answers"]
        route_raw = answers.get("route")
        route = str(_answer_value(route_raw) or "unclear").strip().lower()
        if route not in {"metadata", "reference", "content", "unclear"}:
            raise JevError(f"Jev 返回了未知 route：{route}", code="jev_invalid_response")
        route_meta = route_raw if isinstance(route_raw, dict) else {}
        confidence = _answer_value(answers.get("confidence") or payload.get("confidence") or route_meta.get("confidence"))
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        probabilities = answers.get("probabilities") or payload.get("probabilities") or route_meta.get("probabilities") or {}
        if not isinstance(probabilities, dict):
            probabilities = {}
        normalized_probabilities: dict[str, float] = {}
        for key, value in probabilities.items():
            try:
                normalized_probabilities[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        # Some Jev responses omit a top-level confidence and only return the
        # choice distribution.  Treat the highest route probability as the
        # decision confidence so the same low-confidence contract applies to
        # both response shapes.
        if confidence is None:
            route_probabilities = [
                value
                for key, value in normalized_probabilities.items()
                if key in {"metadata", "reference", "content", "unclear"}
            ]
            if route_probabilities:
                confidence = max(route_probabilities)
        complexity = _answer_value(answers.get("complexity", 1))
        try:
            complexity = max(1, min(5, int(float(complexity))))
        except (TypeError, ValueError):
            complexity = 1
        return cls(
            route=route,
            metadata_intent=_nullable(_answer_value(answers.get("metadata_intent"))),
            reference_intent=_nullable(_answer_value(answers.get("reference_intent"))),
            reference_side=_nullable(_answer_value(answers.get("reference_side"))),
            content_intent=_nullable(_answer_value(answers.get("content_intent"))),
            needs_synthesis=_as_bool(_answer_value(answers.get("needs_synthesis", False))),
            complexity=complexity,
            confidence=confidence,
            probabilities=normalized_probabilities,
        )


def _nullable(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return None if text in {"", "null", "none"} else text


def _as_bool(value: Any) -> bool:
    """Normalize Jev choice answers such as ``true``/``false``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().casefold() in {"true", "yes", "1"}


def _answer_value(value: Any) -> Any:
    """兼容 Jev 将 answer 包成 {value, confidence, probabilities} 的响应。"""
    if isinstance(value, dict):
        return value.get("value", value.get("choice", value.get("answer")))
    return value


class JevDecisionClient:
    def __init__(self, base_url: str, api_key: str, *, model: str = "jev-1.13.0", timeout_seconds: int = 20, min_confidence: float = 0.55):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.min_confidence = min_confidence

    @classmethod
    def from_settings(cls, settings):
        if not getattr(settings, "jev_base_url", "") or not getattr(settings, "jev_api_key", None):
            raise JevError("Jev 未配置", code="jev_unavailable")
        return cls(settings.jev_base_url, settings.jev_api_key, model=getattr(settings, "jev_model", "jev-1.13.0"), timeout_seconds=getattr(settings, "jev_timeout_seconds", 20), min_confidence=getattr(settings, "jev_min_confidence", 0.55))

    def decide(self, query: str) -> JevDecision:
        # Jev systemone 接口按 state + questions contract 接收请求，不使用 chat model 字段。
        body = {"state": str(query), "questions": decision_questions()}
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise JevError("Jev 请求失败", code="jev_unavailable") from exc
        decision = JevDecision.from_payload(payload)
        if decision.confidence is not None and decision.confidence < self.min_confidence:
            raise JevError("Jev 置信度低于阈值", code="jev_low_confidence")
        return decision


def fallback_decision(query: str, reason: str = "jev_unavailable") -> JevDecision:
    values = rule_fallback(query)
    return JevDecision(**values, backend="rules", fallback=True, confidence=None, probabilities={}, policy_version=POLICY_VERSION)
