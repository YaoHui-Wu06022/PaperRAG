from __future__ import annotations

import json
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.routing import RetrieveRequest, RetrieveTask, RouteIntent, classify_retrieve
from paper_rag.routing.jev import JevClient


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int = -1) -> bytes:
        return self.payload

    def getcode(self) -> int:
        return self.status


def test_jev_classifies_only_retrieve_tasks(tmp_path: Path):
    settings = Settings.load(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "jev_api_key": "test-secret", "jev_base_url": "https://jev.test/v1/systemone"})
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse({"result": {"retrieve_task": {"choice": "reason"}, "confidence": 0.73}})

    decision = JevClient(settings, opener=opener, sleeper=lambda _seconds: None).classify(RetrieveRequest("这个方法为什么有效"))
    assert decision.route_intent is RouteIntent.RETRIEVE
    assert decision.task is RetrieveTask.REASON
    assert decision.confidence == 0.73
    payload = json.loads(calls[0][0].data.decode("utf-8"))
    assert payload["state"]["route_intent"] == "retrieve"
    assert "library_search" not in json.dumps(payload, ensure_ascii=False)


def test_jev_failure_uses_retrieve_rule_fallback(tmp_path: Path):
    settings = Settings.load(tmp_path)
    decision = classify_retrieve(settings, "比较两篇论文的方法")
    assert decision.route_intent is RouteIntent.RETRIEVE
    assert decision.task is RetrieveTask.COMPARISON
    assert decision.fallback_used is True


def test_mcp_registers_one_body_rag_tool():
    from paper_rag.mcp import server
    from paper_rag.mcp.toolsets import CORE_TOOLS

    assert "library_retrieve" in server._registered_tool_names()
    assert "library_context" not in server._registered_tool_names()
    assert "library_retrieve" in CORE_TOOLS



def test_jev_timeout_waits_then_retry_success_even_when_retry_count_zero(tmp_path: Path):
    from dataclasses import replace

    settings = replace(Settings.load(tmp_path), jev_enabled=True, jev_api_key="test-secret", jev_retry_count=0)
    events = []

    def opener(_request, timeout):
        events.append(("request", timeout))
        if len(events) == 1:
            raise TimeoutError("The read operation timed out")
        return FakeResponse({"result": {"retrieve_task": "fact", "confidence": 0.81}})

    client = JevClient(settings, opener=opener, sleeper=lambda seconds: events.append(("wait", seconds)))
    decision = classify_retrieve(settings, "BERT 的预训练任务有哪些", client=client)

    assert events == [("request", settings.jev_timeout_seconds), ("wait", 1), ("request", settings.jev_timeout_seconds)]
    assert decision.provider == "jev"
    assert not decision.fallback_used
    assert decision.confidence == 0.81


def test_jev_rules_fallback_only_after_waited_retries_exhausted(tmp_path: Path, monkeypatch):
    from dataclasses import replace
    from urllib.error import URLError
    from paper_rag.routing import router

    settings = replace(Settings.load(tmp_path), jev_enabled=True, jev_api_key="test-secret", jev_retry_count=2)
    events = []

    def opener(_request, timeout):
        events.append(("request", timeout))
        raise URLError(TimeoutError("The read operation timed out"))

    original_rules = router.classify_by_rules

    def rules(request):
        events.append(("rules", request.query))
        return original_rules(request)

    monkeypatch.setattr(router, "classify_by_rules", rules)
    client = JevClient(settings, opener=opener, sleeper=lambda seconds: events.append(("wait", seconds)))
    decision = classify_retrieve(settings, "PagedAttention 如何减少显存浪费", client=client)

    assert events == [
        ("request", settings.jev_timeout_seconds), ("wait", 1),
        ("request", settings.jev_timeout_seconds), ("wait", 2),
        ("request", settings.jev_timeout_seconds), ("rules", "PagedAttention 如何减少显存浪费"),
    ]
    assert decision.provider == "rules"
    assert decision.fallback_used
    assert "请求重试失败" in decision.warning
