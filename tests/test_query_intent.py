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

