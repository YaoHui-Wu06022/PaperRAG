from __future__ import annotations

import json
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.query.classifier import classify_query
from paper_rag.query.jev import JevClient
from paper_rag.query.rules import classify_by_rules, contains_arxiv_reference
from paper_rag.query.schemas import QueryIntent, QueryRequest
from paper_rag.query.service import query_papers


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


def test_settings_loads_jev_key_without_exposing_key(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "JEV_API_KEY=test-secret\nJEV_MODEL=jev-1.13.0\n",
        encoding="utf-8",
    )
    settings = Settings.load(tmp_path)

    assert settings.jev_api_key == "test-secret"
    assert settings.jev_model == "jev-1.13.0"
    assert "test-secret" not in repr(settings)


def test_query_prefers_jev_even_for_a_clear_query(tmp_path: Path):
    settings = Settings.load(tmp_path)

    calls = []

    class FakeClient:
        def classify(self, *args, **kwargs):
            calls.append((args, kwargs))
            return type(
                "Decision",
                (),
                {
                    "intent": QueryIntent.PAPER_COMPARISON,
                    "needs_clarification": False,
                    "provider": "jev",
                    "candidate_handlers": (QueryIntent.PAPER_COMPARISON.value,),
                    "matched_intents": (QueryIntent.PAPER_COMPARISON,),
                    "confidence": None,
                },
            )()

    decision = classify_query(
        settings,
        QueryRequest("比较两篇论文的方法差异"),
        client=FakeClient(),
    )

    assert decision.intent == QueryIntent.PAPER_COMPARISON
    assert decision.provider == "jev"
    assert calls


def test_jev_request_and_choice_parsing(tmp_path: Path):
    settings = Settings.load(tmp_path)
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "jev_api_key": "test-secret",
            "jev_base_url": "https://jev.test/v1/systemone",
        }
    )
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(
            {
                "result": {
                    "intent": {"choice": "paper_summary"},
                    "needs_clarification": {"probability": 0.1},
                }
            }
        )

    client = JevClient(settings, opener=opener, sleeper=lambda _seconds: None)
    decision = client.classify("总结 Attention Is All You Need")

    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://jev.test/v1/systemone"
    assert request.headers["Authorization"] == "Bearer test-secret"
    assert request.headers.get("Idempotency-key")
    assert payload["model"] == "jev-1.13.0"
    assert payload["questions"]["intent"]["type"] == "choice"
    assert timeout == settings.jev_timeout_seconds
    assert decision.intent == QueryIntent.PAPER_SUMMARY
    assert decision.needs_clarification is False


def test_query_falls_back_to_clarify_when_jev_unavailable(tmp_path: Path):
    result = query_papers(Settings.load(tmp_path), "帮我处理一下这篇资料")

    assert result.decision.intent == QueryIntent.CLARIFY
    assert result.decision.fallback_used is True
    assert result.to_dict()["read_only"] is True


def test_operational_terms_are_not_knowledge_rules(tmp_path: Path):
    settings = Settings.load(tmp_path)

    # 操作请求由 Agent 选择专用 MCP 工具，不在 paper_query 的本地规则中拦截。
    assert classify_by_rules("查看 MinerU 解析状态") is None
    assert classify_by_rules("下载 1706.03762") is None

    # Jev 未配置时，误传给 paper_query 的操作请求安全回退为澄清，不执行写操作。
    result = query_papers(settings, "下载 1706.03762")
    assert result.decision.intent == QueryIntent.CLARIFY
    assert result.decision.fallback_used is True


def test_query_returns_local_metadata_and_content(tmp_path: Path):
    settings = Settings.load(tmp_path)
    source = settings.arxiv_data_dir / "1706.03762"
    (source / "mineru").mkdir(parents=True)
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "base_id": "1706.03762",
                "canonical_id": "1706.03762v7",
                "title": "Attention Is All You Need",
                "authors": ["Alice"],
                "abstract": "Transformer attention paper.",
                "categories": ["cs.CL"],
            }
        ),
        encoding="utf-8",
    )
    (source / "mineru" / "full.md").write_text(
        "# Attention\n\nThe Transformer uses attention.\n",
        encoding="utf-8",
    )

    result = query_papers(settings, "1706.03762v7 如何实现这个方法")

    assert result.decision.intent == QueryIntent.PAPER_CONTENT
    assert result.items[0]["canonical_id"] == "1706.03762v7"
    assert "attention" in result.context[0]["text"].casefold()


def test_mcp_registers_query_tool():
    from paper_rag.mcp import server

    assert "paper_query" in server._registered_tool_names()
    assert "paper_query" in server.TOOLSETS["core"]


def test_rules_cover_common_chinese_intents_and_delegate_other_languages():
    assert classify_by_rules("找相关论文").intent == QueryIntent.PAPER_DISCOVERY
    assert classify_by_rules("总结这篇论文的主要贡献").intent == QueryIntent.PAPER_SUMMARY
    assert classify_by_rules("比较两篇论文的方法差异").intent == QueryIntent.PAPER_COMPARISON
    assert classify_by_rules("这个方法如何实现").intent == QueryIntent.PAPER_CONTENT
    assert classify_by_rules("这篇论文的作者是谁").intent == QueryIntent.METADATA_LOOKUP
    assert classify_by_rules("summarize the LoRA paper") is None
    assert classify_by_rules("讲了什么") is None
    assert classify_by_rules("找") is None


def test_arxiv_reference_regex_accepts_ids_and_rejects_partial_numbers():
    accepted = (
        "1706.03762",
        "1706.03762v7",
        "https://arxiv.org/pdf/1706.03762v7.pdf",
        "hep-th/9901001",
        "https://arxiv.org/abs/hep-th/9901001v2",
    )
    rejected = (
        "11706.03762",
        "1706.03762v0",
        "1706.03762.1",
        "12345",
    )
    assert all(contains_arxiv_reference(value) for value in accepted)
    assert not any(contains_arxiv_reference(value) for value in rejected)
