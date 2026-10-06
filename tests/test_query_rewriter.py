from __future__ import annotations

import json
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.llamaindex.query_rewriter import QueryRewriterClient, QueryRewriterError


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


def test_query_rewriter_parses_only_core_terms(tmp_path: Path):
    settings = Settings.load(tmp_path)
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "query_rewriter_enabled": True,
            "query_rewriter_api_key": "test-key",
            "query_rewriter_base_url": "https://rewriter.test/v1/systemone",
        }
    )
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                    "content": json.dumps(
                                {"core_terms": ["注意力机制"]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

    result = QueryRewriterClient(settings, opener=opener, sleeper=lambda _seconds: None).rewrite("注意力机制")

    assert result.core_terms == ("注意力机制",)
    payload = json.loads(calls[0][0].data.decode("utf-8"))
    assert payload["messages"][1]["content"] == "注意力机制"
    assert payload["messages"][0]["role"] == "system"
    assert payload["response_format"] == {"type": "json_object"}


def test_query_rewriter_rejects_empty_result(tmp_path: Path):
    settings = Settings.load(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "query_rewriter_enabled": True, "query_rewriter_api_key": "test-key"})

    client = QueryRewriterClient(
        settings,
        opener=lambda request, timeout: FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"core_terms": []}'
                        }
                    }
                ]
            }
        ),
        sleeper=lambda _seconds: None,
    )

    try:
        client.rewrite("注意力机制")
    except QueryRewriterError as exc:
        assert "没有返回" in str(exc)
    else:
        raise AssertionError("empty rewrite result should fail")


def test_query_rewriter_rejects_phrases_and_extra_fields(tmp_path: Path):
    settings = Settings.load(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "query_rewriter_enabled": True, "query_rewriter_api_key": "test-key"})
    client = QueryRewriterClient(
        settings,
        opener=lambda request, timeout: FakeResponse(
            {"choices": [{"message": {"content": '{"phrases": ["注意力机制"]}'}}]}
        ),
        sleeper=lambda _seconds: None,
    )

    try:
        client.rewrite("什么是注意力机制？")
    except QueryRewriterError as exc:
        assert "只能返回 core_terms" in str(exc)
    else:
        raise AssertionError("extra fields should fail")
