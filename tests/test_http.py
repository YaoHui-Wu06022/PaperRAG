from __future__ import annotations

import json

from paper_rag.http import JsonHttpClient


class _Response:
    def __init__(self, payload: dict, status: int = 200):
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int = -1) -> bytes:
        return self._payload

    def getcode(self) -> int:
        return self.status


def test_json_http_client_retries_retryable_status():
    calls = []

    def opener(_request, timeout):
        assert timeout == 1
        calls.append(True)
        return _Response({"ok": True} if len(calls) == 2 else {"error": "busy"}, 200 if len(calls) == 2 else 503)

    result = JsonHttpClient(opener=opener, sleeper=lambda _seconds: None).post_json(
        "https://example.test",
        {"query": "attention"},
        headers={"Accept": "application/json"},
        timeout=1,
        retries=1,
        error_prefix="test ",
    )

    assert result == {"ok": True}
    assert len(calls) == 2
