"""DashScope OpenAI 兼容 Embedding 客户端。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from paper_rag.config import Settings


class EmbeddingError(RuntimeError):
    pass


class DashScopeEmbeddingClient:
    def __init__(self, settings: Settings, *, opener: Any | None = None, sleeper: Any = time.sleep):
        self.settings = settings
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if not self.settings.dashscope_api_key:
            raise EmbeddingError("DASHSCOPE_API_KEY 未配置")
        if not inputs:
            return []
        if len(inputs) > self.settings.embedding_batch_size:
            raise EmbeddingError("Embedding 批次超过配置上限")
        payload = {"model": self.settings.embedding_model, "input": inputs, "dimensions": self.settings.embedding_dimensions, "encoding_format": "float"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(f"{self.settings.dashscope_base_url.rstrip('/')}/embeddings", data=body, method="POST", headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}", "Content-Type": "application/json", "Accept": "application/json"})
        last: Exception | None = None
        for attempt in range(self.settings.embedding_retry_count + 1):
            try:
                with self.opener(request, timeout=self.settings.embedding_timeout_seconds) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    raw = response.read().decode("utf-8", errors="replace")
                    if status < 200 or status >= 300:
                        if status in {429, 500, 502, 503, 504}:
                            raise _RetryableEmbeddingError(f"HTTP {status}")
                        raise EmbeddingError(f"Embedding 请求失败：HTTP {status}")
                    data = json.loads(raw)
                    vectors = [item.get("embedding") for item in sorted(data.get("data", []), key=lambda item: item.get("index", 0))]
                    if len(vectors) != len(inputs) or any(not isinstance(v, list) or len(v) != self.settings.embedding_dimensions for v in vectors):
                        raise EmbeddingError("Embedding 返回数量或维度不匹配")
                    return [[float(value) for value in vector] for vector in vectors]
            except _RetryableEmbeddingError as exc:
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.settings.embedding_retry_count:
                self.sleeper(2 ** attempt)
        raise EmbeddingError(f"Embedding 请求重试失败：{last}") from last


class _RetryableEmbeddingError(EmbeddingError):
    pass


__all__ = ["DashScopeEmbeddingClient", "EmbeddingError"]
