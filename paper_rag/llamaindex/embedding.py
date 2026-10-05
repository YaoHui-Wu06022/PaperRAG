"""DashScope Embedding 客户端和 LlamaIndex 适配器。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from paper_rag.config import Settings


class EmbeddingError(RuntimeError):
    """Embedding 请求或响应校验失败。"""


class DashScopeEmbeddingClient:
    """复用现有 DashScope HTTP 契约的批量客户端。"""

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
        request = urllib.request.Request(f"{self.settings.dashscope_base_url.rstrip('/')}/embeddings", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST", headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}", "Content-Type": "application/json", "Accept": "application/json"})
        last: Exception | None = None
        for attempt in range(self.settings.embedding_retry_count + 1):
            try:
                with self.opener(request, timeout=self.settings.embedding_timeout_seconds) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    raw = response.read().decode("utf-8", errors="replace")
                    if status < 200 or status >= 300:
                        if status in {429, 500, 502, 503, 504}:
                            raise OSError(f"HTTP {status}")
                        raise EmbeddingError(f"Embedding 请求失败：HTTP {status} {raw[:500]}")
                    data = json.loads(raw)
                    vectors = [item.get("embedding") for item in sorted(data.get("data", []), key=lambda item: item.get("index", 0))]
                    if len(vectors) != len(inputs) or any(not isinstance(vector, list) or len(vector) != self.settings.embedding_dimensions for vector in vectors):
                        raise EmbeddingError("Embedding 返回数量或维度不匹配")
                    return [[float(value) for value in vector] for vector in vectors]
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except OSError:
                    pass
                if exc.code in {429, 500, 502, 503, 504}:
                    last = OSError(f"HTTP {exc.code}")
                else:
                    raise EmbeddingError(f"Embedding 请求失败：HTTP {exc.code} {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.settings.embedding_retry_count:
                self.sleeper(2 ** attempt)
        raise EmbeddingError(f"Embedding 请求重试失败：{last}") from last


class DashScopeEmbedding(BaseEmbedding):
    """把项目 DashScope 客户端适配为 LlamaIndex Embedding。"""

    _client: DashScopeEmbeddingClient = PrivateAttr()

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(model_name=settings.embedding_model, embed_batch_size=settings.embedding_batch_size, **kwargs)
        self._client = DashScopeEmbeddingClient(settings)

    @classmethod
    def class_name(cls) -> str:
        return "dashscope_embedding"

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._client.embed([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._client.embed([query])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


__all__ = ["DashScopeEmbedding", "DashScopeEmbeddingClient", "EmbeddingError"]
