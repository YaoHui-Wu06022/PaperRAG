from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingClient:
    base_url: str
    api_key: str | None
    model: str
    dimensions: int
    timeout: int = 60

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise EmbeddingError("EMBEDDING_API_KEY is missing in .env")
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Paper_RAG/0.1 retrieval",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise EmbeddingError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(str(exc)) from exc
        return parse_embedding_response(data, len(texts))


def parse_embedding_response(data: dict[str, Any], expected_count: int) -> list[list[float]]:
    rows = data.get("data")
    if not isinstance(rows, list):
        raise EmbeddingError("Embedding response missing data list")
    indexed: list[tuple[int, list[float]]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            continue
        vector = [float(value) for value in embedding]
        index = row.get("index")
        indexed.append((index if isinstance(index, int) else position, vector))
    indexed.sort(key=lambda item: item[0])
    vectors = [vector for _, vector in indexed]
    if len(vectors) != expected_count:
        raise EmbeddingError(f"Expected {expected_count} embedding(s), got {len(vectors)}")
    return vectors

