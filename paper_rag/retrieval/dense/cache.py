from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self._vectors: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def get(self, key: str) -> list[float] | None:
        return self._vectors.get(key)

    def set(self, key: str, vector: list[float]) -> None:
        self._vectors[key] = vector
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"key": key, "vector": vector}, ensure_ascii=False)
            for key, vector in sorted(self._vectors.items())
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._dirty = False

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            vector = row.get("vector")
            if isinstance(key, str) and isinstance(vector, list):
                self._vectors[key] = [float(value) for value in vector]


def embedding_cache_key(model: str, dimensions: int, text: str) -> str:
    payload = json.dumps(
        {"model": model, "dimensions": dimensions, "text": text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedEmbedder:
    def __init__(self, client, cache: EmbeddingCache, *, model: str, dimensions: int, batch_size: int):
        self.client = client
        self.cache = cache
        self.model = model
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        output: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = embedding_cache_key(self.model, self.dimensions, text)
            cached = self.cache.get(key)
            if cached is None:
                misses.append((index, key, text))
            else:
                output[index] = cached
        for start in range(0, len(misses), self.batch_size):
            batch = misses[start:start + self.batch_size]
            vectors = self.client.embed_texts([text for _, _, text in batch])
            for (index, key, _), vector in zip(batch, vectors):
                self.cache.set(key, vector)
                output[index] = vector
        self.cache.save()
        return [vector for vector in output if vector is not None]

