"""embedding 本地缓存，减少重复向量化请求。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """把 embedding 向量按稳定 key 缓存在本地 jsonl 文件中。"""

    def __init__(self, path: Path):
        self.path = path
        self._vectors: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def get(self, key: str) -> list[float] | None:
        """按 cache key 读取已缓存向量。"""
        return self._vectors.get(key)

    def set(self, key: str, vector: list[float]) -> None:
        """写入向量并标记缓存需要落盘。"""
        self._vectors[key] = vector
        self._dirty = True

    def save(self) -> None:
        """如果缓存有更新，将全部向量按 key 排序写回 jsonl。"""
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
        """启动时读取已有 jsonl 缓存，坏行直接交给 json 解析抛错。"""
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
    """用模型、维度和文本生成稳定 cache key。"""
    payload = json.dumps(
        {"model": model, "dimensions": dimensions, "text": text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """先查本地缓存，只对未命中文本批量请求 embedding。"""

    def __init__(self, client, cache: EmbeddingCache, *, model: str, dimensions: int, batch_size: int):
        self.client = client
        self.cache = cache
        self.model = model
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """返回与输入 texts 顺序一致的向量列表。"""
        output: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = embedding_cache_key(self.model, self.dimensions, text)
            cached = self.cache.get(key)
            if cached is None:
                misses.append((index, key, text))
            else:
                output[index] = cached
        # 只对未命中的文本分批请求 embedding，避免破坏输出顺序。
        for start in range(0, len(misses), self.batch_size):
            batch = misses[start:start + self.batch_size]
            vectors = self.client.embed_texts([text for _, _, text in batch])
            for (index, key, _), vector in zip(batch, vectors):
                self.cache.set(key, vector)
                output[index] = vector
        self.cache.save()
        return [vector for vector in output if vector is not None]
